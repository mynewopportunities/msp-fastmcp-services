#!/usr/bin/env python3
"""
FastMCP-based RAG Knowledge Management System for MSP Operations
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import asyncpg
import numpy as np
import openai
from fastmcp import FastMCP
from mcp.types import Resource, Tool, TextContent
from sentence_transformers import SentenceTransformer
import redis.asyncio as redis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FastMCPRAGSystem:
    """FastMCP-based RAG System for MSP Knowledge Management"""
    
    def __init__(self):
        self.app = FastMCP("MSP RAG System")
        self.db_pool = None
        self.redis_client = None
        self.embedding_model = None
        self.openai_client = None
        self.setup_tools()
        self.setup_resources()
        
    async def initialize(self):
        """Initialize database connections and models"""
        try:
            DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://user:pass@localhost:5432/msp_db')
            self.db_pool = await asyncpg.create_pool(DATABASE_URL)
            
            REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
            self.redis_client = await redis.from_url(REDIS_URL)
            
            model_name = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
            self.embedding_model = SentenceTransformer(model_name)
            
            openai_key = os.getenv('OPENAI_API_KEY')
            if openai_key:
                self.openai_client = openai.AsyncOpenAI(api_key=openai_key)
            
            logger.info("FastMCP RAG System initialized successfully")
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise
    
    def setup_tools(self):
        """Setup FastMCP tools for RAG operations"""
        
        @self.app.tool()
        async def search_knowledge_base(
            query: str,
            category: Optional[str] = None,
            limit: int = 5,
            similarity_threshold: float = 0.7
        ) -> List[Dict[str, Any]]:
            """Search the RAG knowledge base for relevant documents"""
            try:
                query_embedding = self.embedding_model.encode([query])
                
                sql_query = """
                    SELECT d.id, d.title, d.content, d.category, d.metadata,
                           d.created_at, d.updated_at, d.embedding,
                           1 - (d.embedding <=> $1::vector) as similarity_score
                    FROM knowledge_documents d
                    WHERE ($2::text IS NULL OR d.category = $2)
                    ORDER BY d.embedding <=> $1::vector
                    LIMIT $3
                """
                
                async with self.db_pool.acquire() as conn:
                    rows = await conn.fetch(
                        sql_query,
                        query_embedding[0].tolist(),
                        category,
                        limit
                    )
                
                results = []
                for row in rows:
                    if row['similarity_score'] >= similarity_threshold:
                        result = {
                            'id': row['id'],
                            'title': row['title'],
                            'content': row['content'][:500] + '...' if len(row['content']) > 500 else row['content'],
                            'category': row['category'],
                            'similarity_score': float(row['similarity_score']),
                            'metadata': json.loads(row['metadata'] or '{}'),
                            'created_at': row['created_at'].isoformat(),
                            'updated_at': row['updated_at'].isoformat()
                        }
                        results.append(result)
                
                return results
                
            except Exception as e:
                logger.error(f"Knowledge base search failed: {e}")
                return []
        
        @self.app.tool()
        async def add_document(
            title: str,
            content: str,
            category: str,
            metadata: Optional[Dict[str, Any]] = None,
            source: Optional[str] = None
        ) -> Dict[str, Any]:
            """Add a new document to the knowledge base"""
            try:
                embedding = self.embedding_model.encode([content])[0]
                
                doc_metadata = metadata or {}
                if source:
                    doc_metadata['source'] = source
                doc_metadata['indexed_at'] = datetime.utcnow().isoformat()
                
                sql_query = """
                    INSERT INTO knowledge_documents 
                    (title, content, category, metadata, embedding, source, created_at, updated_at)
                    VALUES ($1, $2, $3, $4, $5, $6, NOW(), NOW())
                    RETURNING id, created_at
                """
                
                async with self.db_pool.acquire() as conn:
                    row = await conn.fetchrow(
                        sql_query,
                        title,
                        content,
                        category,
                        json.dumps(doc_metadata),
                        embedding.tolist(),
                        source
                    )
                
                return {
                    'id': row['id'],
                    'title': title,
                    'category': category,
                    'created_at': row['created_at'].isoformat(),
                    'status': 'success',
                    'message': 'Document added successfully'
                }
                
            except Exception as e:
                logger.error(f"Document addition failed: {e}")
                return {
                    'status': 'error',
                    'message': f'Failed to add document: {str(e)}'
                }
        
        @self.app.tool()
        async def generate_answer(
            question: str,
            context_documents: Optional[List[str]] = None,
            max_tokens: int = 500
        ) -> Dict[str, Any]:
            """Generate an AI answer using RAG context"""
            try:
                if not self.openai_client:
                    return {
                        'status': 'error',
                        'message': 'OpenAI API not configured'
                    }
                
                context = ""
                sources = []
                
                if context_documents:
                    sql_query = """
                        SELECT id, title, content, category
                        FROM knowledge_documents
                        WHERE id = ANY($1)
                    """
                    
                    async with self.db_pool.acquire() as conn:
                        rows = await conn.fetch(sql_query, context_documents)
                    
                    for row in rows:
                        context += f"\n\n{row['title']}\n{row['content']}"
                        sources.append({
                            'id': row['id'],
                            'title': row['title'],
                            'category': row['category']
                        })
                else:
                    search_results = await search_knowledge_base(
                        question, limit=3, similarity_threshold=0.6
                    )
                    
                    for doc in search_results:
                        context += f"\n\n{doc['title']}\n{doc['content']}"
                        sources.append({
                            'id': doc['id'],
                            'title': doc['title'],
                            'category': doc['category'],
                            'similarity_score': doc['similarity_score']
                        })
                
                system_prompt = """
                You are an expert MSP assistant with access to the company's knowledge base.
                Provide accurate answers based on the provided context.
                Always cite your sources when possible.
                """
                
                user_prompt = f"""
                Question: {question}
                
                Context:
                {context}
                
                Please provide a comprehensive answer based on the context above.
                """
                
                response = await self.openai_client.chat.completions.create(
                    model=os.getenv('OPENAI_MODEL', 'gpt-4'),
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=0.7
                )
                
                answer = response.choices[0].message.content
                
                return {
                    'answer': answer,
                    'sources': sources,
                    'question': question,
                    'timestamp': datetime.utcnow().isoformat(),
                    'status': 'success'
                }
                
            except Exception as e:
                logger.error(f"Answer generation failed: {e}")
                return {
                    'status': 'error',
                    'message': f'Failed to generate answer: {str(e)}'
                }
    
    def setup_resources(self):
        """Setup FastMCP resources for knowledge base access"""
        
        @self.app.resource("knowledge-categories")
        async def get_knowledge_categories() -> List[Resource]:
            """Get all available knowledge base categories"""
            try:
                async with self.db_pool.acquire() as conn:
                    rows = await conn.fetch("""
                        SELECT category, COUNT(*) as document_count
                        FROM knowledge_documents
                        GROUP BY category
                        ORDER BY category
                    """)
                
                categories = []
                for row in rows:
                    categories.append(Resource(
                        uri=f"knowledge://category/{row['category']}",
                        name=f"Category: {row['category']}",
                        description=f"Knowledge base category with {row['document_count']} documents",
                        mimeType="application/json"
                    ))
                
                return categories
                
            except Exception as e:
                logger.error(f"Failed to get categories: {e}")
                return []
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.db_pool:
                await self.db_pool.close()
            if self.redis_client:
                await self.redis_client.close()
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

rag_system = FastMCPRAGSystem()

async def main():
    """Main entry point for FastMCP RAG System"""
    try:
        await rag_system.initialize()
        async with rag_system.app.run_server() as server:
            await server.run()
    except KeyboardInterrupt:
        logger.info("Shutting down RAG system...")
    except Exception as e:
        logger.error(f"RAG system error: {e}")
    finally:
        await rag_system.cleanup()

if __name__ == "__main__":
    asyncio.run(main())