#!/usr/bin/env python3
"""
FastMCP-based Intelligent Chatbot for MSP Operations
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from enum import Enum

import asyncpg
import openai
from fastmcp import FastMCP
from mcp.types import Resource, Tool, TextContent
import redis.asyncio as redis
from sentence_transformers import SentenceTransformer
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConversationState(Enum):
    GREETING = "greeting"
    QUALIFYING = "qualifying"
    ASSESSING_NEEDS = "assessing_needs"
    PROVIDING_SOLUTION = "providing_solution"
    SCHEDULING = "scheduling"
    ESCALATING = "escalating"
    CLOSING = "closing"

class IntentType(Enum):
    GENERAL_INQUIRY = "general_inquiry"
    TECHNICAL_SUPPORT = "technical_support" 
    SALES_INQUIRY = "sales_inquiry"
    BILLING_QUESTION = "billing_question"
    APPOINTMENT_SCHEDULING = "appointment_scheduling"
    COMPLAINT = "complaint"
    EMERGENCY = "emergency"

class FastMCPIntelligentChatbot:
    """FastMCP-based Intelligent Chatbot"""
    
    def __init__(self):
        self.app = FastMCP("MSP Intelligent Chatbot")
        self.db_pool = None
        self.redis_client = None
        self.openai_client = None
        self.embedding_model = None
        self.setup_tools()
        self.setup_resources()
        
    async def initialize(self):
        """Initialize services"""
        try:
            # Database connection
            DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://user:pass@localhost:5432/msp_db')
            self.db_pool = await asyncpg.create_pool(DATABASE_URL)
            
            # Redis connection
            REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
            self.redis_client = await redis.from_url(REDIS_URL)
            
            # OpenAI client
            openai_key = os.getenv('OPENAI_API_KEY')
            if openai_key:
                self.openai_client = openai.AsyncOpenAI(api_key=openai_key)
            
            # Embedding model for intent classification
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            logger.info("FastMCP Intelligent Chatbot initialized successfully")
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise
    
    def setup_tools(self):
        """Setup FastMCP tools for chatbot operations"""
        
        @self.app.tool()
        async def start_conversation(
            user_message: str,
            channel: str = "web",
            user_id: Optional[str] = None,
            session_data: Optional[Dict[str, Any]] = None
        ) -> Dict[str, Any]:
            """Start a new conversation with the chatbot"""
            try:
                # Generate session ID
                session_id = str(uuid.uuid4())
                
                # Initialize conversation state
                conversation_state = {
                    'session_id': session_id,
                    'user_id': user_id,
                    'channel': channel,
                    'state': ConversationState.GREETING.value,
                    'intent': None,
                    'context': {},
                    'messages': [],
                    'created_at': datetime.utcnow(),
                    'updated_at': datetime.utcnow()
                }
                
                # Add session data if provided
                if session_data:
                    conversation_state['context'].update(session_data)
                
                # Store conversation in Redis
                await self.redis_client.setex(
                    f"conversation:{session_id}",
                    3600,  # 1 hour expiry
                    json.dumps(conversation_state, default=str)
                )
                
                # Process first message
                response = await self._process_message(session_id, user_message)
                
                return {
                    'session_id': session_id,
                    'response': response['message'],
                    'state': response['state'],
                    'intent': response['intent'],
                    'actions': response.get('actions', []),
                    'status': 'success'
                }
                
            except Exception as e:
                logger.error(f"Conversation start failed: {e}")
                return {
                    'status': 'error',
                    'message': f'Failed to start conversation: {str(e)}'
                }
        
        @self.app.tool()
        async def continue_conversation(
            session_id: str,
            user_message: str
        ) -> Dict[str, Any]:
            """Continue an existing conversation"""
            try:
                # Get conversation state
                conversation_data = await self.redis_client.get(f"conversation:{session_id}")
                if not conversation_data:
                    return {
                        'status': 'error',
                        'message': 'Session not found or expired'
                    }
                
                # Process message
                response = await self._process_message(session_id, user_message)
                
                return {
                    'session_id': session_id,
                    'response': response['message'],
                    'state': response['state'],
                    'intent': response['intent'],
                    'actions': response.get('actions', []),
                    'status': 'success'
                }
                
            except Exception as e:
                logger.error(f"Conversation continuation failed: {e}")
                return {
                    'status': 'error',
                    'message': f'Failed to continue conversation: {str(e)}'
                }
        
        @self.app.tool()
        async def get_conversation_history(
            session_id: str
        ) -> Dict[str, Any]:
            """Get conversation history for a session"""
            try:
                conversation_data = await self.redis_client.get(f"conversation:{session_id}")
                if not conversation_data:
                    return {
                        'status': 'error',
                        'message': 'Session not found'
                    }
                
                conversation = json.loads(conversation_data)
                
                return {
                    'session_id': session_id,
                    'messages': conversation['messages'],
                    'state': conversation['state'],
                    'intent': conversation['intent'],
                    'context': conversation['context'],
                    'created_at': conversation['created_at'],
                    'updated_at': conversation['updated_at'],
                    'status': 'success'
                }
                
            except Exception as e:
                logger.error(f"Failed to get conversation history: {e}")
                return {
                    'status': 'error',
                    'message': f'Failed to get conversation history: {str(e)}'
                }
        
        @self.app.tool()
        async def classify_intent(
            message: str,
            context: Optional[Dict[str, Any]] = None
        ) -> Dict[str, Any]:
            """Classify the intent of a user message"""
            try:
                if not self.openai_client:
                    # Fallback to simple keyword-based classification
                    return self._classify_intent_simple(message)
                
                # Use AI for intent classification
                context_str = json.dumps(context) if context else "No prior context"
                
                prompt = f"""
                Classify the intent of this customer message for an MSP (Managed Service Provider):

                Message: "{message}"
                Context: {context_str}

                Available intents:
                - general_inquiry: General questions about services
                - technical_support: Technical issues or support requests
                - sales_inquiry: Interest in purchasing services
                - billing_question: Questions about invoices or payments
                - appointment_scheduling: Want to schedule meetings or service
                - complaint: Complaints or negative feedback
                - emergency: Urgent technical issues

                Respond with JSON: {{"intent": "intent_name", "confidence": 0.95, "entities": {{"key": "value"}}, "reasoning": "explanation"}}
                """
                
                response = await self.openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200,
                    temperature=0.1
                )
                
                try:
                    result = json.loads(response.choices[0].message.content)
                    return {
                        'intent': result['intent'],
                        'confidence': result['confidence'],
                        'entities': result.get('entities', {}),
                        'reasoning': result.get('reasoning', ''),
                        'status': 'success'
                    }
                except json.JSONDecodeError:
                    return self._classify_intent_simple(message)
                
            except Exception as e:
                logger.error(f"Intent classification failed: {e}")
                return self._classify_intent_simple(message)
        
        @self.app.tool()
        async def generate_response(
            message: str,
            conversation_state: Dict[str, Any],
            knowledge_base_results: Optional[List[Dict]] = None
        ) -> Dict[str, Any]:
            """Generate an appropriate response using AI"""
            try:
                if not self.openai_client:
                    return {
                        'message': "I understand your inquiry. Let me connect you with a human agent who can better assist you.",
                        'actions': [{'type': 'escalate', 'reason': 'AI not available'}],
                        'status': 'success'
                    }
                
                # Build context for AI
                context = self._build_ai_context(conversation_state, knowledge_base_results)
                
                system_prompt = """
                You are an intelligent AI assistant for an MSP (Managed Service Provider).
                You help customers with inquiries, technical support, sales questions, and scheduling.
                
                Guidelines:
                - Be professional, helpful, and empathetic
                - Use the knowledge base information when available
                - Suggest next steps or actions when appropriate
                - Escalate to human agents for complex technical issues
                - Always aim to move the conversation toward resolution
                
                Respond with JSON containing:
                - message: Your response to the customer
                - actions: Array of suggested actions (e.g., schedule_call, escalate, create_ticket)
                - state_change: New conversation state if needed
                """
                
                user_prompt = f"""
                Customer message: "{message}"
                Conversation context: {json.dumps(context, default=str)}
                
                Generate an appropriate response.
                """
                
                response = await self.openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=500,
                    temperature=0.7
                )
                
                try:
                    result = json.loads(response.choices[0].message.content)
                    return {
                        'message': result['message'],
                        'actions': result.get('actions', []),
                        'state_change': result.get('state_change'),
                        'status': 'success'
                    }
                except json.JSONDecodeError:
                    # Fallback to plain text response
                    return {
                        'message': response.choices[0].message.content,
                        'actions': [],
                        'status': 'success'
                    }
                
            except Exception as e:
                logger.error(f"Response generation failed: {e}")
                return {
                    'message': "I apologize, but I'm having trouble processing your request right now. Let me connect you with a human agent.",
                    'actions': [{'type': 'escalate', 'reason': 'AI processing error'}],
                    'status': 'error'
                }
        
        @self.app.tool()
        async def search_knowledge_base(
            query: str,
            category: Optional[str] = None,
            limit: int = 3
        ) -> List[Dict[str, Any]]:
            """Search knowledge base for relevant information"""
            try:
                # Use existing RAG system functionality
                if self.embedding_model:
                    query_embedding = self.embedding_model.encode([query])
                    
                    sql_query = """
                        SELECT d.id, d.title, d.content, d.category,
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
                        if row['similarity_score'] >= 0.6:  # Minimum relevance threshold
                            results.append({
                                'id': row['id'],
                                'title': row['title'],
                                'content': row['content'][:300] + '...' if len(row['content']) > 300 else row['content'],
                                'category': row['category'],
                                'relevance_score': float(row['similarity_score'])
                            })
                    
                    return results
                
                return []
                
            except Exception as e:
                logger.error(f"Knowledge base search failed: {e}")
                return []
        
        @self.app.tool()
        async def create_support_ticket(
            session_id: str,
            issue_description: str,
            priority: str = "normal",
            category: Optional[str] = None
        ) -> Dict[str, Any]:
            """Create a support ticket from chatbot conversation"""
            try:
                # Get conversation context
                conversation_data = await self.redis_client.get(f"conversation:{session_id}")
                if not conversation_data:
                    return {
                        'status': 'error',
                        'message': 'Session not found'
                    }
                
                conversation = json.loads(conversation_data)
                
                # Create ticket in database
                ticket_data = {
                    'title': f"Chatbot Generated Ticket - {category or 'General'}",
                    'description': issue_description,
                    'priority': priority,
                    'status': 'open',
                    'source': 'chatbot',
                    'session_id': session_id,
                    'created_at': datetime.utcnow()
                }
                
                async with self.db_pool.acquire() as conn:
                    ticket_id = await conn.fetchval("""
                        INSERT INTO support_tickets 
                        (title, description, priority, status, source, metadata, created_at, updated_at)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                        RETURNING id
                    """,
                    ticket_data['title'],
                    ticket_data['description'], 
                    ticket_data['priority'],
                    ticket_data['status'],
                    ticket_data['source'],
                    json.dumps({'session_id': session_id, 'conversation_history': conversation['messages']}),
                    ticket_data['created_at'],
                    ticket_data['created_at'])
                
                return {
                    'ticket_id': ticket_id,
                    'status': 'created',
                    'priority': priority,
                    'message': f'Support ticket #{ticket_id} has been created. Our team will contact you soon.',
                    'estimated_response_time': self._get_estimated_response_time(priority)
                }
                
            except Exception as e:
                logger.error(f"Ticket creation failed: {e}")
                return {
                    'status': 'error',
                    'message': f'Failed to create support ticket: {str(e)}'
                }
        
        @self.app.tool()
        async def schedule_appointment(
            session_id: str,
            service_type: str,
            preferred_date: str,
            preferred_time: str,
            contact_info: Dict[str, str]
        ) -> Dict[str, Any]:
            """Schedule an appointment through chatbot"""
            try:
                # Parse datetime
                appointment_dt = datetime.fromisoformat(f"{preferred_date} {preferred_time}")
                
                # Create appointment request
                appointment_data = {
                    'service_type': service_type,
                    'requested_datetime': appointment_dt,
                    'contact_name': contact_info.get('name', ''),
                    'contact_phone': contact_info.get('phone', ''),
                    'contact_email': contact_info.get('email', ''),
                    'source': 'chatbot',
                    'session_id': session_id,
                    'status': 'requested',
                    'created_at': datetime.utcnow()
                }
                
                async with self.db_pool.acquire() as conn:
                    appointment_id = await conn.fetchval("""
                        INSERT INTO appointments 
                        (service_type, requested_datetime, contact_name, contact_phone, 
                         contact_email, source, metadata, status, created_at, updated_at)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                        RETURNING id
                    """,
                    appointment_data['service_type'],
                    appointment_data['requested_datetime'],
                    appointment_data['contact_name'],
                    appointment_data['contact_phone'],
                    appointment_data['contact_email'],
                    appointment_data['source'],
                    json.dumps({'session_id': session_id}),
                    appointment_data['status'],
                    appointment_data['created_at'],
                    appointment_data['created_at'])
                
                return {
                    'appointment_id': appointment_id,
                    'status': 'requested',
                    'requested_datetime': appointment_dt.isoformat(),
                    'message': f'Appointment request #{appointment_id} has been submitted. We will confirm the availability and contact you shortly.',
                    'next_steps': [
                        'Our team will review your request',
                        'You will receive a confirmation call/email within 24 hours',
                        'Alternative times may be suggested if requested slot is unavailable'
                    ]
                }
                
            except Exception as e:
                logger.error(f"Appointment scheduling failed: {e}")
                return {
                    'status': 'error',
                    'message': f'Failed to schedule appointment: {str(e)}'
                }
        
        @self.app.tool()
        async def get_chatbot_analytics(
            start_date: Optional[str] = None,
            end_date: Optional[str] = None
        ) -> Dict[str, Any]:
            """Get chatbot usage and performance analytics"""
            try:
                # Default to last 30 days
                if not start_date:
                    start_date = (datetime.utcnow() - timedelta(days=30)).isoformat()
                if not end_date:
                    end_date = datetime.utcnow().isoformat()
                
                start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                
                # Get conversation statistics from Redis
                # This is a simplified version - in production you'd store metrics in database
                
                analytics = {
                    'period': {
                        'start_date': start_dt.isoformat(),
                        'end_date': end_dt.isoformat()
                    },
                    'conversations': {
                        'total_sessions': 0,
                        'avg_session_length': 0,
                        'resolution_rate': 0,
                        'escalation_rate': 0
                    },
                    'intents': {
                        'technical_support': 0,
                        'sales_inquiry': 0,
                        'billing_question': 0,
                        'general_inquiry': 0,
                        'appointment_scheduling': 0,
                        'complaint': 0,
                        'emergency': 0
                    },
                    'actions_taken': {
                        'tickets_created': 0,
                        'appointments_scheduled': 0,
                        'escalations': 0,
                        'knowledge_base_queries': 0
                    },
                    'satisfaction': {
                        'positive_responses': 0,
                        'negative_responses': 0,
                        'neutral_responses': 0
                    },
                    'generated_at': datetime.utcnow().isoformat()
                }
                
                return analytics
                
            except Exception as e:
                logger.error(f"Analytics generation failed: {e}")
                return {
                    'status': 'error',
                    'message': f'Failed to generate analytics: {str(e)}'
                }
    
    def setup_resources(self):
        """Setup FastMCP resources"""
        
        @self.app.resource("conversation-templates")
        async def get_conversation_templates() -> List[Resource]:
            """Get available conversation flow templates"""
            templates = [
                Resource(
                    uri="template://technical-support-flow",
                    name="Technical Support Flow",
                    description="Conversation flow for technical support inquiries",
                    mimeType="application/json"
                ),
                Resource(
                    uri="template://sales-qualification-flow", 
                    name="Sales Qualification Flow",
                    description="Conversation flow for sales inquiries and lead qualification",
                    mimeType="application/json"
                ),
                Resource(
                    uri="template://appointment-booking-flow",
                    name="Appointment Booking Flow", 
                    description="Conversation flow for appointment scheduling",
                    mimeType="application/json"
                )
            ]
            return templates
    
    async def _process_message(self, session_id: str, message: str) -> Dict[str, Any]:
        """Process a user message and generate response"""
        # Get conversation state
        conversation_data = await self.redis_client.get(f"conversation:{session_id}")
        conversation = json.loads(conversation_data)
        
        # Add user message to history
        conversation['messages'].append({
            'role': 'user',
            'content': message,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        # Classify intent if not already set
        if not conversation.get('intent'):
            intent_result = await classify_intent(message, conversation.get('context'))
            conversation['intent'] = intent_result.get('intent')
        
        # Search knowledge base for relevant information
        kb_results = await search_knowledge_base(message, limit=2)
        
        # Generate response
        response_result = await generate_response(message, conversation, kb_results)
        
        # Add bot response to history
        conversation['messages'].append({
            'role': 'assistant',
            'content': response_result['message'],
            'timestamp': datetime.utcnow().isoformat(),
            'actions': response_result.get('actions', [])
        })
        
        # Update conversation state
        if response_result.get('state_change'):
            conversation['state'] = response_result['state_change']
        
        conversation['updated_at'] = datetime.utcnow().isoformat()
        
        # Save updated conversation
        await self.redis_client.setex(
            f"conversation:{session_id}",
            3600,
            json.dumps(conversation, default=str)
        )
        
        return {
            'message': response_result['message'],
            'state': conversation['state'],
            'intent': conversation['intent'],
            'actions': response_result.get('actions', [])
        }
    
    def _classify_intent_simple(self, message: str) -> Dict[str, Any]:
        """Simple keyword-based intent classification fallback"""
        message_lower = message.lower()
        
        # Technical support keywords
        if any(word in message_lower for word in ['error', 'broken', 'not working', 'issue', 'problem', 'help', 'support']):
            return {'intent': 'technical_support', 'confidence': 0.7, 'entities': {}}
        
        # Sales keywords  
        if any(word in message_lower for word in ['price', 'cost', 'buy', 'purchase', 'service', 'quote']):
            return {'intent': 'sales_inquiry', 'confidence': 0.7, 'entities': {}}
        
        # Billing keywords
        if any(word in message_lower for word in ['bill', 'invoice', 'payment', 'charge', 'account']):
            return {'intent': 'billing_question', 'confidence': 0.7, 'entities': {}}
        
        # Appointment keywords
        if any(word in message_lower for word in ['schedule', 'appointment', 'meeting', 'visit', 'book']):
            return {'intent': 'appointment_scheduling', 'confidence': 0.7, 'entities': {}}
        
        # Emergency keywords
        if any(word in message_lower for word in ['urgent', 'emergency', 'critical', 'down', 'outage']):
            return {'intent': 'emergency', 'confidence': 0.8, 'entities': {}}
        
        return {'intent': 'general_inquiry', 'confidence': 0.5, 'entities': {}}
    
    def _build_ai_context(self, conversation_state: Dict[str, Any], kb_results: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Build context for AI response generation"""
        context = {
            'conversation_state': conversation_state['state'],
            'intent': conversation_state.get('intent'),
            'message_history': conversation_state['messages'][-5:],  # Last 5 messages
            'customer_context': conversation_state.get('context', {}),
        }
        
        if kb_results:
            context['knowledge_base_info'] = kb_results
        
        return context
    
    def _get_estimated_response_time(self, priority: str) -> str:
        """Get estimated response time based on priority"""
        response_times = {
            'low': '3-5 business days',
            'normal': '1-2 business days',
            'high': '4-8 hours',
            'critical': '1-2 hours'
        }
        return response_times.get(priority, '1-2 business days')
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.db_pool:
                await self.db_pool.close()
            if self.redis_client:
                await self.redis_client.close()
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

chatbot = FastMCPIntelligentChatbot()

async def main():
    """Main entry point"""
    try:
        await chatbot.initialize()
        async with chatbot.app.run_server() as server:
            await server.run()
    except KeyboardInterrupt:
        logger.info("Shutting down chatbot...")
    except Exception as e:
        logger.error(f"Chatbot error: {e}")
    finally:
        await chatbot.cleanup()

if __name__ == "__main__":
    asyncio.run(main())