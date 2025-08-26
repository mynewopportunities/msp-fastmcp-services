#!/usr/bin/env python3
"""
FastMCP-based AI Calling Assistant for MSP Operations
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from enum import Enum

import asyncpg
import openai
from fastmcp import FastMCP
from mcp.types import Resource, Tool, TextContent
import redis.asyncio as redis
from twilio.rest import Client as TwilioClient
from twilio.twiml.voice_response import VoiceResponse, Gather
import speech_recognition as sr
from pydub import AudioSegment
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CallStatus(Enum):
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class CallType(Enum):
    OUTBOUND_SALES = "outbound_sales"
    CUSTOMER_SUPPORT = "customer_support"
    APPOINTMENT_REMINDER = "appointment_reminder"
    FOLLOW_UP = "follow_up"
    EMERGENCY_ALERT = "emergency_alert"

class FastMCPCallingAssistant:
    """FastMCP-based AI Calling Assistant"""
    
    def __init__(self):
        self.app = FastMCP("MSP AI Calling Assistant")
        self.db_pool = None
        self.redis_client = None
        self.openai_client = None
        self.twilio_client = None
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
            
            # Twilio client
            twilio_sid = os.getenv('TWILIO_ACCOUNT_SID')
            twilio_token = os.getenv('TWILIO_AUTH_TOKEN')
            if twilio_sid and twilio_token:
                self.twilio_client = TwilioClient(twilio_sid, twilio_token)
            
            logger.info("FastMCP Calling Assistant initialized successfully")
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise
    
    def setup_tools(self):
        """Setup FastMCP tools for calling operations"""
        
        @self.app.tool()
        async def schedule_call(
            contact_id: int,
            call_type: str,
            scheduled_time: str,
            priority: str = "normal",
            notes: Optional[str] = None,
            script_template: Optional[str] = None
        ) -> Dict[str, Any]:
            """Schedule a new AI-powered call"""
            try:
                # Validate call type
                if call_type not in [ct.value for ct in CallType]:
                    return {
                        'status': 'error',
                        'message': f'Invalid call type. Must be one of: {[ct.value for ct in CallType]}'
                    }
                
                # Parse scheduled time
                try:
                    scheduled_dt = datetime.fromisoformat(scheduled_time.replace('Z', '+00:00'))
                except ValueError:
                    return {
                        'status': 'error',
                        'message': 'Invalid datetime format. Use ISO 8601 format.'
                    }
                
                # Get contact information
                async with self.db_pool.acquire() as conn:
                    contact = await conn.fetchrow("""
                        SELECT id, first_name, last_name, phone, email, organization_id
                        FROM contacts WHERE id = $1
                    """, contact_id)
                    
                    if not contact:
                        return {
                            'status': 'error',
                            'message': 'Contact not found'
                        }
                
                # Create call record
                call_data = {
                    'contact_id': contact_id,
                    'call_type': call_type,
                    'status': CallStatus.SCHEDULED.value,
                    'scheduled_time': scheduled_dt,
                    'priority': priority,
                    'notes': notes or '',
                    'script_template': script_template or self._get_default_script(call_type),
                    'created_at': datetime.utcnow(),
                    'updated_at': datetime.utcnow()
                }
                
                async with self.db_pool.acquire() as conn:
                    call_id = await conn.fetchval("""
                        INSERT INTO ai_calls 
                        (contact_id, call_type, status, scheduled_time, priority, notes, script_template, created_at, updated_at)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                        RETURNING id
                    """, *call_data.values())
                
                # Schedule the call execution
                await self._schedule_call_execution(call_id, scheduled_dt)
                
                return {
                    'call_id': call_id,
                    'contact': {
                        'name': f"{contact['first_name']} {contact['last_name']}",
                        'phone': contact['phone']
                    },
                    'scheduled_time': scheduled_dt.isoformat(),
                    'call_type': call_type,
                    'status': 'scheduled',
                    'message': 'Call scheduled successfully'
                }
                
            except Exception as e:
                logger.error(f"Call scheduling failed: {e}")
                return {
                    'status': 'error',
                    'message': f'Failed to schedule call: {str(e)}'
                }
        
        @self.app.tool()
        async def execute_call(call_id: int) -> Dict[str, Any]:
            """Execute an AI-powered call"""
            try:
                # Get call details
                async with self.db_pool.acquire() as conn:
                    call = await conn.fetchrow("""
                        SELECT c.*, ct.first_name, ct.last_name, ct.phone, ct.email
                        FROM ai_calls c
                        JOIN contacts ct ON c.contact_id = ct.id
                        WHERE c.id = $1
                    """, call_id)
                    
                    if not call:
                        return {
                            'status': 'error',
                            'message': 'Call not found'
                        }
                
                # Update call status
                await self._update_call_status(call_id, CallStatus.IN_PROGRESS)
                
                # Generate personalized script
                script = await self._generate_call_script(call)
                
                # Execute the call via Twilio
                if self.twilio_client:
                    call_result = await self._execute_twilio_call(call, script)
                else:
                    # Simulate call for testing
                    call_result = await self._simulate_call(call, script)
                
                # Update call with results
                await self._update_call_results(call_id, call_result)
                
                return {
                    'call_id': call_id,
                    'status': call_result['status'],
                    'duration': call_result.get('duration', 0),
                    'transcript': call_result.get('transcript', ''),
                    'outcome': call_result.get('outcome', ''),
                    'next_action': call_result.get('next_action', ''),
                    'message': 'Call executed successfully'
                }
                
            except Exception as e:
                logger.error(f"Call execution failed: {e}")
                await self._update_call_status(call_id, CallStatus.FAILED)
                return {
                    'status': 'error',
                    'message': f'Failed to execute call: {str(e)}'
                }
        
        @self.app.tool()
        async def get_call_status(call_id: int) -> Dict[str, Any]:
            """Get the status and details of a specific call"""
            try:
                async with self.db_pool.acquire() as conn:
                    call = await conn.fetchrow("""
                        SELECT c.*, ct.first_name, ct.last_name, ct.phone
                        FROM ai_calls c
                        JOIN contacts ct ON c.contact_id = ct.id
                        WHERE c.id = $1
                    """, call_id)
                    
                    if not call:
                        return {
                            'status': 'error',
                            'message': 'Call not found'
                        }
                
                return {
                    'call_id': call_id,
                    'contact': {
                        'name': f"{call['first_name']} {call['last_name']}",
                        'phone': call['phone']
                    },
                    'call_type': call['call_type'],
                    'status': call['status'],
                    'scheduled_time': call['scheduled_time'].isoformat() if call['scheduled_time'] else None,
                    'started_at': call['started_at'].isoformat() if call['started_at'] else None,
                    'ended_at': call['ended_at'].isoformat() if call['ended_at'] else None,
                    'duration': call['duration'],
                    'transcript': call['transcript'],
                    'outcome': call['outcome'],
                    'next_action': call['next_action'],
                    'created_at': call['created_at'].isoformat()
                }
                
            except Exception as e:
                logger.error(f"Failed to get call status: {e}")
                return {
                    'status': 'error',
                    'message': f'Failed to get call status: {str(e)}'
                }
        
        @self.app.tool()
        async def list_calls(
            status: Optional[str] = None,
            call_type: Optional[str] = None,
            contact_id: Optional[int] = None,
            limit: int = 20
        ) -> List[Dict[str, Any]]:
            """List calls with optional filters"""
            try:
                where_conditions = []
                params = []
                param_count = 1
                
                if status:
                    where_conditions.append(f"c.status = ${param_count}")
                    params.append(status)
                    param_count += 1
                
                if call_type:
                    where_conditions.append(f"c.call_type = ${param_count}")
                    params.append(call_type)
                    param_count += 1
                
                if contact_id:
                    where_conditions.append(f"c.contact_id = ${param_count}")
                    params.append(contact_id)
                    param_count += 1
                
                where_clause = "WHERE " + " AND ".join(where_conditions) if where_conditions else ""
                params.append(limit)
                
                query = f"""
                    SELECT c.*, ct.first_name, ct.last_name, ct.phone
                    FROM ai_calls c
                    JOIN contacts ct ON c.contact_id = ct.id
                    {where_clause}
                    ORDER BY c.created_at DESC
                    LIMIT ${param_count}
                """
                
                async with self.db_pool.acquire() as conn:
                    rows = await conn.fetch(query, *params)
                
                calls = []
                for row in rows:
                    calls.append({
                        'call_id': row['id'],
                        'contact': {
                            'name': f"{row['first_name']} {row['last_name']}",
                            'phone': row['phone']
                        },
                        'call_type': row['call_type'],
                        'status': row['status'],
                        'scheduled_time': row['scheduled_time'].isoformat() if row['scheduled_time'] else None,
                        'duration': row['duration'],
                        'outcome': row['outcome'],
                        'created_at': row['created_at'].isoformat()
                    })
                
                return calls
                
            except Exception as e:
                logger.error(f"Failed to list calls: {e}")
                return []
        
        @self.app.tool()
        async def cancel_call(call_id: int, reason: Optional[str] = None) -> Dict[str, Any]:
            """Cancel a scheduled call"""
            try:
                async with self.db_pool.acquire() as conn:
                    # Check if call exists and can be cancelled
                    call = await conn.fetchrow("""
                        SELECT id, status FROM ai_calls WHERE id = $1
                    """, call_id)
                    
                    if not call:
                        return {
                            'status': 'error',
                            'message': 'Call not found'
                        }
                    
                    if call['status'] not in [CallStatus.SCHEDULED.value]:
                        return {
                            'status': 'error',
                            'message': f'Cannot cancel call with status: {call["status"]}'
                        }
                    
                    # Update call status
                    await conn.execute("""
                        UPDATE ai_calls 
                        SET status = $1, notes = COALESCE(notes, '') || $2, updated_at = NOW()
                        WHERE id = $3
                    """, CallStatus.CANCELLED.value, f"\nCancellation reason: {reason or 'Not specified'}", call_id)
                
                # Remove from scheduled execution
                await self._remove_scheduled_call(call_id)
                
                return {
                    'call_id': call_id,
                    'status': 'cancelled',
                    'reason': reason,
                    'message': 'Call cancelled successfully'
                }
                
            except Exception as e:
                logger.error(f"Call cancellation failed: {e}")
                return {
                    'status': 'error',
                    'message': f'Failed to cancel call: {str(e)}'
                }
        
        @self.app.tool()
        async def get_call_analytics(
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            call_type: Optional[str] = None
        ) -> Dict[str, Any]:
            """Get analytics for AI calling operations"""
            try:
                # Default to last 30 days
                if not start_date:
                    start_date = (datetime.utcnow() - timedelta(days=30)).isoformat()
                if not end_date:
                    end_date = datetime.utcnow().isoformat()
                
                start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                
                async with self.db_pool.acquire() as conn:
                    # Overall stats
                    stats_query = """
                        SELECT 
                            COUNT(*) as total_calls,
                            COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_calls,
                            COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_calls,
                            AVG(duration) as avg_duration,
                            SUM(duration) as total_duration
                        FROM ai_calls
                        WHERE created_at BETWEEN $1 AND $2
                        AND ($3::text IS NULL OR call_type = $3)
                    """
                    
                    stats = await conn.fetchrow(stats_query, start_dt, end_dt, call_type)
                    
                    # Call type breakdown
                    type_query = """
                        SELECT 
                            call_type,
                            COUNT(*) as count,
                            AVG(duration) as avg_duration
                        FROM ai_calls
                        WHERE created_at BETWEEN $1 AND $2
                        GROUP BY call_type
                        ORDER BY count DESC
                    """
                    
                    type_stats = await conn.fetch(type_query, start_dt, end_dt)
                    
                    # Daily activity
                    daily_query = """
                        SELECT 
                            DATE(created_at) as date,
                            COUNT(*) as calls_made,
                            COUNT(CASE WHEN status = 'completed' THEN 1 END) as successful_calls
                        FROM ai_calls
                        WHERE created_at BETWEEN $1 AND $2
                        GROUP BY DATE(created_at)
                        ORDER BY date DESC
                    """
                    
                    daily_stats = await conn.fetch(daily_query, start_dt, end_dt)
                
                success_rate = (
                    (stats['completed_calls'] / stats['total_calls'] * 100) 
                    if stats['total_calls'] > 0 else 0
                )
                
                analytics = {
                    'period': {
                        'start_date': start_dt.isoformat(),
                        'end_date': end_dt.isoformat()
                    },
                    'summary': {
                        'total_calls': stats['total_calls'],
                        'completed_calls': stats['completed_calls'],
                        'failed_calls': stats['failed_calls'],
                        'success_rate': round(success_rate, 2),
                        'avg_duration': round(float(stats['avg_duration'] or 0), 2),
                        'total_duration': round(float(stats['total_duration'] or 0), 2)
                    },
                    'by_call_type': [
                        {
                            'call_type': row['call_type'],
                            'count': row['count'],
                            'avg_duration': round(float(row['avg_duration'] or 0), 2)
                        }
                        for row in type_stats
                    ],
                    'daily_activity': [
                        {
                            'date': row['date'].isoformat(),
                            'calls_made': row['calls_made'],
                            'successful_calls': row['successful_calls'],
                            'success_rate': round((row['successful_calls'] / row['calls_made'] * 100), 2)
                        }
                        for row in daily_stats
                    ],
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
        
        @self.app.resource("call-templates")
        async def get_call_templates() -> List[Resource]:
            """Get available call script templates"""
            templates = [
                Resource(
                    uri="template://sales-outbound",
                    name="Sales Outbound Template",
                    description="Template for outbound sales calls",
                    mimeType="text/plain"
                ),
                Resource(
                    uri="template://customer-support", 
                    name="Customer Support Template",
                    description="Template for customer support calls",
                    mimeType="text/plain"
                ),
                Resource(
                    uri="template://appointment-reminder",
                    name="Appointment Reminder Template", 
                    description="Template for appointment reminder calls",
                    mimeType="text/plain"
                )
            ]
            return templates
    
    def _get_default_script(self, call_type: str) -> str:
        """Get default script template for call type"""
        scripts = {
            "outbound_sales": "Hello, this is an AI assistant calling on behalf of [COMPANY]. I'd like to discuss our managed IT services...",
            "customer_support": "Hello, this is [COMPANY] AI support. I'm calling regarding your recent service ticket...",
            "appointment_reminder": "Hello, this is a reminder from [COMPANY] about your upcoming appointment...",
            "follow_up": "Hello, this is a follow-up call from [COMPANY] regarding...",
            "emergency_alert": "This is an urgent notification from [COMPANY] regarding..."
        }
        return scripts.get(call_type, "Hello, this is an AI assistant from [COMPANY]...")
    
    async def _generate_call_script(self, call: dict) -> str:
        """Generate personalized call script using AI"""
        if not self.openai_client:
            return call['script_template']
        
        try:
            prompt = f"""
            Generate a personalized call script for the following:
            - Contact: {call['first_name']} {call['last_name']}
            - Call Type: {call['call_type']}
            - Template: {call['script_template']}
            - Notes: {call['notes']}
            
            Create a natural, professional script that sounds human-like.
            """
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Script generation failed: {e}")
            return call['script_template']
    
    async def _execute_twilio_call(self, call: dict, script: str) -> dict:
        """Execute call via Twilio"""
        # Implementation would integrate with Twilio Voice API
        # This is a simplified version
        return {
            'status': 'completed',
            'duration': 120,
            'transcript': 'Call completed successfully',
            'outcome': 'Positive response',
            'next_action': 'Send follow-up email'
        }
    
    async def _simulate_call(self, call: dict, script: str) -> dict:
        """Simulate call execution for testing"""
        await asyncio.sleep(2)  # Simulate call duration
        return {
            'status': 'completed',
            'duration': 90,
            'transcript': f"Simulated call to {call['first_name']} {call['last_name']}",
            'outcome': 'Call completed successfully',
            'next_action': 'Schedule follow-up'
        }
    
    async def _update_call_status(self, call_id: int, status: CallStatus):
        """Update call status in database"""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE ai_calls 
                SET status = $1, updated_at = NOW()
                WHERE id = $2
            """, status.value, call_id)
    
    async def _update_call_results(self, call_id: int, results: dict):
        """Update call with execution results"""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE ai_calls 
                SET status = $1, duration = $2, transcript = $3, 
                    outcome = $4, next_action = $5, ended_at = NOW(), updated_at = NOW()
                WHERE id = $6
            """, 
            results['status'], 
            results.get('duration', 0),
            results.get('transcript', ''),
            results.get('outcome', ''),
            results.get('next_action', ''),
            call_id)
    
    async def _schedule_call_execution(self, call_id: int, scheduled_time: datetime):
        """Schedule call execution in Redis"""
        delay = (scheduled_time - datetime.utcnow()).total_seconds()
        if delay > 0:
            await self.redis_client.setex(
                f"scheduled_call:{call_id}",
                int(delay),
                json.dumps({'call_id': call_id, 'scheduled_time': scheduled_time.isoformat()})
            )
    
    async def _remove_scheduled_call(self, call_id: int):
        """Remove scheduled call from Redis"""
        await self.redis_client.delete(f"scheduled_call:{call_id}")
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.db_pool:
                await self.db_pool.close()
            if self.redis_client:
                await self.redis_client.close()
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

calling_assistant = FastMCPCallingAssistant()

async def main():
    """Main entry point"""
    try:
        await calling_assistant.initialize()
        async with calling_assistant.app.run_server() as server:
            await server.run()
    except KeyboardInterrupt:
        logger.info("Shutting down calling assistant...")
    except Exception as e:
        logger.error(f"Calling assistant error: {e}")
    finally:
        await calling_assistant.cleanup()

if __name__ == "__main__":
    asyncio.run(main())