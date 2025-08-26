#!/usr/bin/env python3
"""
FastMCP Service Orchestrator for MSP Operations
Main entry point for all FastMCP-based AI services
"""

import asyncio
import json
import logging
import os
import signal
from typing import Dict, List, Any
from contextlib import asynccontextmanager
import uvloop
from fastmcp import FastMCP
from mcp.types import Resource, Tool

# Import all FastMCP services
from fastmcp_rag_system import FastMCPRAGSystem
from fastmcp_calling_assistant import FastMCPCallingAssistant
from fastmcp_intelligent_chatbot import FastMCPIntelligentChatbot
from fastmcp_analytics_dashboard import FastMCPAnalyticsDashboard

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MSPServiceOrchestrator:
    """Main orchestrator for all MSP FastMCP services"""
    
    def __init__(self):
        self.app = FastMCP("MSP Service Orchestrator")
        self.services = {}
        self.running = False
        self.setup_orchestrator_tools()
        
    async def initialize_services(self):
        """Initialize all MSP services"""
        try:
            logger.info("Initializing MSP FastMCP services...")
            
            # Initialize RAG System
            self.services['rag'] = FastMCPRAGSystem()
            await self.services['rag'].initialize()
            logger.info("✓ RAG System initialized")
            
            # Initialize Calling Assistant
            self.services['calling'] = FastMCPCallingAssistant()
            await self.services['calling'].initialize()
            logger.info("✓ Calling Assistant initialized")
            
            # Initialize Chatbot
            self.services['chatbot'] = FastMCPIntelligentChatbot()
            await self.services['chatbot'].initialize()
            logger.info("✓ Intelligent Chatbot initialized")
            
            # Initialize Analytics Dashboard
            self.services['analytics'] = FastMCPAnalyticsDashboard()
            await self.services['analytics'].initialize()
            logger.info("✓ Analytics Dashboard initialized")
            
            logger.info("All MSP FastMCP services initialized successfully")
            
        except Exception as e:
            logger.error(f"Service initialization failed: {e}")
            raise
    
    def setup_orchestrator_tools(self):
        """Setup orchestrator-level tools"""
        
        @self.app.tool()
        async def get_service_status() -> Dict[str, Any]:
            """Get status of all MSP services"""
            try:
                status = {
                    'orchestrator': {
                        'running': self.running,
                        'services_loaded': len(self.services),
                        'timestamp': asyncio.get_event_loop().time()
                    },
                    'services': {}
                }
                
                for service_name, service in self.services.items():
                    try:
                        # Basic health check - verify service objects exist and are initialized
                        service_status = {
                            'initialized': hasattr(service, 'db_pool') and service.db_pool is not None,
                            'redis_connected': hasattr(service, 'redis_client') and service.redis_client is not None,
                            'ai_available': hasattr(service, 'openai_client') and service.openai_client is not None,
                            'status': 'healthy' if hasattr(service, 'db_pool') and service.db_pool else 'degraded'
                        }
                        status['services'][service_name] = service_status
                    except Exception as e:
                        status['services'][service_name] = {
                            'status': 'error',
                            'error': str(e)
                        }
                
                return status
                
            except Exception as e:
                logger.error(f"Service status check failed: {e}")
                return {
                    'status': 'error',
                    'message': f'Failed to get service status: {str(e)}'
                }
        
        @self.app.tool()
        async def execute_cross_service_workflow(
            workflow_name: str,
            parameters: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Execute workflows that span multiple services"""
            try:
                if workflow_name == "customer_inquiry_to_ticket":
                    return await self._handle_customer_inquiry_workflow(parameters)
                elif workflow_name == "sales_lead_qualification":
                    return await self._handle_sales_qualification_workflow(parameters)
                elif workflow_name == "compliance_assessment_report":
                    return await self._handle_compliance_report_workflow(parameters)
                elif workflow_name == "proactive_client_outreach":
                    return await self._handle_proactive_outreach_workflow(parameters)
                else:
                    return {
                        'status': 'error',
                        'message': f'Unknown workflow: {workflow_name}'
                    }
                    
            except Exception as e:
                logger.error(f"Cross-service workflow failed: {e}")
                return {
                    'status': 'error',
                    'message': f'Workflow execution failed: {str(e)}'
                }
        
        @self.app.tool()
        async def get_unified_analytics(
            start_date: str = None,
            end_date: str = None,
            include_predictions: bool = False
        ) -> Dict[str, Any]:
            """Get unified analytics across all services"""
            try:
                analytics = {
                    'period': {
                        'start_date': start_date,
                        'end_date': end_date
                    },
                    'summary': {
                        'total_customer_interactions': 0,
                        'revenue_impact': 0,
                        'operational_efficiency': 0,
                        'compliance_score': 0
                    },
                    'service_metrics': {},
                    'generated_at': asyncio.get_event_loop().time()
                }
                
                # Get analytics from each service
                if 'analytics' in self.services:
                    # Revenue analytics
                    revenue_data = await self.services['analytics'].app.tools['get_revenue_analytics'](
                        start_date=start_date,
                        end_date=end_date
                    )
                    analytics['service_metrics']['revenue'] = revenue_data
                    analytics['summary']['revenue_impact'] = revenue_data.get('summary', {}).get('total_revenue', 0)
                    
                    # Operational metrics
                    operational_data = await self.services['analytics'].app.tools['get_operational_metrics'](
                        period_days=30
                    )
                    analytics['service_metrics']['operations'] = operational_data
                    
                    # Compliance dashboard
                    compliance_data = await self.services['analytics'].app.tools['get_compliance_dashboard']()
                    analytics['service_metrics']['compliance'] = compliance_data
                    analytics['summary']['compliance_score'] = compliance_data.get('summary', {}).get('compliance_rate', 0)
                
                # Get chatbot analytics
                if 'chatbot' in self.services:
                    chatbot_data = await self.services['chatbot'].app.tools['get_chatbot_analytics'](
                        start_date=start_date,
                        end_date=end_date
                    )
                    analytics['service_metrics']['chatbot'] = chatbot_data
                    analytics['summary']['total_customer_interactions'] += chatbot_data.get('conversations', {}).get('total_sessions', 0)
                
                # Get calling analytics
                if 'calling' in self.services:
                    calling_data = await self.services['calling'].app.tools['get_call_analytics'](
                        start_date=start_date,
                        end_date=end_date
                    )
                    analytics['service_metrics']['calling'] = calling_data
                    analytics['summary']['total_customer_interactions'] += calling_data.get('summary', {}).get('total_calls', 0)
                
                return analytics
                
            except Exception as e:
                logger.error(f"Unified analytics failed: {e}")
                return {
                    'status': 'error',
                    'message': f'Failed to generate unified analytics: {str(e)}'
                }
    
    async def _handle_customer_inquiry_workflow(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle customer inquiry from chatbot to support ticket creation"""
        try:
            # 1. Start conversation with chatbot
            chatbot_response = await self.services['chatbot'].app.tools['start_conversation'](
                user_message=parameters['initial_message'],
                channel=parameters.get('channel', 'web'),
                user_id=parameters.get('user_id')
            )
            
            session_id = chatbot_response['session_id']
            
            # 2. If intent is technical support, create ticket
            if chatbot_response['intent'] == 'technical_support':
                ticket_result = await self.services['chatbot'].app.tools['create_support_ticket'](
                    session_id=session_id,
                    issue_description=parameters['initial_message'],
                    priority=parameters.get('priority', 'normal')
                )
                
                # 3. Search knowledge base for relevant solutions
                kb_results = await self.services['rag'].app.tools['search_knowledge_base'](
                    query=parameters['initial_message'],
                    category='technical',
                    limit=3
                )
                
                return {
                    'workflow': 'customer_inquiry_to_ticket',
                    'session_id': session_id,
                    'chatbot_response': chatbot_response,
                    'ticket_created': ticket_result,
                    'knowledge_base_suggestions': kb_results,
                    'status': 'completed'
                }
            
            return {
                'workflow': 'customer_inquiry_to_ticket',
                'session_id': session_id,
                'chatbot_response': chatbot_response,
                'next_action': 'continue_conversation',
                'status': 'in_progress'
            }
            
        except Exception as e:
            logger.error(f"Customer inquiry workflow failed: {e}")
            return {
                'status': 'error',
                'message': f'Customer inquiry workflow failed: {str(e)}'
            }
    
    async def _handle_sales_qualification_workflow(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle sales lead qualification workflow"""
        try:
            # 1. Use chatbot for initial qualification
            chatbot_response = await self.services['chatbot'].app.tools['start_conversation'](
                user_message=parameters['initial_inquiry'],
                channel='sales',
                session_data={'lead_source': parameters.get('lead_source', 'unknown')}
            )
            
            # 2. If qualified, search for relevant service information
            if chatbot_response['intent'] == 'sales_inquiry':
                service_info = await self.services['rag'].app.tools['search_knowledge_base'](
                    query=parameters['initial_inquiry'],
                    category='sales',
                    limit=5
                )
                
                # 3. Schedule follow-up call if requested
                if parameters.get('schedule_call', False):
                    call_result = await self.services['calling'].app.tools['schedule_call'](
                        contact_id=parameters['contact_id'],
                        call_type='outbound_sales',
                        scheduled_time=parameters['preferred_call_time'],
                        notes=f"Follow-up for sales inquiry: {parameters['initial_inquiry']}"
                    )
                    
                    return {
                        'workflow': 'sales_lead_qualification',
                        'qualification_result': chatbot_response,
                        'service_information': service_info,
                        'call_scheduled': call_result,
                        'status': 'completed'
                    }
                
                return {
                    'workflow': 'sales_lead_qualification',
                    'qualification_result': chatbot_response,
                    'service_information': service_info,
                    'recommended_action': 'schedule_call',
                    'status': 'qualified'
                }
            
            return {
                'workflow': 'sales_lead_qualification',
                'qualification_result': chatbot_response,
                'status': 'not_qualified'
            }
            
        except Exception as e:
            logger.error(f"Sales qualification workflow failed: {e}")
            return {
                'status': 'error',
                'message': f'Sales qualification workflow failed: {str(e)}'
            }
    
    async def _handle_compliance_report_workflow(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle compliance assessment and reporting workflow"""
        try:
            # 1. Get compliance dashboard data
            compliance_data = await self.services['analytics'].app.tools['get_compliance_dashboard'](
                framework=parameters.get('framework'),
                organization_id=parameters.get('organization_id')
            )
            
            # 2. Get relevant compliance knowledge from RAG
            compliance_kb = await self.services['rag'].app.tools['search_knowledge_base'](
                query=f"compliance {parameters.get('framework', 'general')} requirements best practices",
                category='compliance',
                limit=10
            )
            
            # 3. Generate AI-powered compliance report summary
            if compliance_kb:
                report_summary = await self.services['rag'].app.tools['generate_answer'](
                    question=f"Generate a compliance report summary for {parameters.get('framework', 'general')} framework based on current status",
                    context_documents=[str(doc['id']) for doc in compliance_kb[:5]]
                )
            else:
                report_summary = {'answer': 'Compliance report generated from dashboard data only'}
            
            # 4. If non-compliant items found, schedule remediation calls
            scheduled_calls = []
            if compliance_data.get('summary', {}).get('compliance_rate', 100) < 90:
                if parameters.get('auto_schedule_remediation', False):
                    for assessment in compliance_data.get('recent_assessments', [])[:3]:
                        if assessment['status'] != 'compliant':
                            call_result = await self.services['calling'].app.tools['schedule_call'](
                                contact_id=parameters.get('contact_id'),
                                call_type='follow_up',
                                scheduled_time=parameters.get('remediation_call_time'),
                                notes=f"Compliance remediation discussion for {assessment['framework']}"
                            )
                            scheduled_calls.append(call_result)
            
            return {
                'workflow': 'compliance_assessment_report',
                'compliance_status': compliance_data,
                'knowledge_base_insights': compliance_kb,
                'ai_report_summary': report_summary,
                'remediation_calls': scheduled_calls,
                'status': 'completed'
            }
            
        except Exception as e:
            logger.error(f"Compliance report workflow failed: {e}")
            return {
                'status': 'error',
                'message': f'Compliance report workflow failed: {str(e)}'
            }
    
    async def _handle_proactive_outreach_workflow(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle proactive client outreach based on analytics"""
        try:
            # 1. Get client profitability analysis
            profitability_data = await self.services['analytics'].app.tools['get_client_profitability_analysis'](
                organization_id=parameters.get('organization_id'),
                period_months=parameters.get('analysis_period', 6)
            )
            
            # 2. Identify clients needing attention (low profitability, high support load)
            at_risk_clients = []
            for client in profitability_data.get('client_profitability', []):
                if (client.get('profit_margin', 0) < 10 or 
                    client.get('support_metrics', {}).get('ticket_count', 0) > 20):
                    at_risk_clients.append(client)
            
            # 3. Get relevant knowledge for client success strategies
            success_strategies = await self.services['rag'].app.tools['search_knowledge_base'](
                query="client retention strategies account management best practices",
                category='sales',
                limit=5
            )
            
            # 4. Schedule proactive calls for at-risk clients
            scheduled_calls = []
            for client in at_risk_clients[:parameters.get('max_calls', 5)]:
                call_notes = f"Proactive outreach - Profit margin: {client.get('profit_margin', 0)}%, Support tickets: {client.get('support_metrics', {}).get('ticket_count', 0)}"
                
                call_result = await self.services['calling'].app.tools['schedule_call'](
                    contact_id=parameters.get('default_contact_id', 1),  # Would need proper contact mapping
                    call_type='follow_up',
                    scheduled_time=parameters.get('call_time'),
                    notes=call_notes,
                    script_template="Proactive client success call template"
                )
                scheduled_calls.append({
                    'client': client,
                    'call_details': call_result
                })
            
            return {
                'workflow': 'proactive_client_outreach',
                'profitability_analysis': profitability_data,
                'at_risk_clients': at_risk_clients,
                'success_strategies': success_strategies,
                'scheduled_calls': scheduled_calls,
                'status': 'completed'
            }
            
        except Exception as e:
            logger.error(f"Proactive outreach workflow failed: {e}")
            return {
                'status': 'error',
                'message': f'Proactive outreach workflow failed: {str(e)}'
            }
    
    async def start_services(self):
        """Start all services in the orchestrator"""
        try:
            self.running = True
            logger.info("Starting MSP Service Orchestrator...")
            
            # Initialize all services
            await self.initialize_services()
            
            # Start orchestrator server
            async with self.app.run_server() as server:
                logger.info("MSP Service Orchestrator is running")
                await server.run()
                
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        except Exception as e:
            logger.error(f"Service orchestrator error: {e}")
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """Cleanup all services"""
        try:
            logger.info("Shutting down MSP services...")
            self.running = False
            
            for service_name, service in self.services.items():
                try:
                    await service.cleanup()
                    logger.info(f"✓ {service_name} service cleaned up")
                except Exception as e:
                    logger.error(f"Cleanup failed for {service_name}: {e}")
            
            logger.info("MSP Service Orchestrator shutdown complete")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

# Global orchestrator instance
orchestrator = MSPServiceOrchestrator()

def setup_signal_handlers():
    """Setup graceful shutdown signal handlers"""
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}")
        # The event loop will handle the KeyboardInterrupt
        raise KeyboardInterrupt()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

async def main():
    """Main entry point for the MSP Service Orchestrator"""
    try:
        # Use uvloop for better performance
        if hasattr(uvloop, 'install'):
            uvloop.install()
        
        # Setup signal handlers
        setup_signal_handlers()
        
        # Start the orchestrator
        await orchestrator.start_services()
        
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)