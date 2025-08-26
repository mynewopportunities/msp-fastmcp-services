#!/usr/bin/env python3
"""
FastMCP-based Analytics Dashboard for MSP Operations
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from decimal import Decimal

import asyncpg
from fastmcp import FastMCP
from mcp.types import Resource, Tool, TextContent
import redis.asyncio as redis
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FastMCPAnalyticsDashboard:
    """FastMCP-based Analytics Dashboard"""
    
    def __init__(self):
        self.app = FastMCP("MSP Analytics Dashboard")
        self.db_pool = None
        self.redis_client = None
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
            
            logger.info("FastMCP Analytics Dashboard initialized successfully")
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise
    
    def setup_tools(self):
        """Setup FastMCP tools for analytics operations"""
        
        @self.app.tool()
        async def get_revenue_analytics(
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            granularity: str = "monthly",
            organization_id: Optional[int] = None
        ) -> Dict[str, Any]:
            """Get comprehensive revenue analytics"""
            try:
                # Default to last 12 months
                if not start_date:
                    start_date = (datetime.utcnow() - timedelta(days=365)).isoformat()
                if not end_date:
                    end_date = datetime.utcnow().isoformat()
                
                start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                
                # Build time grouping based on granularity
                time_group = {
                    'daily': "DATE(i.invoice_date)",
                    'weekly': "DATE_TRUNC('week', i.invoice_date)",
                    'monthly': "DATE_TRUNC('month', i.invoice_date)",
                    'quarterly': "DATE_TRUNC('quarter', i.invoice_date)"
                }.get(granularity, "DATE_TRUNC('month', i.invoice_date)")
                
                async with self.db_pool.acquire() as conn:
                    # Revenue over time
                    revenue_query = f"""
                        SELECT 
                            {time_group} as period,
                            SUM(i.total_amount) as total_revenue,
                            COUNT(DISTINCT i.id) as invoice_count,
                            COUNT(DISTINCT i.organization_id) as unique_clients,
                            AVG(i.total_amount) as avg_invoice_amount
                        FROM invoices i
                        WHERE i.invoice_date BETWEEN $1 AND $2
                        AND i.status != 'cancelled'
                        AND ($3::int IS NULL OR i.organization_id = $3)
                        GROUP BY {time_group}
                        ORDER BY period
                    """
                    
                    revenue_data = await conn.fetch(revenue_query, start_dt, end_dt, organization_id)
                    
                    # Revenue by service type
                    service_query = """
                        SELECT 
                            si.service_name,
                            SUM(ii.amount) as total_revenue,
                            COUNT(DISTINCT i.organization_id) as client_count,
                            AVG(ii.amount) as avg_amount
                        FROM invoice_items ii
                        JOIN invoices i ON ii.invoice_id = i.id
                        JOIN service_items si ON ii.service_item_id = si.id
                        WHERE i.invoice_date BETWEEN $1 AND $2
                        AND i.status != 'cancelled'
                        AND ($3::int IS NULL OR i.organization_id = $3)
                        GROUP BY si.service_name
                        ORDER BY total_revenue DESC
                    """
                    
                    service_data = await conn.fetch(service_query, start_dt, end_dt, organization_id)
                    
                    # Top clients by revenue
                    client_query = """
                        SELECT 
                            o.name as client_name,
                            o.id as organization_id,
                            SUM(i.total_amount) as total_revenue,
                            COUNT(i.id) as invoice_count,
                            AVG(i.total_amount) as avg_invoice_amount
                        FROM invoices i
                        JOIN organizations o ON i.organization_id = o.id
                        WHERE i.invoice_date BETWEEN $1 AND $2
                        AND i.status != 'cancelled'
                        AND ($3::int IS NULL OR i.organization_id = $3)
                        GROUP BY o.id, o.name
                        ORDER BY total_revenue DESC
                        LIMIT 10
                    """
                    
                    client_data = await conn.fetch(client_query, start_dt, end_dt, organization_id)
                
                # Calculate key metrics
                total_revenue = sum(float(row['total_revenue']) for row in revenue_data)
                total_invoices = sum(row['invoice_count'] for row in revenue_data)
                
                # Growth calculation
                if len(revenue_data) >= 2:
                    current_period_revenue = float(revenue_data[-1]['total_revenue'])
                    previous_period_revenue = float(revenue_data[-2]['total_revenue'])
                    growth_rate = ((current_period_revenue - previous_period_revenue) / previous_period_revenue * 100) if previous_period_revenue > 0 else 0
                else:
                    growth_rate = 0
                
                analytics = {
                    'period': {
                        'start_date': start_dt.isoformat(),
                        'end_date': end_dt.isoformat(),
                        'granularity': granularity
                    },
                    'summary': {
                        'total_revenue': round(total_revenue, 2),
                        'total_invoices': total_invoices,
                        'avg_invoice_amount': round(total_revenue / total_invoices if total_invoices > 0 else 0, 2),
                        'growth_rate': round(growth_rate, 2)
                    },
                    'revenue_over_time': [
                        {
                            'period': row['period'].isoformat(),
                            'total_revenue': float(row['total_revenue']),
                            'invoice_count': row['invoice_count'],
                            'unique_clients': row['unique_clients'],
                            'avg_invoice_amount': float(row['avg_invoice_amount'])
                        }
                        for row in revenue_data
                    ],
                    'revenue_by_service': [
                        {
                            'service_name': row['service_name'],
                            'total_revenue': float(row['total_revenue']),
                            'client_count': row['client_count'],
                            'avg_amount': float(row['avg_amount']),
                            'percentage': round((float(row['total_revenue']) / total_revenue * 100) if total_revenue > 0 else 0, 1)
                        }
                        for row in service_data
                    ],
                    'top_clients': [
                        {
                            'client_name': row['client_name'],
                            'organization_id': row['organization_id'],
                            'total_revenue': float(row['total_revenue']),
                            'invoice_count': row['invoice_count'],
                            'avg_invoice_amount': float(row['avg_invoice_amount']),
                            'percentage': round((float(row['total_revenue']) / total_revenue * 100) if total_revenue > 0 else 0, 1)
                        }
                        for row in client_data
                    ],
                    'generated_at': datetime.utcnow().isoformat()
                }
                
                return analytics
                
            except Exception as e:
                logger.error(f"Revenue analytics failed: {e}")
                return {
                    'status': 'error',
                    'message': f'Failed to generate revenue analytics: {str(e)}'
                }
        
        @self.app.tool()
        async def get_client_profitability_analysis(
            organization_id: Optional[int] = None,
            include_costs: bool = True,
            period_months: int = 12
        ) -> Dict[str, Any]:
            """Analyze client profitability with cost considerations"""
            try:
                start_dt = datetime.utcnow() - timedelta(days=period_months * 30)
                end_dt = datetime.utcnow()
                
                async with self.db_pool.acquire() as conn:
                    # Client profitability query
                    profitability_query = """
                        WITH client_revenue AS (
                            SELECT 
                                o.id,
                                o.name,
                                o.tier,
                                SUM(i.total_amount) as total_revenue,
                                COUNT(DISTINCT i.id) as invoice_count,
                                AVG(i.total_amount) as avg_invoice_amount
                            FROM organizations o
                            LEFT JOIN invoices i ON o.id = i.organization_id 
                                AND i.invoice_date BETWEEN $1 AND $2
                                AND i.status != 'cancelled'
                            WHERE ($3::int IS NULL OR o.id = $3)
                            GROUP BY o.id, o.name, o.tier
                        ),
                        client_costs AS (
                            SELECT 
                                organization_id,
                                SUM(CASE WHEN category = 'support' THEN amount ELSE 0 END) as support_costs,
                                SUM(CASE WHEN category = 'infrastructure' THEN amount ELSE 0 END) as infrastructure_costs,
                                SUM(CASE WHEN category = 'license' THEN amount ELSE 0 END) as license_costs,
                                SUM(amount) as total_costs
                            FROM cost_allocations
                            WHERE date_incurred BETWEEN $1 AND $2
                            GROUP BY organization_id
                        ),
                        support_metrics AS (
                            SELECT 
                                organization_id,
                                COUNT(*) as ticket_count,
                                AVG(EXTRACT(EPOCH FROM (resolved_at - created_at))/3600) as avg_resolution_hours,
                                SUM(time_spent_hours) as total_support_hours
                            FROM support_tickets
                            WHERE created_at BETWEEN $1 AND $2
                            AND status = 'resolved'
                            GROUP BY organization_id
                        )
                        SELECT 
                            cr.id,
                            cr.name,
                            cr.tier,
                            COALESCE(cr.total_revenue, 0) as total_revenue,
                            COALESCE(cr.invoice_count, 0) as invoice_count,
                            COALESCE(cr.avg_invoice_amount, 0) as avg_invoice_amount,
                            COALESCE(cc.total_costs, 0) as total_costs,
                            COALESCE(cc.support_costs, 0) as support_costs,
                            COALESCE(cc.infrastructure_costs, 0) as infrastructure_costs,
                            COALESCE(cc.license_costs, 0) as license_costs,
                            COALESCE(sm.ticket_count, 0) as ticket_count,
                            COALESCE(sm.avg_resolution_hours, 0) as avg_resolution_hours,
                            COALESCE(sm.total_support_hours, 0) as total_support_hours,
                            (COALESCE(cr.total_revenue, 0) - COALESCE(cc.total_costs, 0)) as net_profit,
                            CASE 
                                WHEN COALESCE(cr.total_revenue, 0) > 0 
                                THEN ((COALESCE(cr.total_revenue, 0) - COALESCE(cc.total_costs, 0)) / COALESCE(cr.total_revenue, 0) * 100)
                                ELSE 0 
                            END as profit_margin
                        FROM client_revenue cr
                        LEFT JOIN client_costs cc ON cr.id = cc.organization_id
                        LEFT JOIN support_metrics sm ON cr.id = sm.organization_id
                        ORDER BY net_profit DESC
                    """
                    
                    profitability_data = await conn.fetch(profitability_query, start_dt, end_dt, organization_id)
                    
                    # Client tier analysis
                    tier_query = """
                        SELECT 
                            o.tier,
                            COUNT(DISTINCT o.id) as client_count,
                            AVG(COALESCE(revenue.total_revenue, 0)) as avg_revenue_per_client,
                            AVG(COALESCE(costs.total_costs, 0)) as avg_costs_per_client,
                            SUM(COALESCE(revenue.total_revenue, 0)) as total_tier_revenue,
                            SUM(COALESCE(costs.total_costs, 0)) as total_tier_costs
                        FROM organizations o
                        LEFT JOIN (
                            SELECT organization_id, SUM(total_amount) as total_revenue
                            FROM invoices 
                            WHERE invoice_date BETWEEN $1 AND $2 AND status != 'cancelled'
                            GROUP BY organization_id
                        ) revenue ON o.id = revenue.organization_id
                        LEFT JOIN (
                            SELECT organization_id, SUM(amount) as total_costs
                            FROM cost_allocations
                            WHERE date_incurred BETWEEN $1 AND $2
                            GROUP BY organization_id
                        ) costs ON o.id = costs.organization_id
                        WHERE o.tier IS NOT NULL
                        GROUP BY o.tier
                        ORDER BY total_tier_revenue DESC
                    """
                    
                    tier_data = await conn.fetch(tier_query, start_dt, end_dt)
                
                # Calculate aggregated metrics
                total_revenue = sum(float(row['total_revenue']) for row in profitability_data)
                total_costs = sum(float(row['total_costs']) for row in profitability_data) if include_costs else 0
                total_profit = total_revenue - total_costs
                overall_margin = (total_profit / total_revenue * 100) if total_revenue > 0 else 0
                
                analysis = {
                    'period': {
                        'start_date': start_dt.isoformat(),
                        'end_date': end_dt.isoformat(),
                        'period_months': period_months
                    },
                    'summary': {
                        'total_revenue': round(total_revenue, 2),
                        'total_costs': round(total_costs, 2) if include_costs else None,
                        'total_profit': round(total_profit, 2) if include_costs else None,
                        'overall_margin': round(overall_margin, 2) if include_costs else None,
                        'client_count': len(profitability_data)
                    },
                    'client_profitability': [
                        {
                            'organization_id': row['id'],
                            'name': row['name'],
                            'tier': row['tier'],
                            'total_revenue': float(row['total_revenue']),
                            'total_costs': float(row['total_costs']) if include_costs else None,
                            'net_profit': float(row['net_profit']) if include_costs else None,
                            'profit_margin': float(row['profit_margin']) if include_costs else None,
                            'invoice_count': row['invoice_count'],
                            'avg_invoice_amount': float(row['avg_invoice_amount']),
                            'support_metrics': {
                                'ticket_count': row['ticket_count'],
                                'avg_resolution_hours': float(row['avg_resolution_hours']),
                                'total_support_hours': float(row['total_support_hours'])
                            } if include_costs else None
                        }
                        for row in profitability_data
                    ],
                    'tier_analysis': [
                        {
                            'tier': row['tier'],
                            'client_count': row['client_count'],
                            'avg_revenue_per_client': float(row['avg_revenue_per_client']),
                            'avg_costs_per_client': float(row['avg_costs_per_client']) if include_costs else None,
                            'total_tier_revenue': float(row['total_tier_revenue']),
                            'total_tier_costs': float(row['total_tier_costs']) if include_costs else None,
                            'tier_profit_margin': round(((float(row['total_tier_revenue']) - float(row['total_tier_costs'])) / float(row['total_tier_revenue']) * 100) if float(row['total_tier_revenue']) > 0 else 0, 2) if include_costs else None
                        }
                        for row in tier_data
                    ],
                    'generated_at': datetime.utcnow().isoformat()
                }
                
                return analysis
                
            except Exception as e:
                logger.error(f"Client profitability analysis failed: {e}")
                return {
                    'status': 'error',
                    'message': f'Failed to generate client profitability analysis: {str(e)}'
                }
        
        @self.app.tool()
        async def get_compliance_dashboard(
            framework: Optional[str] = None,
            organization_id: Optional[int] = None
        ) -> Dict[str, Any]:
            """Get compliance status dashboard"""
            try:
                async with self.db_pool.acquire() as conn:
                    # Compliance overview
                    compliance_query = """
                        SELECT 
                            framework,
                            status,
                            COUNT(*) as count,
                            AVG(compliance_score) as avg_score
                        FROM compliance_assessments ca
                        WHERE ($1::text IS NULL OR framework = $1)
                        AND ($2::int IS NULL OR organization_id = $2)
                        AND assessment_date >= NOW() - INTERVAL '90 days'
                        GROUP BY framework, status
                        ORDER BY framework, status
                    """
                    
                    compliance_data = await conn.fetch(compliance_query, framework, organization_id)
                    
                    # Recent assessments
                    recent_query = """
                        SELECT 
                            ca.id,
                            ca.framework,
                            ca.status,
                            ca.compliance_score,
                            ca.assessment_date,
                            ca.next_assessment_date,
                            o.name as organization_name,
                            ca.findings_summary
                        FROM compliance_assessments ca
                        LEFT JOIN organizations o ON ca.organization_id = o.id
                        WHERE ($1::text IS NULL OR ca.framework = $1)
                        AND ($2::int IS NULL OR ca.organization_id = $2)
                        ORDER BY ca.assessment_date DESC
                        LIMIT 20
                    """
                    
                    recent_assessments = await conn.fetch(recent_query, framework, organization_id)
                    
                    # Control implementation status
                    controls_query = """
                        SELECT 
                            framework,
                            control_category,
                            status,
                            COUNT(*) as control_count,
                            AVG(effectiveness_score) as avg_effectiveness
                        FROM compliance_controls cc
                        WHERE ($1::text IS NULL OR framework = $1)
                        AND ($2::int IS NULL OR organization_id = $2)
                        GROUP BY framework, control_category, status
                        ORDER BY framework, control_category
                    """
                    
                    controls_data = await conn.fetch(controls_query, framework, organization_id)
                    
                    # Upcoming assessments
                    upcoming_query = """
                        SELECT 
                            ca.id,
                            ca.framework,
                            ca.next_assessment_date,
                            o.name as organization_name,
                            ca.assessment_type
                        FROM compliance_assessments ca
                        LEFT JOIN organizations o ON ca.organization_id = o.id
                        WHERE ca.next_assessment_date BETWEEN NOW() AND NOW() + INTERVAL '60 days'
                        AND ($1::text IS NULL OR ca.framework = $1)
                        AND ($2::int IS NULL OR ca.organization_id = $2)
                        ORDER BY ca.next_assessment_date
                    """
                    
                    upcoming_assessments = await conn.fetch(upcoming_query, framework, organization_id)
                
                # Calculate compliance metrics
                total_assessments = len(recent_assessments)
                compliant_assessments = len([a for a in recent_assessments if a['status'] == 'compliant'])
                compliance_rate = (compliant_assessments / total_assessments * 100) if total_assessments > 0 else 0
                
                dashboard = {
                    'summary': {
                        'total_assessments': total_assessments,
                        'compliant_assessments': compliant_assessments,
                        'compliance_rate': round(compliance_rate, 1),
                        'avg_compliance_score': round(
                            sum(float(a['compliance_score']) for a in recent_assessments if a['compliance_score']) / 
                            len([a for a in recent_assessments if a['compliance_score']]) if len([a for a in recent_assessments if a['compliance_score']]) > 0 else 0, 1
                        ),
                        'upcoming_assessments': len(upcoming_assessments)
                    },
                    'compliance_by_framework': {},
                    'recent_assessments': [
                        {
                            'id': row['id'],
                            'framework': row['framework'],
                            'status': row['status'],
                            'compliance_score': float(row['compliance_score']) if row['compliance_score'] else None,
                            'assessment_date': row['assessment_date'].isoformat(),
                            'next_assessment_date': row['next_assessment_date'].isoformat() if row['next_assessment_date'] else None,
                            'organization_name': row['organization_name'],
                            'findings_summary': row['findings_summary']
                        }
                        for row in recent_assessments
                    ],
                    'upcoming_assessments': [
                        {
                            'id': row['id'],
                            'framework': row['framework'],
                            'next_assessment_date': row['next_assessment_date'].isoformat(),
                            'organization_name': row['organization_name'],
                            'assessment_type': row['assessment_type'],
                            'days_until': (row['next_assessment_date'] - datetime.utcnow().date()).days
                        }
                        for row in upcoming_assessments
                    ],
                    'control_implementation': {},
                    'generated_at': datetime.utcnow().isoformat()
                }
                
                # Group compliance data by framework
                for row in compliance_data:
                    framework_name = row['framework']
                    if framework_name not in dashboard['compliance_by_framework']:
                        dashboard['compliance_by_framework'][framework_name] = {
                            'total_count': 0,
                            'status_breakdown': {},
                            'avg_score': 0
                        }
                    
                    dashboard['compliance_by_framework'][framework_name]['total_count'] += row['count']
                    dashboard['compliance_by_framework'][framework_name]['status_breakdown'][row['status']] = row['count']
                    dashboard['compliance_by_framework'][framework_name]['avg_score'] = round(float(row['avg_score']), 1)
                
                # Group control data
                for row in controls_data:
                    framework_name = row['framework']
                    if framework_name not in dashboard['control_implementation']:
                        dashboard['control_implementation'][framework_name] = {}
                    
                    category = row['control_category']
                    if category not in dashboard['control_implementation'][framework_name]:
                        dashboard['control_implementation'][framework_name][category] = {
                            'total_controls': 0,
                            'status_breakdown': {},
                            'avg_effectiveness': 0
                        }
                    
                    dashboard['control_implementation'][framework_name][category]['total_controls'] += row['control_count']
                    dashboard['control_implementation'][framework_name][category]['status_breakdown'][row['status']] = row['control_count']
                    dashboard['control_implementation'][framework_name][category]['avg_effectiveness'] = round(float(row['avg_effectiveness']), 1)
                
                return dashboard
                
            except Exception as e:
                logger.error(f"Compliance dashboard generation failed: {e}")
                return {
                    'status': 'error',
                    'message': f'Failed to generate compliance dashboard: {str(e)}'
                }
        
        @self.app.tool()
        async def get_operational_metrics(
            metric_type: str = "all",
            period_days: int = 30,
            organization_id: Optional[int] = None
        ) -> Dict[str, Any]:
            """Get operational metrics and KPIs"""
            try:
                start_dt = datetime.utcnow() - timedelta(days=period_days)
                end_dt = datetime.utcnow()
                
                metrics = {}
                
                async with self.db_pool.acquire() as conn:
                    if metric_type in ["all", "tickets"]:
                        # Support ticket metrics
                        ticket_query = """
                            SELECT 
                                COUNT(*) as total_tickets,
                                COUNT(CASE WHEN status = 'resolved' THEN 1 END) as resolved_tickets,
                                COUNT(CASE WHEN priority = 'critical' THEN 1 END) as critical_tickets,
                                COUNT(CASE WHEN priority = 'high' THEN 1 END) as high_priority_tickets,
                                AVG(EXTRACT(EPOCH FROM (resolved_at - created_at))/3600) as avg_resolution_time_hours,
                                AVG(satisfaction_rating) as avg_satisfaction_rating
                            FROM support_tickets
                            WHERE created_at BETWEEN $1 AND $2
                            AND ($3::int IS NULL OR organization_id = $3)
                        """
                        
                        ticket_data = await conn.fetchrow(ticket_query, start_dt, end_dt, organization_id)
                        
                        resolution_rate = (ticket_data['resolved_tickets'] / ticket_data['total_tickets'] * 100) if ticket_data['total_tickets'] > 0 else 0
                        
                        metrics['support_tickets'] = {
                            'total_tickets': ticket_data['total_tickets'],
                            'resolved_tickets': ticket_data['resolved_tickets'],
                            'resolution_rate': round(resolution_rate, 1),
                            'critical_tickets': ticket_data['critical_tickets'],
                            'high_priority_tickets': ticket_data['high_priority_tickets'],
                            'avg_resolution_time_hours': round(float(ticket_data['avg_resolution_time_hours']) if ticket_data['avg_resolution_time_hours'] else 0, 1),
                            'avg_satisfaction_rating': round(float(ticket_data['avg_satisfaction_rating']) if ticket_data['avg_satisfaction_rating'] else 0, 1)
                        }
                    
                    if metric_type in ["all", "monitoring"]:
                        # System monitoring metrics
                        monitoring_query = """
                            SELECT 
                                COUNT(*) as total_alerts,
                                COUNT(CASE WHEN severity = 'critical' THEN 1 END) as critical_alerts,
                                COUNT(CASE WHEN status = 'resolved' THEN 1 END) as resolved_alerts,
                                AVG(EXTRACT(EPOCH FROM (resolved_at - created_at))/60) as avg_resolution_time_minutes,
                                COUNT(DISTINCT asset_id) as affected_assets
                            FROM monitoring_alerts
                            WHERE created_at BETWEEN $1 AND $2
                            AND ($3::int IS NULL OR organization_id = $3)
                        """
                        
                        monitoring_data = await conn.fetchrow(monitoring_query, start_dt, end_dt, organization_id)
                        
                        metrics['monitoring'] = {
                            'total_alerts': monitoring_data['total_alerts'],
                            'critical_alerts': monitoring_data['critical_alerts'],
                            'resolved_alerts': monitoring_data['resolved_alerts'],
                            'resolution_rate': round((monitoring_data['resolved_alerts'] / monitoring_data['total_alerts'] * 100) if monitoring_data['total_alerts'] > 0 else 0, 1),
                            'avg_resolution_time_minutes': round(float(monitoring_data['avg_resolution_time_minutes']) if monitoring_data['avg_resolution_time_minutes'] else 0, 1),
                            'affected_assets': monitoring_data['affected_assets']
                        }
                    
                    if metric_type in ["all", "performance"]:
                        # Performance metrics
                        performance_query = """
                            SELECT 
                                AVG(cpu_usage) as avg_cpu_usage,
                                AVG(memory_usage) as avg_memory_usage,
                                AVG(disk_usage) as avg_disk_usage,
                                AVG(network_latency) as avg_network_latency,
                                AVG(uptime_percentage) as avg_uptime,
                                COUNT(DISTINCT asset_id) as monitored_assets
                            FROM performance_metrics
                            WHERE recorded_at BETWEEN $1 AND $2
                            AND ($3::int IS NULL OR organization_id = $3)
                        """
                        
                        performance_data = await conn.fetchrow(performance_query, start_dt, end_dt, organization_id)
                        
                        metrics['performance'] = {
                            'avg_cpu_usage': round(float(performance_data['avg_cpu_usage']) if performance_data['avg_cpu_usage'] else 0, 1),
                            'avg_memory_usage': round(float(performance_data['avg_memory_usage']) if performance_data['avg_memory_usage'] else 0, 1),
                            'avg_disk_usage': round(float(performance_data['avg_disk_usage']) if performance_data['avg_disk_usage'] else 0, 1),
                            'avg_network_latency': round(float(performance_data['avg_network_latency']) if performance_data['avg_network_latency'] else 0, 2),
                            'avg_uptime': round(float(performance_data['avg_uptime']) if performance_data['avg_uptime'] else 0, 2),
                            'monitored_assets': performance_data['monitored_assets']
                        }
                    
                    if metric_type in ["all", "sla"]:
                        # SLA metrics
                        sla_query = """
                            SELECT 
                                sla_type,
                                AVG(actual_value) as avg_actual_value,
                                AVG(target_value) as avg_target_value,
                                COUNT(CASE WHEN actual_value >= target_value THEN 1 END) as met_count,
                                COUNT(*) as total_count
                            FROM sla_metrics
                            WHERE measurement_date BETWEEN $1 AND $2
                            AND ($3::int IS NULL OR organization_id = $3)
                            GROUP BY sla_type
                        """
                        
                        sla_data = await conn.fetch(sla_query, start_dt, end_dt, organization_id)
                        
                        metrics['sla'] = {}
                        for row in sla_data:
                            sla_compliance = (row['met_count'] / row['total_count'] * 100) if row['total_count'] > 0 else 0
                            metrics['sla'][row['sla_type']] = {
                                'avg_actual_value': round(float(row['avg_actual_value']), 2),
                                'avg_target_value': round(float(row['avg_target_value']), 2),
                                'compliance_rate': round(sla_compliance, 1),
                                'measurements': row['total_count']
                            }
                
                operational_metrics = {
                    'period': {
                        'start_date': start_dt.isoformat(),
                        'end_date': end_dt.isoformat(),
                        'period_days': period_days
                    },
                    'metrics': metrics,
                    'generated_at': datetime.utcnow().isoformat()
                }
                
                return operational_metrics
                
            except Exception as e:
                logger.error(f"Operational metrics generation failed: {e}")
                return {
                    'status': 'error',
                    'message': f'Failed to generate operational metrics: {str(e)}'
                }
        
        @self.app.tool()
        async def generate_chart_data(
            chart_type: str,
            data_source: str,
            parameters: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Generate chart data for dashboard visualizations"""
            try:
                chart_config = {
                    'chart_type': chart_type,
                    'data_source': data_source,
                    'parameters': parameters
                }
                
                if data_source == "revenue" and chart_type == "line":
                    # Revenue over time line chart
                    revenue_data = await get_revenue_analytics(
                        start_date=parameters.get('start_date'),
                        end_date=parameters.get('end_date'),
                        granularity=parameters.get('granularity', 'monthly')
                    )
                    
                    chart_data = {
                        'x': [item['period'] for item in revenue_data['revenue_over_time']],
                        'y': [item['total_revenue'] for item in revenue_data['revenue_over_time']],
                        'title': 'Revenue Over Time',
                        'x_label': 'Period',
                        'y_label': 'Revenue ($)'
                    }
                    
                elif data_source == "revenue" and chart_type == "pie":
                    # Service revenue pie chart
                    revenue_data = await get_revenue_analytics(
                        start_date=parameters.get('start_date'),
                        end_date=parameters.get('end_date')
                    )
                    
                    chart_data = {
                        'labels': [item['service_name'] for item in revenue_data['revenue_by_service'][:10]],
                        'values': [item['total_revenue'] for item in revenue_data['revenue_by_service'][:10]],
                        'title': 'Revenue by Service Type'
                    }
                    
                elif data_source == "compliance" and chart_type == "bar":
                    # Compliance status bar chart
                    compliance_data = await get_compliance_dashboard(
                        framework=parameters.get('framework')
                    )
                    
                    frameworks = list(compliance_data['compliance_by_framework'].keys())
                    compliance_rates = []
                    
                    for framework in frameworks:
                        data = compliance_data['compliance_by_framework'][framework]
                        compliant = data['status_breakdown'].get('compliant', 0)
                        total = data['total_count']
                        rate = (compliant / total * 100) if total > 0 else 0
                        compliance_rates.append(rate)
                    
                    chart_data = {
                        'x': frameworks,
                        'y': compliance_rates,
                        'title': 'Compliance Rates by Framework',
                        'x_label': 'Framework',
                        'y_label': 'Compliance Rate (%)'
                    }
                    
                else:
                    return {
                        'status': 'error',
                        'message': f'Unsupported chart configuration: {chart_type} for {data_source}'
                    }
                
                return {
                    'chart_config': chart_config,
                    'chart_data': chart_data,
                    'generated_at': datetime.utcnow().isoformat(),
                    'status': 'success'
                }
                
            except Exception as e:
                logger.error(f"Chart data generation failed: {e}")
                return {
                    'status': 'error',
                    'message': f'Failed to generate chart data: {str(e)}'
                }
        
        @self.app.tool()
        async def export_analytics_data(
            data_type: str,
            format: str = "json",
            parameters: Optional[Dict[str, Any]] = None
        ) -> Dict[str, Any]:
            """Export analytics data in various formats"""
            try:
                parameters = parameters or {}
                
                if data_type == "revenue":
                    data = await get_revenue_analytics(**parameters)
                elif data_type == "profitability":
                    data = await get_client_profitability_analysis(**parameters)
                elif data_type == "compliance":
                    data = await get_compliance_dashboard(**parameters)
                elif data_type == "operational":
                    data = await get_operational_metrics(**parameters)
                else:
                    return {
                        'status': 'error',
                        'message': f'Unknown data type: {data_type}'
                    }
                
                if format.lower() == "json":
                    return {
                        'data': data,
                        'format': 'json',
                        'data_type': data_type,
                        'exported_at': datetime.utcnow().isoformat(),
                        'status': 'success'
                    }
                elif format.lower() == "csv":
                    # Convert data to CSV format (simplified)
                    csv_data = "Generated CSV data would be here"
                    return {
                        'data': csv_data,
                        'format': 'csv',
                        'data_type': data_type,
                        'exported_at': datetime.utcnow().isoformat(),
                        'status': 'success'
                    }
                else:
                    return {
                        'status': 'error',
                        'message': f'Unsupported export format: {format}'
                    }
                    
            except Exception as e:
                logger.error(f"Data export failed: {e}")
                return {
                    'status': 'error',
                    'message': f'Failed to export data: {str(e)}'
                }
    
    def setup_resources(self):
        """Setup FastMCP resources"""
        
        @self.app.resource("dashboard-templates")
        async def get_dashboard_templates() -> List[Resource]:
            """Get available dashboard templates"""
            templates = [
                Resource(
                    uri="template://executive-dashboard",
                    name="Executive Dashboard",
                    description="High-level KPIs and metrics for executives",
                    mimeType="application/json"
                ),
                Resource(
                    uri="template://operations-dashboard", 
                    name="Operations Dashboard",
                    description="Operational metrics and performance indicators",
                    mimeType="application/json"
                ),
                Resource(
                    uri="template://financial-dashboard",
                    name="Financial Dashboard", 
                    description="Revenue, profitability, and financial analytics",
                    mimeType="application/json"
                ),
                Resource(
                    uri="template://compliance-dashboard",
                    name="Compliance Dashboard",
                    description="Compliance status and regulatory metrics",
                    mimeType="application/json"
                )
            ]
            return templates
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.db_pool:
                await self.db_pool.close()
            if self.redis_client:
                await self.redis_client.close()
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

analytics_dashboard = FastMCPAnalyticsDashboard()

async def main():
    """Main entry point"""
    try:
        await analytics_dashboard.initialize()
        async with analytics_dashboard.app.run_server() as server:
            await server.run()
    except KeyboardInterrupt:
        logger.info("Shutting down analytics dashboard...")
    except Exception as e:
        logger.error(f"Analytics dashboard error: {e}")
    finally:
        await analytics_dashboard.cleanup()

if __name__ == "__main__":
    asyncio.run(main())