# MSP FastMCP Services

> **Enterprise-grade AI services built with FastMCP architecture for Managed Service Providers**

A comprehensive suite of AI-powered services designed specifically for MSP operations, built using the FastMCP (Model Context Protocol) framework for maximum interoperability and performance.

## 🏗️ Architecture Overview

This system converts all traditional web services to FastMCP-based architecture, providing:

- **Unified Protocol**: All services communicate via the Model Context Protocol (MCP)
- **Enhanced AI Integration**: Native support for AI tools and resources
- **Scalable Design**: Microservices architecture with container orchestration
- **Real-time Capabilities**: WebSocket-based communication for instant responses

## 🚀 FastMCP Services

### 1. **RAG Knowledge Management System** (`fastmcp_rag_system.py`)
- **Purpose**: Intelligent knowledge base with semantic search
- **Key Features**:
  - Vector-based document embedding and retrieval
  - AI-powered answer generation
  - Multi-category knowledge organization
  - Real-time document indexing

**FastMCP Tools**:
- `search_knowledge_base()` - Semantic document search
- `add_document()` - Add new knowledge articles
- `generate_answer()` - AI-powered Q&A
- `update_document()` - Modify existing content

### 2. **AI Calling Assistant** (`fastmcp_calling_assistant.py`)
- **Purpose**: Automated voice calling with AI conversation management
- **Key Features**:
  - Intelligent call scheduling and execution
  - Real-time speech processing and response
  - Multi-type call support (sales, support, reminders)
  - Call outcome analysis and follow-up automation

**FastMCP Tools**:
- `schedule_call()` - Schedule AI-powered calls
- `execute_call()` - Perform automated calling
- `get_call_analytics()` - Calling performance metrics
- `cancel_call()` - Cancel scheduled calls

### 3. **Intelligent Chatbot** (`fastmcp_intelligent_chatbot.py`)
- **Purpose**: Multi-channel conversational AI for customer engagement
- **Key Features**:
  - Advanced intent classification and entity extraction
  - Context-aware conversation management
  - Seamless escalation to human agents
  - Integration with support ticket system

**FastMCP Tools**:
- `start_conversation()` - Initialize new chat sessions
- `continue_conversation()` - Process ongoing interactions
- `classify_intent()` - Understand customer needs
- `create_support_ticket()` - Generate tickets from chats

### 4. **Analytics Dashboard** (`fastmcp_analytics_dashboard.py`)
- **Purpose**: Comprehensive business intelligence and reporting
- **Key Features**:
  - Real-time revenue and profitability analysis
  - Client satisfaction and retention metrics
  - Compliance dashboard with automated reporting
  - Predictive analytics for business optimization

**FastMCP Tools**:
- `get_revenue_analytics()` - Financial performance data
- `get_client_profitability_analysis()` - Customer value metrics
- `get_compliance_dashboard()` - Regulatory status overview
- `generate_chart_data()` - Visualization data export

### 5. **Service Orchestrator** (`msp_service_orchestrator.py`)
- **Purpose**: Central coordination hub for all FastMCP services
- **Key Features**:
  - Cross-service workflow automation
  - Unified service monitoring and health checks
  - Intelligent load balancing and failover
  - Consolidated analytics and reporting

**FastMCP Tools**:
- `get_service_status()` - Monitor all services
- `execute_cross_service_workflow()` - Multi-service processes
- `get_unified_analytics()` - Comprehensive insights

## 🔧 Installation & Deployment

### Prerequisites

- Docker & Docker Compose
- Python 3.11+
- PostgreSQL 15+ with pgvector extension
- Redis 7+
- OpenAI API key
- Twilio account (for calling features)

### Quick Start

1. **Clone the repository**:
```bash
git clone https://github.com/mynewopportunities/msp-fastmcp-services.git
cd msp-fastmcp-services
```

2. **Set up environment variables**:
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. **Deploy with Docker Compose**:
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Check service status
docker-compose ps
```

4. **Initialize the database**:
```bash
# Run database migrations
docker-compose exec msp-orchestrator python init_db.py
```

## 🎯 Key FastMCP Advantages

### Traditional vs FastMCP Architecture

| Aspect | Traditional | FastMCP |
|--------|-------------|---------|
| **Communication** | REST/HTTP APIs | Model Context Protocol |
| **AI Integration** | Custom implementations | Native MCP tools/resources |
| **Interoperability** | Service-specific | Standardized protocol |
| **Tool Discovery** | Manual documentation | Automatic introspection |
| **Context Sharing** | Limited | Rich context passing |
| **Real-time** | Polling/WebHooks | Native streaming |

### Benefits of FastMCP

1. **Unified Protocol**: All services speak the same language
2. **Enhanced AI Capabilities**: Native support for AI tools and resources
3. **Better Observability**: Built-in service introspection and monitoring
4. **Easier Integration**: Standardized tool and resource interfaces
5. **Future-Proof**: Designed for the AI-first era

## 📊 Service Architecture

```
┌─────────────────────────────────────────────────┐
│              FastMCP Orchestrator               │
│         (Service Coordination Hub)              │
└─────────────────────┬───────────────────────────┘
                      │
    ┌─────────────────┼─────────────────┐
    │                 │                 │
    ▼                 ▼                 ▼
┌─────────┐   ┌─────────────┐   ┌─────────────┐
│   RAG   │   │   Calling   │   │   Chatbot   │
│ System  │   │ Assistant   │   │   Service   │
└─────────┘   └─────────────┘   └─────────────┘
    │                 │                 │
    └─────────────────┼─────────────────┘
                      │
                      ▼
            ┌─────────────────┐
            │    Analytics    │
            │    Dashboard    │
            └─────────────────┘
```

## 🚀 Usage Examples

### Starting a Conversation
```python
from mcp import Client

# Connect to chatbot service
client = Client("ws://localhost:8003/mcp")

# Start conversation
result = await client.call_tool("start_conversation", {
    "user_message": "I need help with my server backup",
    "channel": "web",
    "user_id": "customer_123"
})

print(f"Session ID: {result['session_id']}")
print(f"Response: {result['response']}")
```

### Searching Knowledge Base
```python
# Connect to RAG service
rag_client = Client("ws://localhost:8001/mcp")

# Search for information
results = await rag_client.call_tool("search_knowledge_base", {
    "query": "server backup best practices",
    "category": "technical",
    "limit": 5
})

for doc in results:
    print(f"Title: {doc['title']}")
    print(f"Relevance: {doc['similarity_score']}")
```

### Scheduling AI Calls
```python
# Connect to calling service
call_client = Client("ws://localhost:8002/mcp")

# Schedule a call
call_result = await call_client.call_tool("schedule_call", {
    "contact_id": 123,
    "call_type": "customer_support",
    "scheduled_time": "2024-01-15T14:30:00Z",
    "notes": "Follow up on backup issue"
})

print(f"Call ID: {call_result['call_id']}")
```

## 📈 Monitoring & Observability

### Service Health Monitoring
- **Prometheus**: Metrics collection and alerting
- **Grafana**: Real-time dashboards and visualization
- **ELK Stack**: Centralized logging and log analysis
- **Health Checks**: Built-in service health endpoints

### Accessing Dashboards
- **Grafana**: http://localhost:3000 (admin/password)
- **Prometheus**: http://localhost:9090
- **Kibana**: http://localhost:5601
- **RabbitMQ Management**: http://localhost:15672

## 🔒 Security Features

- **Encryption**: All sensitive data encrypted at rest and in transit
- **Authentication**: Multi-factor authentication support
- **Authorization**: Role-based access control (RBAC)
- **Audit Logging**: Comprehensive audit trails for all operations
- **Data Privacy**: GDPR/CCPA compliant data handling

## 🛠️ Development

### Development Setup
```bash
# Create development environment
python -m venv fastmcp-dev
source fastmcp-dev/bin/activate  # On Windows: fastmcp-dev\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run services locally
python msp_service_orchestrator.py
```

### Code Quality
- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking
- **pre-commit**: Git hooks for code quality

## 🚀 Deployment

### Production Deployment

For production deployment:

```bash
# Start all services
docker-compose up -d

# Scale services as needed
docker-compose up -d --scale worker=3
```

## 📄 License

This project is licensed under the MIT License.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

**Built with ❤️ for the MSP community using FastMCP architecture**