FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        postgresql-client \
        redis-tools \
        ffmpeg \
        portaudio19-dev \
        python3-dev \
        gcc \
        g++ \
        pkg-config \
        libffi-dev \
        libssl-dev \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs /app/models /app/data

# Set permissions
RUN chmod +x *.py

# Create non-root user
RUN groupadd -r mspuser && useradd -r -g mspuser mspuser
RUN chown -R mspuser:mspuser /app
USER mspuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "msp_service_orchestrator.py"]