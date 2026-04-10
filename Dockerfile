FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source files
COPY openenv.yaml .
COPY server.py .
COPY adem_env.py .
COPY adem/ ./adem/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=5s --timeout=10s --start-period=15s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Run server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]