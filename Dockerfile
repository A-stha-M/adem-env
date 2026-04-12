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
COPY server/ ./server/
COPY env/ ./env/
COPY graders/ ./graders/
COPY tasks/ ./tasks/
COPY adem_env.py .
COPY models.py .

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=5s --timeout=10s --start-period=15s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

# Run server
CMD ["python", "-m", "server.app"]