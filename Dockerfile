# ── Base Image ─────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Prevents Python from writing pyc files and buffering stdout/stderr
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8501

# Install system dependencies (curl for container healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root app user for security
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install Python dependencies first (leverages Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY scripbox_kb/ ./scripbox_kb/
COPY app.py ./app.py
COPY ingest.py ./ingest.py
COPY scraper.py ./scraper.py
COPY build_index.py ./build_index.py
COPY pyproject.toml ./pyproject.toml

# Set permissions for non-root execution
RUN chown -R appuser:appuser /app

USER appuser

# Expose default Streamlit port
EXPOSE 8501

# Built-in Streamlit health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

# Start Streamlit in production headless mode
ENTRYPOINT ["streamlit", "run", "app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true", \
    "--browser.gatherUsageStats=false"]
