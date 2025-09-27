FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create necessary directories
RUN mkdir -p tennis_data_enhanced tennis_models betting_data logs cache

# Create user
RUN useradd --create-home --shell /bin/bash app
RUN chown -R app:app /app

# Fix logs permissions specifically - ensure app user can write to logs
RUN chmod 755 /app/logs
RUN chown app:app /app/logs
USER app

# Configure environment variables
ENV FLASK_ENV=production
ENV PYTHONPATH=/app

# Expose port
EXPOSE 5001

# Container health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5001/api/health || exit 1

# Start application (using main.py instead of gunicorn for integrated service)
CMD ["python", "main.py"]
