# Dockerfile for Agentic RAG
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create documents directory
RUN mkdir -p documents

# Expose ports
EXPOSE 8000 8501

# Default command runs the API
CMD ["python", "agentic_rag.py"]