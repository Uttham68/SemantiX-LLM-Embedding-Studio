FROM python:3.11-slim

WORKDIR /app

# Configure environment for fast, permission-safe cloud execution
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/tmp/huggingface \
    PORT=7860

# Install dependencies (uses CPU-only PyTorch to keep image small & fast)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-cache the primary Transformer bi-encoder so cloud startup is instant
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Copy application code, web studio assets, dataset, and precomputed outputs
COPY . .

# Ensure writable permissions for non-root container runtimes (e.g. Hugging Face Spaces)
RUN chmod -R 777 /app /tmp/huggingface

EXPOSE 7860

# Launch the FastAPI SemantiX Web Studio (respects $PORT on Render/Railway or defaults to 7860 on HF Spaces)
CMD ["sh", "-c", "uvicorn web_app:app --host 0.0.0.0 --port ${PORT:-7860}"]

