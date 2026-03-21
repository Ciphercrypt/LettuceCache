FROM python:3.12-slim

WORKDIR /app

# Install system deps for sentence-transformers (tokenizers build)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# Pre-download model during image build so cold-start is instant
ARG MODEL_NAME=all-MiniLM-L6-v2
ENV MODEL_NAME=${MODEL_NAME}
RUN python - <<'EOF'
from sentence_transformers import SentenceTransformer
import os
SentenceTransformer(os.environ["MODEL_NAME"])
EOF

EXPOSE 8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "2"]
