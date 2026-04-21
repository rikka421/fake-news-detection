# ──────────────────────────────────────────────
# Fake-News Detection — inference-only image
# Usage:
#   docker build -t fakenews-api .
#   docker run -p 8000:8000 fakenews-api
# ──────────────────────────────────────────────
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install only inference dependencies
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Copy source and pre-trained model artifact
COPY src/ src/
COPY models/serving/ models/serving/

# Install the package (no extra deps — uses already-installed reqs)
COPY setup.py .
COPY README.md .
RUN pip install --no-cache-dir --no-deps -e .

# Expose API port
EXPOSE 8000

# Start the inference server (model loaded from models/serving/*.joblib)
CMD ["python", "-m", "data_analysis_progress.api"]
