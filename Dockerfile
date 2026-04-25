# =============================================================================
# Dockerfile — Multi-stage production image for the Answer Sheet Validator.
# Stage 1 (builder): installs Python dependencies into a virtual environment.
# Stage 2 (runtime): copies only the venv + app code for a lean final image.
# Tesseract OCR is installed as a system dependency.
# SECURITY: runs as non-root user; no credentials baked into the image.
# =============================================================================

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /install

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install dependencies into a virtual environment
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt


# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Install Tesseract OCR (system dependency for pytesseract)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Make venv binaries available on PATH
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copy application source
COPY app/       ./app/
COPY model/     ./model/
COPY data/      ./data/

# Create uploads directory (images are deleted after processing)
RUN mkdir -p uploads

# ── Security: run as non-root user ──────────────────────────────────────────
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser
USER appuser

# Expose Flask port
EXPOSE 5000

# Environment defaults (override at runtime with -e or --env-file)
ENV FLASK_APP=app \
    FLASK_ENV=production \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Start the Flask application
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=5000"]
