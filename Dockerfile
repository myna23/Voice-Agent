FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Smaller default than the .env.example's "small" — Render's free tier is
# 512MB RAM total for the whole container, and "small" cuts that close once
# FastAPI/uvicorn/ctranslate2 overhead is included. Drop to "tiny" via an
# env var on the host if this still OOMs; bump back up on a paid tier.
ENV LOCAL_WHISPER_MODEL=base

# Render sets $PORT at runtime; default to 7860 for local `docker run` /
# other hosts (e.g. Hugging Face Spaces' Docker SDK) that don't set it.
EXPOSE 7860
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]
