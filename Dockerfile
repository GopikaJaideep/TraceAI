FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt insightface \
    && python -m spacy download en_core_web_sm

COPY traceai ./traceai
COPY dashboard ./dashboard
COPY scripts ./scripts

ENV TRACEAI_DATA_DIR=/data
EXPOSE 8000 8501
CMD ["uvicorn", "traceai.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
