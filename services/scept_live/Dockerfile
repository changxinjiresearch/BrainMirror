FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY services/scept_live/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY brainmirror /app/brainmirror
COPY scripts /app/scripts
COPY services /app/services
EXPOSE 8000
CMD ["uvicorn","services.scept_live.api:app","--host","0.0.0.0","--port","8000"]
