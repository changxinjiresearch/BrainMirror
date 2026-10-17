FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 SCEPT_MODEL_CACHE=/var/cache/scept
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY services/scept_live/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu 'torch==2.6.0+cpu' \
 && pip install --no-cache-dir -r /app/requirements.txt
COPY brainmirror /app/brainmirror
COPY scripts /app/scripts
COPY services /app/services
EXPOSE 8000
CMD ["uvicorn","services.scept_live.api:app","--host","0.0.0.0","--port","8000"]
