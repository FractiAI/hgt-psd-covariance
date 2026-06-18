FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements_lock.txt .
RUN pip install --no-cache-dir -r requirements_lock.txt

COPY . .
ENV PYTHONPATH=/workspace/src/python

CMD ["bash", "verify_pipeline.sh"]
