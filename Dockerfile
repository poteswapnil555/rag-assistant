FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY eval ./eval
COPY scripts ./scripts
COPY data/sample ./data/sample
ENV PYTHONPATH=/app/src
# The embedding model is downloaded on first real run (needs internet). Tests/mock mode need no download.
ENTRYPOINT ["python", "-m", "rag_assistant"]
