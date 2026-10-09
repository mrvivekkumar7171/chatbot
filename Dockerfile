FROM python:3.14-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends bash git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN mkdir -p /app/workspace

COPY requirements.txt .
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV AGENT_WORKSPACE=/app/workspace
EXPOSE 8501

CMD ["streamlit", "run", "frontend.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.runOnSave=true"]
