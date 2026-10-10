FROM python:3.14-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends bash git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN mkdir -p /app/workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV AGENT_WORKSPACE=/app/workspace \
    PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "streamlit run frontend.py --server.port=${PORT:-7860} --server.address=0.0.0.0 --server.runOnSave=true"]
