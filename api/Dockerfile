FROM python:3.12-slim

WORKDIR /app

RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

COPY requirements.txt .

RUN apt-get update && apt-get install -y libgomp1 netcat-openbsd
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

USER appuser