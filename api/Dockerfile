# syntax=docker/dockerfile:1

# ARG PYTHON_VERSION=3.10.12
FROM python:3.13-slim

# FROM tensorflow/tensorflow:latest-gpu

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr.
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && \
    apt-get install -y cmake build-essential  pkg-config libgoogle-perftools-dev && \
    rm -rf /var/lib/apt/lists/*

# Create a non-privileged user that the app will run under.
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# Copy requirements.txt and install dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir gunicorn

# Copy the source code into the container.
COPY app ./
RUN rm -rf /app/saved_models/t5_wiki_pretrain_fast/checkpoint-*
# COPY .env ./

# Switch to the non-privileged user to run the application.
USER appuser

# Expose the port that the application listens on.
EXPOSE 80

# Run the application.
CMD ["uvicorn", "fastapi_t5:app", "--host", "0.0.0.0", "--port", "80"]
# CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "fastapi_t5:app", "-b", "0.0.0.0:80", "--workers", "4", "--timeout", "120"]
