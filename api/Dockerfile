# syntax=docker/dockerfile:1

# ARG PYTHON_VERSION=3.10.12
FROM python:3.14-slim

# FROM tensorflow/tensorflow:latest-gpu

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr.
ENV PYTHONUNBUFFERED=1

WORKDIR /app

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
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install -r requirements.txt --no-cache-dir

# Switch to the non-privileged user to run the application.
USER appuser

# Copy the source code into the container.
COPY app ./

# Expose the port that the application listens on.
EXPOSE 8000

# Run the application.
CMD ["uvicorn", "app.fastapi_t5:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
# CMD ["fastapi", "dev", "app/main.py"]