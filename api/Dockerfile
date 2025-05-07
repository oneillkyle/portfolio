# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.10.12
FROM python:${PYTHON_VERSION}-slim AS base

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
COPY . .

# Expose the port that the application listens on.
EXPOSE 5000

# Run the application.
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info", "--workers", "1", "--reload"]
CMD ["fastapi", "dev", "app/main.py"]