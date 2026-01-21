# Backend Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (ffmpeg is required for video processing)
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md ./
COPY src ./src

# Install dependencies and the project itself
# We use pip to install the current directory as a package
RUN pip install --no-cache-dir .

# Expose API port
EXPOSE 8001

# Command to run the server
CMD ["uvicorn", "src.video_workflow.server.app:app", "--host", "0.0.0.0", "--port", "8001"]
