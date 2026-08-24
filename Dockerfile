# Backend Dockerfile
FROM python:3.12-slim-bookworm

WORKDIR /app

# Install system dependencies (ffmpeg is required for video processing)
# Debian's plain-HTTP CDN endpoint is intermittently unreachable on some
# Docker Desktop/AutoDL networks. HTTPS uses the same signed repository while
# avoiding that network path.
RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get -o Acquire::Retries=5 update \
    && apt-get -o Acquire::Retries=5 install -y --no-install-recommends ffmpeg

# Copy project files
COPY pyproject.toml README.md ./
COPY src ./src

# Install dependencies and the project itself. BuildKit keeps downloaded wheels
# outside the final image so source-only rebuilds remain fast.
RUN --mount=type=cache,target=/root/.cache/pip pip install .

# Expose API port
EXPOSE 8001

# Command to run the server
CMD ["uvicorn", "src.video_workflow.server.app:app", "--host", "0.0.0.0", "--port", "8001"]
