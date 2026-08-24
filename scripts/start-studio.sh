#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "缺少 .venv，请先按 README 完成 Python 环境安装。"
  exit 1
fi

if [[ ! -d "${PROJECT_DIR}/frontend/node_modules" ]]; then
  echo "缺少前端依赖，请先执行：npm --prefix frontend install"
  exit 1
fi

cd "${PROJECT_DIR}"
"${PYTHON_BIN}" -m uvicorn src.video_workflow.server.app:app --host 127.0.0.1 --port 8001 &
BACKEND_PID=$!

cleanup() {
  kill "${BACKEND_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "VideoWorkflow 后端：http://localhost:8001"
echo "VideoWorkflow 工作台：http://localhost:3002"
npm --prefix frontend run dev
