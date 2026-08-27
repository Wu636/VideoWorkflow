#!/usr/bin/env bash
set -euo pipefail

# 刷新 Seedance 素材公网隧道（Cloudflare Quick Tunnel）并回写运行配置。
#
# Quick Tunnel 域名是会话级的：隧道连接断开/过期后域名不再解析，
# Seedance 提交前的素材可读性检查会批量失败（[Errno -2] Name or service not known）。
# 重启隧道容器会获得新域名；本脚本解析新域名并通过后端设置 API 热回写，
# 无需手动复制粘贴，也无需重启后端。

CONTAINER="${SEEDANCE_TUNNEL_CONTAINER:-videoworkflow-seedance-tunnel}"
BACKEND_URL="${BACKEND_URL:-http://127.0.0.1:8001}"
TUNNEL_TARGET="${SEEDANCE_TUNNEL_URL:-http://host.docker.internal:8001}"

command -v docker >/dev/null 2>&1 || { echo "缺少 docker，无法管理隧道容器"; exit 1; }

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER}"; then
  echo "重启隧道容器 ${CONTAINER} …"
  docker restart "${CONTAINER}" >/dev/null
else
  echo "创建隧道容器 ${CONTAINER} → ${TUNNEL_TARGET} …"
  docker run -d --name "${CONTAINER}" cloudflare/cloudflared:latest \
    tunnel --no-autoupdate --url "${TUNNEL_TARGET}" >/dev/null
fi

# 等待新 Quick Tunnel 域名出现在日志中（最多 45 秒）
URL=""
for _ in $(seq 1 45); do
  URL="$(docker logs --since 90s "${CONTAINER}" 2>&1 | grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' | tail -n 1 || true)"
  if [[ -n "${URL}" ]]; then break; fi
  sleep 1
done
if [[ -z "${URL}" ]]; then
  echo "失败：未能从隧道日志解析出新的 trycloudflare 域名，请查看 docker logs ${CONTAINER}"
  exit 1
fi
echo "新素材公网地址：${URL}"

# 验证隧道确实路由到后端（签名探针预期返回 403/404，即请求已到达后端）
STATUS="$(curl -sk -o /dev/null -w '%{http_code}' -m 20 "${URL}/api/projects/seedance-assets/__probe__?expires=0&signature=probe" || true)"
case "${STATUS}" in
  403|404|422) echo "隧道可达性验证通过（HTTP ${STATUS}）" ;;
  *) echo "注意：隧道地址返回 HTTP ${STATUS:-无响应}，可能尚未就绪，稍后请在设置页复核" ;;
esac

# 回写运行配置（后端即时热加载，无需重启）
if curl -sf -X PUT "${BACKEND_URL}/api/settings" \
  -H 'Content-Type: application/json' \
  -d "{\"values\": {\"SEEDANCE_PUBLIC_ASSET_BASE_URL\": \"${URL}\"}}" >/dev/null; then
  echo "已回写 SEEDANCE_PUBLIC_ASSET_BASE_URL，可直接重试失败的渲染任务。"
else
  echo "失败：无法回写配置（后端 ${BACKEND_URL} 不可达），请在设置页手动填写 ${URL}"
  exit 1
fi
