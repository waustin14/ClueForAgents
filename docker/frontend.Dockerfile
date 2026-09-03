# Static test console + nginx reverse proxy to the game-logic API.
# Built from repo root: docker build -f docker/frontend.Dockerfile .
FROM nginx:alpine

COPY docker/nginx.conf /etc/nginx/conf.d/default.conf
COPY frontend/ /usr/share/nginx/html/

# 127.0.0.1, not localhost: busybox wget resolves "localhost" to ::1 first
# and (unlike Python's urlopen) doesn't fall back to IPv4 on refusal, and
# nginx's `listen 80;` here is IPv4-only.
HEALTHCHECK --interval=5s --timeout=3s --start-period=5s --retries=5 \
    CMD wget -q -O /dev/null http://127.0.0.1/ || exit 1
