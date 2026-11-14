# Brain image: gesture FSM + action mapping + WebSocket server.
# Perception (GPU hand detection) lives in the edge agent on the host;
# the brain consumes lightweight LandmarksMsg and stays dependency-light.

FROM python:3.12-slim

# Default: the official Debian CDN (Fastly) — fastest with a VPN on.
# On a direct Russian link override with:
#   --build-arg DEBIAN_MIRROR=https://mirror.yandex.ru/debian
ARG DEBIAN_MIRROR=http://deb.debian.org/debian

RUN set -eux; \
    sed -i "s|http://deb.debian.org/debian|$DEBIAN_MIRROR|g" /etc/apt/sources.list.d/debian.sources; \
    apt-get update; \
    apt-get install -y --no-install-recommends curl; \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV UV_LINK_MODE=copy
# PyPI via Fastly is fine over VPN; override with --build-arg if ever needed.
ARG UV_INDEX_URL=https://pypi.org/simple

WORKDIR /app

COPY src/protocol /app/src/protocol
COPY src/brain /app/src/brain
COPY configs /app/configs

RUN --mount=type=cache,target=/root/.cache/uv \
    env UV_INDEX_URL="$UV_INDEX_URL" uv pip install --system -e /app/src/protocol -e /app/src/brain \
    && python -c "from neuroint_protocol.gestures import Pose; assert 'pinch' in [p.value for p in Pose], 'stale protocol build'"

RUN useradd --create-home appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8765

CMD ["python", "-m", "neuroint_brain.main"]
