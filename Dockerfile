FROM python:3.12-slim-bookworm AS builder

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_NO_CACHE=1
WORKDIR /app
RUN python -m pip install --no-cache-dir uv==0.10.9
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --locked --no-dev --no-editable --python /usr/local/bin/python

FROM python:3.12-slim-bookworm

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp
RUN apt-get update \
    && apt-get install -y --no-install-recommends libexpat1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY config ./config
COPY artifacts/v1/manifest.json artifacts/v1/*.npy ./artifacts/v1/
RUN mkdir -p outputs \
    && chmod -R a+rX /app \
    && chown -R 10001:10001 artifacts outputs
USER 10001:10001
CMD ["python", "-m", "unet_nowcast.container"]
