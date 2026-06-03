# syntax=docker/dockerfile:1.7

ARG PYTHON_VERSION=3.12
FROM ghcr.io/astral-sh/uv:python${PYTHON_VERSION}-bookworm AS uv-bin

FROM ubuntu:24.04 AS dev

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      bash \
      ca-certificates \
      python3.12 \
      python3.12-venv \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv-bin /usr/local/bin/uv /usr/local/bin/uvx /usr/local/bin/

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /workspace

# Keep dependency installation cached unless pyproject.toml or uv.lock changes.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --extra dev --no-install-project
RUN uv run --no-sync python -c 'from codex_cli_bin import bundled_codex_path; import subprocess; subprocess.run([str(bundled_codex_path()), "--version"], check=True)'

COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --extra dev

CMD ["uv", "run", "--no-sync", "pytest"]
