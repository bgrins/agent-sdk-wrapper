# syntax=docker/dockerfile:1.7
FROM node:22.23.2-bookworm-slim AS node-bin

FROM node-bin AS sample
WORKDIR /sample-app
COPY examples/gvisor/workload/project/package*.json examples/gvisor/workload/project/.npmrc ./
RUN npm ci --ignore-scripts --no-audit --no-fund
COPY examples/gvisor/workload/project/ ./

FROM ubuntu:24.04 AS base
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git libstdc++6 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir /job && chown 1000:1000 /job
# Node/npm run the target project; provider runtimes come from the SDK packages.
COPY --from=node-bin /usr/local/bin/node /usr/local/bin/node
COPY --from=node-bin /usr/local/lib/node_modules/npm /usr/local/lib/node_modules/npm
RUN ln -s ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -s ../lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx
COPY --from=sample /sample-app /sample-app
WORKDIR /example
COPY examples/gvisor/workload/shared/ ./shared/

FROM base AS checks
COPY examples/gvisor/tests/security.test.mjs examples/gvisor/tests/fixture-worker.mjs ./tests/
ENTRYPOINT ["node"]
CMD ["tests/fixture-worker.mjs"]

FROM caddy:2.11.4 AS caddy-bin
FROM base AS gateway
COPY --from=caddy-bin /usr/bin/caddy /usr/local/bin/caddy
# Clear file capabilities: only a Unix socket is needed, and Compose drops all caps.
RUN chown 0:0 /usr/local/bin/caddy
COPY examples/gvisor/infra/gateway/Caddyfile ./gateway/Caddyfile
ENTRYPOINT ["timeout", "--kill-after=3s", "360s", "caddy", "run", "--config", "/example/gateway/Caddyfile", "--adapter", "caddyfile"]

FROM gateway AS gateway-tests
COPY examples/gvisor/tests/gateway.test.mjs examples/gvisor/tests/Caddyfile ./tests/
ENTRYPOINT ["timeout", "--kill-after=3s", "360s", "caddy", "run", "--config", "/example/tests/Caddyfile", "--adapter", "caddyfile"]

FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS uv-bin
FROM base AS python-base
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3.12 python3.12-venv \
    && rm -rf /var/lib/apt/lists/*
ENV PATH="/opt/venv/bin:${PATH}" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

FROM python-base AS python-install
COPY --from=uv-bin /usr/local/bin/uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv
COPY packages/python/ /build/packages/python/
COPY examples/gvisor/workload/agent-python/pyproject.toml examples/gvisor/workload/agent-python/uv.lock /build/examples/gvisor/workload/agent-python/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --directory /build/examples/gvisor/workload/agent-python --locked --no-dev --no-editable

FROM python-base AS python
COPY --from=python-install /opt/venv /opt/venv
COPY examples/gvisor/workload/agent-python/main.py ./main.py
ENTRYPOINT ["python", "/example/main.py"]

FROM node-bin AS typescript-install
WORKDIR /build
COPY package.json package-lock.json .npmrc ./
COPY packages/typescript/package.json packages/typescript/package.json
RUN --mount=type=cache,target=/root/.npm npm ci --ignore-scripts --no-audit --no-fund
COPY packages/typescript/ packages/typescript/
RUN npm run build
COPY examples/gvisor/workload/agent-typescript/package*.json examples/gvisor/workload/agent-typescript/.npmrc examples/gvisor/workload/agent-typescript/
# Install the local package as a consumer, including its built files (not a symlink).
RUN --mount=type=cache,target=/root/.npm \
    npm ci --prefix examples/gvisor/workload/agent-typescript --install-links --ignore-scripts --omit=dev --no-audit --no-fund

FROM base AS typescript
COPY --from=typescript-install /build/examples/gvisor/workload/agent-typescript/node_modules ./node_modules
COPY examples/gvisor/workload/agent-typescript/main.mjs ./main.mjs
ENTRYPOINT ["node", "/example/main.mjs"]
