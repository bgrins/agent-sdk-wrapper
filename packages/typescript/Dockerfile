# syntax=docker/dockerfile:1.7

# Latest Node 22 release in the official index on 2026-09-10 (2026-07-28).
ARG NODE_VERSION=22.23.2
FROM node:${NODE_VERSION}-bookworm-slim AS node-bin

FROM ubuntu:24.04 AS dev
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Keep the final image Ubuntu-based; copy only Node/npm from the official image.
COPY --from=node-bin /usr/local/bin/node /usr/local/bin/node
COPY --from=node-bin /usr/local/lib/node_modules/npm /usr/local/lib/node_modules/npm
RUN ln -s ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -s ../lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx

WORKDIR /workspace
COPY package.json package-lock.json .npmrc ./
COPY packages/typescript/package.json packages/typescript/package.json
RUN --mount=type=cache,target=/root/.npm npm ci --ignore-scripts --no-audit --no-fund

COPY packages/typescript/ packages/typescript/
COPY docs/fixtures/ docs/fixtures/
COPY docs/schemas/ docs/schemas/

# Provider runtimes come from the two SDK dependencies. No standalone CLI install.
CMD ["npm", "run", "verify"]
