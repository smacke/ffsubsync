FROM python:3.14-slim AS base

LABEL name="subsync"
LABEL maintainer="Peter Dave Hello"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

FROM base AS builder

ARG PIP_INSTALL_TARGET=.
ARG FFSUBSYNC_VERSION=

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential git python3-dev \
 && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv "${VIRTUAL_ENV}"
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

COPY . .

RUN install_target="${FFSUBSYNC_VERSION:+ffsubsync==${FFSUBSYNC_VERSION}}" \
 && pip install --no-cache-dir "${install_target:-${PIP_INSTALL_TARGET}}"

RUN subsync --version

FROM base AS runtime

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

COPY --from=builder "${VIRTUAL_ENV}" "${VIRTUAL_ENV}"

WORKDIR /video

ENTRYPOINT [ "subsync" ]
