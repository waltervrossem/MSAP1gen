# -*- coding: utf-8 -*-

ARG DEBIAN_NAME="bookworm"
ARG PYTHON_VERSION="3.12"
FROM python:${PYTHON_VERSION}-slim-${DEBIAN_NAME} AS base

WORKDIR /app
RUN apt-get update -qq \
      && apt upgrade --yes \
      && apt-get clean all \
      && rm -rf /var/lib/apt/lists/*

FROM base AS pre_builder

WORKDIR /app

RUN apt-get update -qq \
      && apt upgrade --yes \
      && apt-get -y --no-install-recommends install \
         build-essential \
         ca-certificates \
         gdb \
         gfortran \
         wget \
         file \
         apt-utils \
         git  \
      && apt-get clean all \
      && rm -r /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
RUN mkdir -p src/MSAP1gen src/psls

RUN pip install --upgrade pip
RUN pip install --only-deps .

FROM pre_builder AS builder

WORKDIR /app

COPY . .
RUN pip install --no-dependencies .

FROM python:3.12-slim-bookworm AS release
ARG BUILD_DATE
ARG VERSION
ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /app


# Labels (based on OCI Image Format Specification)
LABEL \
    org.opencontainers.image.version=$VERSION \
    org.opencontainers.image.created=$BUILD_DATE \
    org.opencontainers.image.title="MSAP1gen" \
    org.opencontainers.image.ref.name="msap1gen" \
    org.opencontainers.image.description="MSAP1gen Synthetic lightcurve generation" \
    org.opencontainers.image.licenses="" \
    org.opencontainers.image.source="https://github.com/waltervrossem/MSAP1gen" \
    org.opencontainers.image.documentation="" \
    org.opencontainers.image.vendor="" \
    org.opencontainers.image.authors=""

# Get the installed libraries and application from the ealier stage
COPY --from=builder /usr/local/ /usr/local/

# Regenerate the shared-library cache
RUN ldconfig

ENTRYPOINT ["MSAP1gen"]
