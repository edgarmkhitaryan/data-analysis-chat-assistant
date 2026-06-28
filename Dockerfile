# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Data Analysis Chat Assistant — runtime image for the CLI.
# Build:  docker build -t assistant .
# Run:    docker run --rm -it \
#           -v "$(pwd)/.env:/app/.env:ro" \
#           -v "$HOME/.config/gcloud/application_default_credentials.json:/gcp/adc.json:ro" \
#           -e GOOGLE_APPLICATION_CREDENTIALS=/gcp/adc.json \
#           assistant
# .env is mounted (not --env-file) so the app's own loader parses it — this handles
# the inline comments in .env.example, which `docker --env-file` does not.
# See README "Run with Docker" for the why behind each flag.
# ---------------------------------------------------------------------------
FROM python:3.12-slim

# Predictable, quiet, no .pyc clutter; one place for cache-free pip.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# 1) Dependencies first, so this layer is cached unless the requirements change.
#    Both files are needed at build time: pyproject reads them for (dev) deps.
COPY requirements.txt requirements-dev.txt ./
RUN pip install -r requirements.txt

# 2) Project metadata + source + seed data, then install the package itself
#    (provides the `assistant` console entrypoint). --no-deps: deps are already in.
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY data/ ./data/
RUN pip install --no-deps .

# 3) Run as a non-root user and pre-create the writable dirs the app uses
#    (SQLite app.db + golden index, structured logs, per-turn traces).
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p data/golden_index logs traces \
    && chown -R appuser:appuser /app
USER appuser

# The CLI is the container's command; pass --user / --json as `docker run` args,
# e.g. `docker run ... assistant --user manager_b`.
ENTRYPOINT ["assistant"]
