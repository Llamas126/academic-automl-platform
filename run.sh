#!/usr/bin/env bash
# One-command setup for the academic-automl-platform Docker stack.
#
#   ./run.sh              build + start full stack, wait until healthy
#   ./run.sh --test       run the full test suite in an ephemeral container first
#   ./run.sh --skip-build start using existing images
#   ./run.sh --down       stop the stack
set -euo pipefail
cd "$(dirname "$0")"

TEST=false
DOWN=false
SKIP_BUILD=false

for arg in "$@"; do
  case "$arg" in
    --test) TEST=true ;;
    --down) DOWN=true ;;
    --skip-build) SKIP_BUILD=true ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "ERROR: unknown flag: $arg" >&2
      exit 1
      ;;
  esac
done

if $DOWN; then
  echo "==> Stopping stack"
  docker compose down
  exit 0
fi

echo "==> Checking environment file"
if [ ! -f .env ]; then
  if command -v openssl >/dev/null 2>&1; then
    PASS="$(openssl rand -hex 12)"
  else
    PASS="$(head -c 12 /dev/urandom | od -An -tx1 | tr -d ' \n')"
  fi
  sed "s/^REDIS_PASSWORD=.*/REDIS_PASSWORD=${PASS}/" .env.example > .env
  chmod 600 .env
  echo "    Created .env from .env.example with a generated REDIS_PASSWORD."
else
  echo "    Existing .env left untouched."
fi

echo "==> Validating Docker engine"
if ! docker info >/dev/null 2>&1; then
  echo "ERROR: Docker daemon is not reachable." >&2
  echo "       Start Docker (Desktop) and retry. Install guide: https://docs.docker.com/get-docker/" >&2
  exit 2
fi
if ! docker compose version >/dev/null 2>&1; then
  echo "ERROR: Docker Compose v2 plugin not found." >&2
  exit 2
fi
echo "    Docker engine and Compose plugin OK."

mkdir -p data

if $TEST; then
  echo "==> Building test image"
  if ! docker compose --profile test build tests; then
    echo "ERROR: test image build failed." >&2
    exit 3
  fi
  echo "==> Running full test suite inside container"
  if ! docker compose --profile test run --rm tests; then
    echo "ERROR: containerized test suite failed; aborting deployment." >&2
    exit 3
  fi
fi

echo "==> Building and starting stack (gateway, ingestion, mlcore, inference, workers, redis)"
UP_ARGS=(up -d)
if ! $SKIP_BUILD; then
  UP_ARGS+=(--build)
fi
if ! docker compose "${UP_ARGS[@]}"; then
  echo "ERROR: docker compose up failed." >&2
  exit 2
fi

echo "==> Waiting for gateway health at http://localhost:8000/health"
DEADLINE=$((SECONDS + 120))
HEALTHY=false
while [ "$SECONDS" -lt "$DEADLINE" ]; do
  if curl -fsS -m 3 http://localhost:8000/health >/dev/null 2>&1; then
    HEALTHY=true
    break
  fi
  printf '.'
  sleep 2
done
echo ""
if ! $HEALTHY; then
  echo "ERROR: gateway did not become healthy within 120 seconds. Diagnostics:" >&2
  docker compose ps
  docker compose logs --tail 50 gateway
  exit 4
fi
echo "    Gateway is healthy."

docker compose ps
cat <<'EOF'

Stack ready:
  Gateway : http://localhost:8000
  Docs    : http://localhost:8000/docs
  Stop    : ./run.sh --down   (or: docker compose down)

Auth and rate limiting are disabled by default; see .env to enable them.
EOF
exit 0
