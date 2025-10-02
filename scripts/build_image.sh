#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-"registry.local/ds-v3:latest"}
DOCKERFILE=${DOCKERFILE:-"Dockerfile"}

echo "Building image $IMAGE"
docker build -t "$IMAGE" -f "$DOCKERFILE" .
echo "Pushing $IMAGE"
docker push "$IMAGE"
#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-"registry.local/ds-v3:latest"}
DOCKERFILE=${DOCKERFILE:-"Dockerfile"}

echo "Building image $IMAGE"
docker build -t "$IMAGE" -f "$DOCKERFILE" .
echo "Pushing $IMAGE"
docker push "$IMAGE"
