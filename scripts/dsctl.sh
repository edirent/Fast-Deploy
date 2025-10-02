#!/usr/bin/env bash
set -euo pipefail

NS="ds-v3"
CHART="./charts/ds-v3"
RELEASE="ds-v3"
VALUES=${VALUES:-"./charts/ds-v3/values.yaml"}

cmd=${1:-help}

case "$cmd" in
  init)
    kubectl create ns $NS || true
    helm repo add nvidia https://nvidia.github.io/gpu-operator || true
    helm upgrade --install gpu-operator nvidia/gpu-operator -n nvidia --create-namespace \
      --set driver.enabled=true --set toolkit.enabled=true --set dcgmExporter.enabled=true
    echo "[ok] cluster inited"
    ;;
  s3secret)
    kubectl -n $NS delete secret s3-cred 2>/dev/null || true
    kubectl -n $NS create secret generic s3-cred \
      --from-literal=S3_ENDPOINT="${S3_ENDPOINT}" \
      --from-literal=AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}" \
      --from-literal=AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}"
    ;;
  build)
    ./scripts/build_image.sh
    ;;
  deploy)
    helm upgrade --install $RELEASE $CHART -n $NS -f $VALUES --create-namespace
    ;;
  canary)
    echo "roll canary 10%… (implement with Istio/NGINX canary annotations)"
    ;;
  scale)
    REPLICAS=${2:-3}
    kubectl -n $NS scale deploy/$RELEASE --replicas=$REPLICAS
    ;;
  status)
    kubectl -n $NS get pods -o wide
    kubectl -n $NS top pods || true
    ;;
  *)
    echo "usage: $0 {init|s3secret|build|deploy|canary|scale|status}"
    ;;
esac
#!/usr/bin/env bash
set -euo pipefail

NS="ds-v3"
CHART="./charts/ds-v3"
RELEASE="ds-v3"
VALUES=${VALUES:-"./charts/ds-v3/values.yaml"}

cmd=${1:-help}

case "$cmd" in
  init)
    kubectl create ns $NS || true
    helm repo add nvidia https://nvidia.github.io/gpu-operator || true
    helm upgrade --install gpu-operator nvidia/gpu-operator -n nvidia --create-namespace \
      --set driver.enabled=true --set toolkit.enabled=true --set dcgmExporter.enabled=true
    echo "[ok] cluster inited"
    ;;
  s3secret)
    kubectl -n $NS delete secret s3-cred 2>/dev/null || true
    kubectl -n $NS create secret generic s3-cred \
      --from-literal=S3_ENDPOINT="${S3_ENDPOINT}" \
      --from-literal=AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}" \
      --from-literal=AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}"
    ;;
  build)
    ./scripts/build_image.sh
    ;;
  deploy)
    helm upgrade --install $RELEASE $CHART -n $NS -f $VALUES --create-namespace
    ;;
  canary)
    echo "roll canary 10%… (implement with Istio/NGINX canary annotations)"
    ;;
  scale)
    REPLICAS=${2:-3}
    kubectl -n $NS scale deploy/$RELEASE --replicas=$REPLICAS
    ;;
  status)
    kubectl -n $NS get pods -o wide
    kubectl -n $NS top pods || true
    ;;
  *)
    echo "usage: $0 {init|s3secret|build|deploy|canary|scale|status}"
    ;;
esac
