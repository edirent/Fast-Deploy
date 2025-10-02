# ds-v3-on-k8s (root)

This repository contains a minimal Helm chart and helper scripts to deploy a large-model inference service on GPU-enabled Kubernetes clusters.

Quickstart

1. Edit `charts/ds-v3/values.yaml` to set image, model and S3 URIs.
2. Provide S3 credentials:

```bash
export S3_ENDPOINT=https://minio.local:9000
export AWS_ACCESS_KEY_ID=access
export AWS_SECRET_ACCESS_KEY=secret
./scripts/dsctl.sh s3secret
```

3. Initialize cluster components (GPU operator):

```bash
./scripts/dsctl.sh init
```

4. Deploy:

```bash
./scripts/dsctl.sh deploy
```

5. Smoke test (adjust host if using port/ingress):

```bash
./scripts/smoke_test.sh http://ds-v3.api.corp.local
```

Customization notes

- Adjust `charts/ds-v3/templates/deployment.yaml` for hostPath/PVC caching or NVMe prefetch.
- Implement `scripts/canary_route.sh` for your ingress (Istio or NGINX).
- Add CI to build/push images and run smoke tests before canary promotion.
# ds-v3-on-k8s

Minimal Helm chart + scripts to deploy a large-model inference service (vLLM / TensorRT / DeepSpeed) on GPU Kubernetes clusters.

Quickstart

1. Edit `charts/ds-v3/values.yaml` to set image, model and S3 URIs.
2. Provide S3 credentials:

```bash
export S3_ENDPOINT=https://minio.local:9000
export AWS_ACCESS_KEY_ID=access
export AWS_SECRET_ACCESS_KEY=secret
./scripts/dsctl.sh s3secret
```

3. Initialize cluster components (GPU operator):

```bash
./scripts/dsctl.sh init
```

4. Deploy:

```bash
./scripts/dsctl.sh deploy
```

5. Smoke test (adjust host if using port/ingress):

```bash
./scripts/smoke_test.sh http://ds-v3.api.corp.local
```

Customization

- Add RDMA/IB module setup and kernel tuning as part of your cluster bootstrap.
- If you use Istio, implement `scripts/canary_route.sh` to adjust VirtualService weights.
