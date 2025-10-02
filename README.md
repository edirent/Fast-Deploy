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

6. Capacity

## Capacity estimator

This repository includes a small utility to estimate GPU and container RAM requirements for causal LMs:

`scripts/capacity_estimator.py`

Example usage (estimates VRAM per GPU and suggested tensor-parallel count):

```bash
python3 scripts/capacity_estimator.py \
	--params 40000000000 \
	--dtype bf16 \
	--kv-bits 16 \
	--batch 8 \
	--ctx 8192 \
	--gen 512 \
	--gpu-mem 80 \
	--max-gpus 8
```

Flags of interest:
- `--params` : total number of model parameters (required if no HF config supplied)
- `--dtype` : weight datatype (bf16, fp16, int8, int4, fp8)
- `--kv-bits` : KV-cache precision in bits (16/8/4)
- `--batch`/`--ctx`/`--gen` : concurrent batch size, prompt length and generation length
- `--gpu-mem`/`--max-gpus` : per-GPU VRAM (GiB) and maximum GPUs to try for tensor-parallel
- `--empirical` : run an empirical probe (requires `vllm`, `torch` and `pynvml`) to measure throughput and peak VRAM

See `scripts/capacity_estimator.py` for more details and additional flags.