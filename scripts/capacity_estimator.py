#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Capacity estimator for ds v3 (Causal LM).
Static estimate + optional empirical probe (vLLM/torch/pynvml).
Formulas moved to README to avoid special unicode issues.
"""
import argparse, json, math, os, time

def mib(x): return x / (1024**2)
def gib(x): return x / (1024**3)

def load_hf_config(path):
    if path and os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

def bytes_per_dtype(dtype: str) -> float:
    d = dtype.lower()
    if d in ["fp16","half","float16","bf16","bfloat16"]: return 2.0
    if d in ["fp8","e4m3","e5m2"]: return 1.0
    if d in ["int8","i8"]: return 1.0
    if d in ["int4","i4","nf4","fp4"]: return 0.5
    return 2.0  # default bf16

def infer_params_from_config(cfg: dict) -> tuple:
    # return (n_layer, n_heads, n_kv_heads, d_model, vocab_size, intermediate_size, params_est)
    n_layer = cfg.get("num_hidden_layers") or cfg.get("n_layer")
    n_heads = cfg.get("num_attention_heads") or cfg.get("n_head")
    n_kv = cfg.get("num_key_value_heads") or cfg.get("n_kv_heads") or n_heads
    d_model = cfg.get("hidden_size") or cfg.get("d_model")
    vocab = cfg.get("vocab_size") or 0
    inter  = cfg.get("intermediate_size") or int(4 * (d_model or 0))
    params_est = 0
    if all([n_layer, n_heads, d_model, inter]):
        # rough estimate (ignore bias/ln): ~12 * d_model^2 per layer + embedding
        params_est = int(12 * (d_model**2) * n_layer + vocab * d_model)
    return n_layer, n_heads, n_kv, d_model, vocab, inter, params_est

def estimate_vram_gib(params_cnt:int, weight_bytes:float, tp:int,
                      batch:int, ctx:int, gen:int,
                      n_layer:int, n_heads:int, n_kv_heads:int, d_model:int,
                      kv_bytes:float, kv_alpha:float, overhead_gb:float) -> float:
    if tp <= 0: tp = 1
    # weights
    w_total = params_cnt * weight_bytes
    w_per_gpu = w_total / tp
    # KV cache: 2 * d_model * (n_kv_heads/n_heads) * kv_bytes per token per layer
    ratio = (n_kv_heads / max(1.0, float(n_heads)))
    kv_per_token_per_layer = 2.0 * d_model * ratio * kv_bytes
    kv_total = batch * (ctx + gen) * n_layer * kv_per_token_per_layer
    kv_per_gpu = (kv_total / tp) * kv_alpha
    vram_bytes = w_per_gpu + kv_per_gpu + overhead_gb * (1024**3)
    return gib(vram_bytes)

def recommend_gpu_count(available_gib_per_gpu:float, max_gpus:int, **kwargs):
    for tp in range(1, max_gpus+1):
        need = estimate_vram_gib(tp=tp, **kwargs)
        if need <= available_gib_per_gpu:
            return tp, need
    return None, estimate_vram_gib(tp=max_gpus, **kwargs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-config", type=str, default="", help="HF config.json path (optional)")
    ap.add_argument("--params", type=int, default=0, help="#params. If 0, infer from config")
    ap.add_argument("--dtype", type=str, default="bf16", help="bf16/fp16/int8/int4/fp8")
    ap.add_argument("--kv-bits", type=int, default=16, help="KV-Cache bits: 16/8/4")
    ap.add_argument("--kv-alpha", type=float, default=1.0, help="effective KV scale (e.g., 0.5 for int8 KV in vLLM)")
    ap.add_argument("--batch", type=int, default=4, help="concurrent batch")
    ap.add_argument("--ctx", type=int, default=8192, help="prompt tokens")
    ap.add_argument("--gen", type=int, default=512, help="generation tokens")
    ap.add_argument("--gpu-mem", type=float, default=80.0, help="per-GPU VRAM GiB")
    ap.add_argument("--max-gpus", type=int, default=8, help="max TP to try")
    ap.add_argument("--overhead-gb", type=float, default=4.0, help="runtime overhead GiB")
    ap.add_argument("--target-tps", type=float, default=0.0, help="target tokens/s (optional)")
    ap.add_argument("--empirical", action="store_true", help="run empirical probe (needs vllm/torch/pynvml)")
    args = ap.parse_args()

    cfg = load_hf_config(args.model_config)
    n_layer, n_heads, n_kv, d_model, vocab, inter, params_est = infer_params_from_config(cfg)
    params_cnt = args.params or params_est
    if not params_cnt:
        raise SystemExit("缺少 --params 或有效的 config.json")

    w_bytes = bytes_per_dtype(args.dtype)
    kv_bytes = 2.0 if args.kv_bits >= 16 else (1.0 if args.kv_bits == 8 else 0.5)
    kv_alpha = float(args.kv_alpha)

    base_kwargs = dict(
        params_cnt=params_cnt,
        weight_bytes=w_bytes,
        batch=args.batch, ctx=args.ctx, gen=args.gen,
        n_layer=n_layer or 60, n_heads=n_heads or 64, n_kv_heads=n_kv or (n_heads or 64),
        d_model=d_model or 8192,
        kv_bytes=kv_bytes, kv_alpha=kv_alpha,
        overhead_gb=args.overhead_gb
    )

    tp_suggest, vram_need_gib = recommend_gpu_count(
        available_gib_per_gpu=args.gpu_mem,
        max_gpus=args.max_gpus,
        **base_kwargs
    )

    # container RAM suggestion: baseline 10GiB + 0.1 * weight_size + 20% headroom
    w_total_gib = gib(params_cnt * w_bytes)
    ram_gib = (10.0 + 0.10 * w_total_gib) * 1.2
    ram_gib = max(12.0, ram_gib)

    result = {
        "estimator": "static",
        "model_params_billion": round(params_cnt/1e9, 2),
        "dtype_weight_bytes": w_bytes,
        "kv_bits": args.kv_bits,
        "kv_alpha": kv_alpha,
        "d_model": d_model,
        "n_layer": n_layer,
        "n_heads": n_heads,
        "n_kv_heads": n_kv,
        "batch": args.batch,
        "ctx_tokens": args.ctx,
        "gen_tokens": args.gen,
        "per_gpu_mem_gib": args.gpu_mem,
        "overhead_gb": args.overhead_gb,
        "tp_suggest": tp_suggest if tp_suggest else f">{args.max_gpus}",
        "vram_need_gib_per_gpu_at_tp": round(vram_need_gib, 2),
        "container_ram_gib_suggest": round(ram_gib, 1),
    }

    if args.empirical:
        try:
            import pynvml, torch
            from vllm import LLM, SamplingParams
            pynvml.nvmlInit()
            gpu_cnt = torch.cuda.device_count()
            tp = tp_suggest or min(args.max_gpus, gpu_cnt)
            tp = max(1, min(tp, gpu_cnt))
            prompt = "A" * args.ctx
            prompts = [prompt] * args.batch
            llm = LLM(model=args.model_config or ".", tensor_parallel_size=tp, dtype=args.dtype)
            sp = SamplingParams(max_tokens=args.gen, temperature=0.0)
            t0 = time.time()
            _ = llm.generate(prompts, sp)
            t1 = time.time()
            elapsed = t1 - t0
            actual_tokens = args.batch * args.gen
            tps = actual_tokens / max(1e-6, elapsed)
            used_gib = 0.0
            for i in range(tp):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                used_gib = max(used_gib, mem.used / (1024**3))
            result.update({
                "estimator": "empirical",
                "tp_used": tp,
                "measured_tokens_per_sec": round(tps, 1),
                "measured_peak_vram_gib_per_gpu": round(used_gib, 2),
            })
            if args.target_tps and tps > 0:
                need_gpu = math.ceil(args.target_tps / tps * tp * 1.1)  # +10% headroom
                result["gpu_count_for_target_tps"] = int(need_gpu)
        except Exception as e:
            result["empirical_error"] = str(e)

    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
