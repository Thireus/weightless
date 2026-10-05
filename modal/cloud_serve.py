"""GLM-5.3 flagship serving on Modal B200:8; run through the Modal CLI.

Reuses refusal-research/experiments/20260829-glm53-flagship's volume,
vLLM image, marlin backend and calibrated GLP-77 directions. Serving runs
CUDA graphs (eager is the capture/eval-lane discipline, never serving).
Set WEIGHTLESS_STEER_ALPHA before deployment to override the 1.0 default.
ensure_dirs restores the validated GLP-77 .pt from
msuiche/GLM-5.3-abliterated-cyber-GLP-77 (pinned sha256) when the volume
has lost it; it never derives directions.

Shape: B200:8 with the full 1M context. vLLM replicates MLA latent KV on
every TP rank (~90 GB/rank at 1M tokens, bf16 — Hopper sparse-MLA rejects
fp8 KV, hence fix_kv_scheme), so 1M does not fit on H100 (~210K) or H200
(~880K); B200's 183 GiB leaves ~113 GB free per rank after the 54.7 GiB
weights. B200 also gives native NVFP4 compute (H100 marlin is weight-only
emulation). GLM53XL_GPU=H100:8 with MAX_MODEL_LEN=131072 reproduces the
original experiment shape.
"""
import json
import os
from pathlib import Path
import subprocess

import modal

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "RadixArk/GLM-5.3-NVFP4"
SERVED_MODEL = "glm-5.3"
VOLUME_NAME = "glm53-nvfp4"
DIRS_PATH = "/data/out/glm53-32perlayer-dirs.pt"
VECTOR_REPO = "msuiche/GLM-5.3-abliterated-cyber-GLP-77"
VECTOR_FILE = "glm53-32perlayer-dirs.pt"
# sha256 of the experiment's validated .pt (out/glm53-32perlayer-dirs.pt),
# republished byte-identical in the vector repo.
VECTOR_SHA256 = "a4907fb1d124e7eda4b7abbd47a40251579f788b67893b8bb713dd89aa194439"
PATCH_NAME = "hotfix-glm53-modal-residual.py"
EXPERIMENT = "refusal-research/experiments/20260829-glm53-flagship"
ENV = {"HF_HOME": "/data/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"}

# Shape switches, read at deploy time (same convention as cloud_serve_glm53).
GPU = os.environ.get("GLM53XL_GPU", "B200:8")
MAX_MODEL_LEN = os.environ.get("MAX_MODEL_LEN", "1048576")
# 800K on H200:8 needs ~0.965 (vLLM 2026-10-04: 69.01 GiB KV/rank needed at
# 800K vs 63.17 available at 0.92 after cudagraph profiling). B200 fits 1M
# at 0.92 with ~20 GiB headroom.
GMU = os.environ.get("GLM53XL_GMU", "0.92")

app = modal.App("weightless-cloud")
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
download_image = (modal.Image.debian_slim(python_version="3.12")
                  .pip_install("huggingface_hub", "hf_transfer"))
image = (modal.Image.from_registry("vllm/vllm-openai:v0.28.0", add_python="3.12")
         .entrypoint([])
         .add_local_file(ROOT / "patches" / PATCH_NAME, "/work/" + PATCH_NAME, copy=True))


def fix_kv_scheme(snapshot):
    """Hopper sparse MLA requires bf16 KV; the checkpoint declares fp8.

    Replace the config symlink, preserving the original HF cache blob.
    This is the experiment's fix_kv_scheme, applied to the resolved snapshot.
    """
    path = Path(snapshot) / "config.json"
    config = json.loads(path.read_text())
    quant = config.get("quantization_config") or {}
    if "kv_cache_scheme" in quant:
        quant.pop("kv_cache_scheme")
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(config, indent=2) + "\n")
        temporary.replace(path)


@app.function(image=download_image, volumes={"/data": vol}, timeout=6 * 3600,
              secrets=[modal.Secret.from_name("hf-token")], env=ENV)
def ensure_weights():
    """Resume/cache the ~465 GB snapshot on CPU; no GPU allocation.

    The weights repo is public: if the hf-token secret has gone stale, HF
    rejects the request with a 401 instead of falling back to anonymous —
    retry token-free so an expired secret can never block public weights
    (the gated vector repo in ensure_dirs still needs a valid token)."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import RepositoryNotFoundError

    try:
        snapshot = snapshot_download(MODEL_ID)
    except RepositoryNotFoundError:
        print("hf-token rejected (401) — retrying anonymously", flush=True)
        snapshot = snapshot_download(MODEL_ID, token=False)
    fix_kv_scheme(snapshot)
    vol.commit()
    print("weights ready:", snapshot)


def require_dirs():
    if not Path(DIRS_PATH).is_file():
        raise FileNotFoundError(
            f"Missing {DIRS_PATH} on volume {VOLUME_NAME}. This serving lane "
            "does not derive directions — run `modal run "
            "modal/cloud_serve.py::ensure_dirs` to restore the validated .pt "
            f"from {VECTOR_REPO}.")


@app.function(image=download_image, volumes={"/data": vol}, timeout=900,
              secrets=[modal.Secret.from_name("hf-token")], env=ENV)
def ensure_dirs():
    """Verify the GLP-77 .pt on the volume, restoring it from the published
    HF artifact (pinned sha256) when the volume has lost it. Restoring the
    validated file is not re-deriving — this lane never captures or derives
    directions at deploy time."""
    if not Path(DIRS_PATH).is_file():
        import hashlib
        from huggingface_hub import hf_hub_download

        print(f"{DIRS_PATH} missing — restoring from "
              f"{VECTOR_REPO}/{VECTOR_FILE}", flush=True)
        fetched = hf_hub_download(VECTOR_REPO, VECTOR_FILE)
        got = hashlib.sha256(Path(fetched).read_bytes()).hexdigest()
        assert got == VECTOR_SHA256, \
            f"{VECTOR_FILE} sha256 {got} != pinned {VECTOR_SHA256}"
        dst = Path(DIRS_PATH)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(Path(fetched).read_bytes())
        vol.commit()
    require_dirs()
    size = Path(DIRS_PATH).stat().st_size
    assert size > 1e6, f"{DIRS_PATH} looks truncated ({size} B)"
    print(f"directions ready: {DIRS_PATH} ({size} B)", flush=True)


@app.function(image=image, volumes={"/data": vol}, gpu=GPU,
              # 30-min idle window: 300s made interactive clients flap
              # orange — every ≥5-min gap cost a ~10-min cold wake
              min_containers=0, max_containers=1, scaledown_window=1800,
              # BOTH startup_timeouts must cover a cold boot: this one gates
              # the runner-init phase that includes waiting for uvicorn to
              # bind (default 1800s killed the 2026-10-04 boot mid-compile
              # even with web_server's at 3600), the web_server's gates the
              # port bind itself.
              startup_timeout=3600,
              timeout=30 * 60,
              env=dict(ENV, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       # compile caches on the volume: without these the first
                       # boot's torch.compile (~20 min for 743B on 8 GPUs)
                       # exceeds the web-server startup window and every
                       # re-boot recompiles from scratch (both sibling lanes
                       # already do this — 2026-10-04 boot-loop fix)
                       TORCHINDUCTOR_CACHE_DIR="/data/cache/inductor",
                       VLLM_CACHE_ROOT="/data/cache/vllm",
                       TRITON_CACHE_DIR="/data/cache/triton",
                       WEIGHTLESS_STEER_PATH=DIRS_PATH,
                       WEIGHTLESS_STEER_ALPHA=os.environ.get("WEIGHTLESS_STEER_ALPHA", "1.0"),
                       # read at deploy time; the container's own env is the
                       # only channel that reaches the serve body
                       MAX_MODEL_LEN=MAX_MODEL_LEN,
                       GLM53XL_GMU=GMU))
@modal.concurrent(max_inputs=32)
@modal.web_server(8000, startup_timeout=3600)
def serve():
    """Cold starts load ~465 GB and compile once; allow up to 60 minutes for
    readiness on an empty compile cache (~15-20 min when warm)."""
    require_dirs()
    subprocess.run(["/usr/bin/python3.12", "/work/" + PATCH_NAME], check=True)
    # Use the image's Python, as in the proven experiment (not Modal's shim).
    # Serving profile: CUDA graphs ON (eager is a capture/eval-lane
    # discipline, never a serving one), prefix caching on, agentic ctx.
    subprocess.Popen([
        "/usr/bin/python3.12", "-m", "vllm.entrypoints.cli.main", "serve", MODEL_ID,
        "--served-model-name", SERVED_MODEL,
        "--host", "0.0.0.0", "--port", "8000",
        "--tensor-parallel-size", "8", "--moe-backend", "marlin",
        "--max-model-len", os.environ["MAX_MODEL_LEN"],
        "--gpu-memory-utilization", os.environ["GLM53XL_GMU"],
        "--enable-auto-tool-choice", "--tool-call-parser", "glm47",
        "--reasoning-parser", "glm45",
        "--default-chat-template-kwargs", '{"enable_thinking": false}',
    ])
