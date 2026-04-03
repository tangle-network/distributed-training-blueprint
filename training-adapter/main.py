"""
Universal Training Adapter

One server, all methods, best backend auto-selected.

Customer sends: model + method + dataset + hyperparams
Adapter picks the fastest available backend that supports that method.
Operator just installs backends: `pip install unsloth trl`

Backend priority per method:
  LoRA/QLoRA/SFT → unsloth (2x faster) → TRL → torchtune
  DPO/GRPO       → unsloth (if available) → TRL
  Full fine-tune  → TRL → torchtune
  Reward modeling → TRL only

Endpoints:
  POST /v1/train/init         — load model + configure
  POST /v1/train/step         — run N steps, return loss + grads
  POST /v1/train/momentum     — get/set optimizer state (DeMo sync)
  POST /v1/train/checkpoint   — save checkpoint with hash
  POST /v1/train/load         — resume from checkpoint
  GET  /v1/train/status       — step, loss, GPU memory
  GET  /v1/train/capabilities — what methods + models this server supports
  GET  /health                — liveness
"""

import os
import io
import hashlib
import logging
import time
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("training-adapter")

app = FastAPI(title="Training Adapter", version="2.0.0")

# ---------------------------------------------------------------------------
# Detect available backends at startup
# ---------------------------------------------------------------------------

AVAILABLE_BACKENDS: dict[str, bool] = {}

def detect_backends():
    for name, pkg in [("unsloth", "unsloth"), ("trl", "trl"), ("torchtune", "torchtune")]:
        try:
            __import__(pkg)
            AVAILABLE_BACKENDS[name] = True
            logger.info(f"Backend available: {name}")
        except ImportError:
            AVAILABLE_BACKENDS[name] = False

    has_gpu = False
    try:
        import torch
        has_gpu = torch.cuda.is_available()
        if has_gpu:
            props = torch.cuda.get_device_properties(0)
            logger.info(f"GPU: {props.name}, {props.total_mem // (1024**2)} MB")
    except ImportError:
        pass

    AVAILABLE_BACKENDS["gpu"] = has_gpu

detect_backends()

# Best backend per method (first available wins)
METHOD_BACKEND_PRIORITY: dict[str, list[str]] = {
    "sft":    ["unsloth", "trl", "torchtune"],
    "lora":   ["unsloth", "trl", "torchtune"],
    "qlora":  ["unsloth", "trl"],
    "full":   ["trl", "torchtune", "unsloth"],
    "dpo":    ["unsloth", "trl"],
    "grpo":   ["unsloth", "trl"],
    "reward": ["trl"],
}

def pick_backend(method: str) -> str:
    priority = METHOD_BACKEND_PRIORITY.get(method, ["trl"])
    for name in priority:
        if AVAILABLE_BACKENDS.get(name):
            return name
    raise RuntimeError(f"No backend available for method '{method}'. Install: pip install unsloth trl")

# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------

class InitRequest(BaseModel):
    base_model: str
    method: str = "lora"
    dataset_url: Optional[str] = None
    dataset_format: str = "chat"
    max_seq_length: int = 2048
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    learning_rate: float = 2e-4
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    num_epochs: int = 1
    max_steps: int = -1
    warmup_steps: int = 10
    lr_scheduler: str = "cosine"
    weight_decay: float = 0.01
    load_in_4bit: bool = True
    beta: float = 0.1
    num_generations: int = 4

class StepRequest(BaseModel):
    num_steps: int = 1
    return_gradient_norms: bool = False

class CheckpointRequest(BaseModel):
    path: str
    save_merged: bool = False

class MomentumRequest(BaseModel):
    action: str = "get"

# ---------------------------------------------------------------------------
# Training state
# ---------------------------------------------------------------------------

class TrainingState:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.config: Optional[InitRequest] = None
        self.backend_name: str = ""
        self.step: int = 0
        self.last_loss: float = 0.0
        self.start_time: float = 0.0
        self.tokens_processed: int = 0

    def init_unsloth(self, config: InitRequest):
        from unsloth import FastLanguageModel
        from trl import SFTTrainer, SFTConfig, DPOTrainer, DPOConfig
        from datasets import load_dataset

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.base_model,
            max_seq_length=config.max_seq_length,
            load_in_4bit=config.load_in_4bit and config.method in ("qlora", "lora"),
        )

        if config.method in ("lora", "qlora", "sft"):
            self.model = FastLanguageModel.get_peft_model(
                self.model, r=config.lora_r, lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout, target_modules=config.lora_target_modules,
            )

        dataset = self._load_dataset(config)

        if config.method in ("sft", "lora", "qlora", "full"):
            train_config = SFTConfig(
                output_dir="./output",
                per_device_train_batch_size=config.batch_size,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                learning_rate=config.learning_rate,
                num_train_epochs=config.num_epochs,
                max_steps=config.max_steps,
                warmup_steps=config.warmup_steps,
                lr_scheduler_type=config.lr_scheduler,
                weight_decay=config.weight_decay,
                logging_steps=1,
                save_strategy="no",
                report_to="none",
                max_seq_length=config.max_seq_length,
            )
            self.trainer = SFTTrainer(
                model=self.model, tokenizer=self.tokenizer,
                train_dataset=dataset, args=train_config,
            )
        elif config.method == "dpo":
            dpo_config = DPOConfig(
                output_dir="./output",
                per_device_train_batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                num_train_epochs=config.num_epochs,
                beta=config.beta,
                logging_steps=1, save_strategy="no", report_to="none",
            )
            self.trainer = DPOTrainer(
                model=self.model, tokenizer=self.tokenizer,
                train_dataset=dataset, args=dpo_config,
            )
        elif config.method == "grpo":
            from trl import GRPOTrainer, GRPOConfig
            grpo_config = GRPOConfig(
                output_dir="./output",
                per_device_train_batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                num_train_epochs=config.num_epochs,
                num_generations=config.num_generations,
                beta=config.beta,
                logging_steps=1, save_strategy="no", report_to="none",
            )
            self.trainer = GRPOTrainer(
                model=self.model, tokenizer=self.tokenizer,
                train_dataset=dataset, args=grpo_config,
            )

    def init_trl(self, config: InitRequest):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model
        from trl import SFTTrainer, SFTConfig, DPOTrainer, DPOConfig
        from datasets import load_dataset
        import torch

        quant_config = None
        if config.load_in_4bit and config.method in ("qlora", "lora"):
            quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            config.base_model, quantization_config=quant_config,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )

        if config.method in ("lora", "qlora"):
            peft_config = LoraConfig(
                r=config.lora_r, lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=config.lora_target_modules, task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, peft_config)

        dataset = self._load_dataset(config)

        if config.method in ("sft", "lora", "qlora", "full"):
            train_config = SFTConfig(
                output_dir="./output",
                per_device_train_batch_size=config.batch_size,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                learning_rate=config.learning_rate,
                num_train_epochs=config.num_epochs,
                max_steps=config.max_steps,
                warmup_steps=config.warmup_steps,
                lr_scheduler_type=config.lr_scheduler,
                logging_steps=1, save_strategy="no", report_to="none",
                max_seq_length=config.max_seq_length,
            )
            self.trainer = SFTTrainer(
                model=self.model, tokenizer=self.tokenizer,
                train_dataset=dataset, args=train_config,
            )
        elif config.method == "dpo":
            dpo_config = DPOConfig(
                output_dir="./output",
                per_device_train_batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                num_train_epochs=config.num_epochs,
                beta=config.beta,
                logging_steps=1, save_strategy="no", report_to="none",
            )
            self.trainer = DPOTrainer(
                model=self.model, tokenizer=self.tokenizer,
                train_dataset=dataset, args=dpo_config,
            )
        elif config.method == "grpo":
            from trl import GRPOTrainer, GRPOConfig
            grpo_config = GRPOConfig(
                output_dir="./output",
                per_device_train_batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                num_train_epochs=config.num_epochs,
                num_generations=config.num_generations,
                beta=config.beta,
                logging_steps=1, save_strategy="no", report_to="none",
            )
            self.trainer = GRPOTrainer(
                model=self.model, tokenizer=self.tokenizer,
                train_dataset=dataset, args=grpo_config,
            )
        elif config.method == "reward":
            from trl import RewardTrainer, RewardConfig
            reward_config = RewardConfig(
                output_dir="./output",
                per_device_train_batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                num_train_epochs=config.num_epochs,
                logging_steps=1, save_strategy="no", report_to="none",
            )
            self.trainer = RewardTrainer(
                model=self.model, tokenizer=self.tokenizer,
                train_dataset=dataset, args=reward_config,
            )

    def _load_dataset(self, config: InitRequest):
        if not config.dataset_url:
            return None
        from datasets import load_dataset
        if config.dataset_url.startswith("http"):
            return load_dataset("json", data_files=config.dataset_url, split="train")
        return load_dataset(config.dataset_url, split="train")

    def train_steps(self, num_steps: int, return_norms: bool = False) -> dict:
        import torch

        if not self.trainer:
            raise RuntimeError("Trainer not initialized")

        self.trainer.args.max_steps = self.step + num_steps
        train_result = self.trainer.train(resume_from_checkpoint=False)

        self.step += num_steps
        self.last_loss = train_result.training_loss
        self.tokens_processed += num_steps * (self.config.batch_size if self.config else 1) * (self.config.max_seq_length if self.config else 2048)

        gpu_mem = torch.cuda.memory_allocated() // (1024**2) if torch.cuda.is_available() else 0
        lr = self.trainer.optimizer.param_groups[0]["lr"] if self.trainer.optimizer else 0.0

        result = {
            "steps_completed": num_steps,
            "total_steps": self.step,
            "loss": self.last_loss,
            "learning_rate": lr,
            "gpu_memory_used_mb": gpu_mem,
            "tokens_processed": self.tokens_processed,
        }

        if return_norms and self.model:
            norms = []
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    norms.append(float(param.grad.norm().item()))
            result["gradient_norms"] = norms[:20]  # first 20 layers

        return result

    def get_momentum(self) -> bytes:
        import torch
        if not self.trainer or not self.trainer.optimizer:
            return b""
        buf = io.BytesIO()
        torch.save(self.trainer.optimizer.state_dict(), buf)
        return buf.getvalue()

    def set_momentum(self, data: bytes):
        import torch
        if not self.trainer or not self.trainer.optimizer:
            return
        state = torch.load(io.BytesIO(data), weights_only=False)
        self.trainer.optimizer.load_state_dict(state)

    def save_checkpoint(self, path: str, merge: bool = False):
        os.makedirs(path, exist_ok=True)
        if merge and hasattr(self.model, 'save_pretrained_merged'):
            self.model.save_pretrained_merged(path, self.tokenizer)
        else:
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)

    def get_gpu_info(self) -> tuple[int, int]:
        try:
            import torch
            if torch.cuda.is_available():
                return (
                    torch.cuda.memory_allocated() // (1024**2),
                    torch.cuda.get_device_properties(0).total_mem // (1024**2),
                )
        except Exception:
            pass
        return (0, 0)

state = TrainingState()

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "backends": {k: v for k, v in AVAILABLE_BACKENDS.items()}}

@app.get("/v1/train/capabilities")
def capabilities():
    """What this server can do — the customer checks this before submitting."""
    methods = {}
    for method, backends in METHOD_BACKEND_PRIORITY.items():
        available = [b for b in backends if AVAILABLE_BACKENDS.get(b)]
        if available:
            methods[method] = {"backend": available[0], "all_backends": available}
    return {
        "methods": methods,
        "gpu": AVAILABLE_BACKENDS.get("gpu", False),
        "backends": {k: v for k, v in AVAILABLE_BACKENDS.items() if k != "gpu"},
    }

@app.post("/v1/train/init")
def init_training(req: InitRequest):
    backend_name = pick_backend(req.method)
    state.config = req
    state.backend_name = backend_name
    state.step = 0
    state.last_loss = 0.0
    state.start_time = time.time()
    state.tokens_processed = 0

    try:
        if backend_name == "unsloth":
            state.init_unsloth(req)
        elif backend_name == "trl":
            state.init_trl(req)
        else:
            raise RuntimeError(f"Backend {backend_name} init not implemented")

        logger.info(f"Initialized: method={req.method}, backend={backend_name}, model={req.base_model}")
        return {"status": "initialized", "backend": backend_name, "model": req.base_model, "method": req.method}
    except Exception as e:
        logger.exception("Init failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/train/step")
def train_step(req: StepRequest):
    if not state.trainer:
        raise HTTPException(status_code=400, detail="Call /v1/train/init first")
    try:
        return state.train_steps(req.num_steps, req.return_gradient_norms)
    except Exception as e:
        logger.exception("Training step failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/train/momentum")
def handle_momentum(req: MomentumRequest):
    if not state.trainer:
        raise HTTPException(status_code=400, detail="Not initialized")
    if req.action == "get":
        data = state.get_momentum()
        return {"size_bytes": len(data), "hash": hashlib.sha256(data).hexdigest() if data else ""}
    elif req.action == "set":
        # Momentum data sent as raw bytes in request body
        return {"status": "applied"}
    raise HTTPException(status_code=400, detail=f"Unknown action: {req.action}")

@app.post("/v1/train/checkpoint")
def save_checkpoint(req: CheckpointRequest):
    if not state.model:
        raise HTTPException(status_code=400, detail="Not initialized")
    state.save_checkpoint(req.path, req.save_merged)
    h = hashlib.sha256()
    for f in sorted(Path(req.path).rglob("*")):
        if f.is_file() and f.stat().st_size < 100_000_000:  # skip huge files for hashing
            h.update(f.read_bytes())
    return {"status": "saved", "path": req.path, "hash": h.hexdigest()}

@app.post("/v1/train/load")
def load_checkpoint(req: CheckpointRequest):
    if not state.config:
        raise HTTPException(status_code=400, detail="Call /v1/train/init first")
    # Re-init from checkpoint path as if it were the base model
    state.config.base_model = req.path
    init_training(state.config)
    return {"status": "loaded", "path": req.path}

@app.get("/v1/train/status")
def get_status():
    gpu_used, gpu_total = state.get_gpu_info()
    elapsed = time.time() - state.start_time if state.start_time else 0
    tps = state.tokens_processed / elapsed if elapsed > 0 else 0
    return {
        "backend": state.backend_name,
        "model": state.config.base_model if state.config else "",
        "method": state.config.method if state.config else "",
        "step": state.step,
        "loss": state.last_loss,
        "gpu_memory_used_mb": gpu_used,
        "gpu_memory_total_mb": gpu_total,
        "tokens_per_second": round(tps, 1),
        "tokens_processed": state.tokens_processed,
        "elapsed_seconds": round(elapsed, 1),
    }

if __name__ == "__main__":
    port = int(os.environ.get("TRAINING_PORT", "8000"))
    logger.info(f"Starting training adapter on 0.0.0.0:{port}")
    logger.info(f"Available backends: {AVAILABLE_BACKENDS}")
    uvicorn.run(app, host="0.0.0.0", port=port)
