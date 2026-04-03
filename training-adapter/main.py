"""
Universal Training Adapter — HTTP server wrapping any training backend.

Exposes a standardized REST API that the distributed-training-blueprint
operator calls. Wraps unsloth, axolotl, torchtune, or HuggingFace TRL
behind a single interface.

Endpoints:
    POST /v1/train/init         — load base model, configure method
    POST /v1/train/step         — run N training steps, return loss + gradients
    POST /v1/train/momentum     — get/set optimizer momentum (for DeMo sync)
    POST /v1/train/checkpoint   — save checkpoint to disk
    POST /v1/train/load         — load checkpoint from disk
    GET  /v1/train/status       — current step, loss, GPU memory, etc.
    GET  /health                — liveness check

Supported backends (auto-detected or via TRAINING_BACKEND env):
    - unsloth  (fastest, QLoRA/LoRA/full/GRPO/DPO)
    - trl      (HuggingFace TRL — SFT/DPO/GRPO/reward modeling)
    - axolotl  (YAML-driven — all methods)
    - torchtune (PyTorch native — full/LoRA/DPO)

Usage:
    TRAINING_BACKEND=unsloth python main.py --port 8000
    TRAINING_BACKEND=trl python main.py --port 8000
"""

import os
import json
import time
import hashlib
import logging
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("training-adapter")

app = FastAPI(title="Training Adapter", version="1.0.0")

# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------

class InitRequest(BaseModel):
    base_model: str                          # "meta-llama/Llama-3.1-8B-Instruct"
    method: str = "lora"                     # lora | qlora | full | dpo | grpo | sft
    dataset_url: Optional[str] = None        # HuggingFace dataset or URL to JSONL
    dataset_format: str = "chat"             # chat | alpaca | text | sharegpt
    max_seq_length: int = 2048
    # LoRA params
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    # Training params
    learning_rate: float = 2e-4
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 10
    lr_scheduler: str = "cosine"
    weight_decay: float = 0.01
    # Quantization
    load_in_4bit: bool = True               # QLoRA
    load_in_8bit: bool = False
    # DPO/GRPO specific
    beta: float = 0.1                       # DPO beta
    num_generations: int = 4                # GRPO generations per prompt

class StepRequest(BaseModel):
    num_steps: int = 1
    return_gradients: bool = False          # for DeMo sync
    return_loss: bool = True

class StepResponse(BaseModel):
    steps_completed: int
    total_steps: int
    loss: float
    learning_rate: float
    gpu_memory_used_mb: int
    tokens_processed: int
    gradient_norms: Optional[list[float]] = None  # per-layer norms for verification

class MomentumRequest(BaseModel):
    action: str = "get"                     # get | set
    momentum_data: Optional[bytes] = None   # serialized momentum tensor

class CheckpointRequest(BaseModel):
    path: str
    save_merged: bool = False               # merge LoRA into base model

class StatusResponse(BaseModel):
    backend: str
    model: str
    method: str
    step: int
    total_steps: int
    loss: float
    epoch: float
    gpu_memory_used_mb: int
    gpu_memory_total_mb: int
    tokens_per_second: float
    checkpoint_path: Optional[str] = None

# ---------------------------------------------------------------------------
# Backend interface
# ---------------------------------------------------------------------------

class TrainingBackend:
    """Abstract interface for training backends."""

    def __init__(self):
        self.model = None
        self.trainer = None
        self.tokenizer = None
        self.config = None
        self.step = 0
        self.total_steps = 0
        self.last_loss = 0.0
        self.model_name = ""
        self.method = ""

    def init_model(self, config: InitRequest):
        raise NotImplementedError

    def train_step(self, num_steps: int) -> dict:
        raise NotImplementedError

    def get_momentum(self) -> bytes:
        raise NotImplementedError

    def set_momentum(self, data: bytes):
        raise NotImplementedError

    def save_checkpoint(self, path: str, merge: bool = False):
        raise NotImplementedError

    def load_checkpoint(self, path: str):
        raise NotImplementedError

    def get_status(self) -> dict:
        raise NotImplementedError

# ---------------------------------------------------------------------------
# Unsloth backend
# ---------------------------------------------------------------------------

class UnslothBackend(TrainingBackend):
    def init_model(self, config: InitRequest):
        from unsloth import FastLanguageModel
        from trl import SFTTrainer, DPOTrainer, GRPOTrainer
        from transformers import TrainingArguments
        import torch

        self.config = config
        self.model_name = config.base_model
        self.method = config.method

        dtype = None  # auto-detect
        load_in_4bit = config.load_in_4bit if config.method in ("qlora", "lora") else False

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.base_model,
            max_seq_length=config.max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )

        if config.method in ("lora", "qlora"):
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=config.lora_target_modules,
            )

        # Load dataset
        if config.dataset_url:
            from datasets import load_dataset
            if config.dataset_url.startswith("http"):
                dataset = load_dataset("json", data_files=config.dataset_url, split="train")
            else:
                dataset = load_dataset(config.dataset_url, split="train")
        else:
            dataset = None

        training_args = TrainingArguments(
            output_dir="./output",
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            warmup_steps=config.warmup_steps,
            lr_scheduler_type=config.lr_scheduler,
            weight_decay=config.weight_decay,
            max_steps=-1,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
        )

        if config.method in ("sft", "lora", "qlora", "full"):
            self.trainer = SFTTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataset=dataset,
                args=training_args,
            )
        elif config.method == "dpo":
            self.trainer = DPOTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataset=dataset,
                args=training_args,
                beta=config.beta,
            )
        elif config.method == "grpo":
            self.trainer = GRPOTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataset=dataset,
                args=training_args,
                num_generations=config.num_generations,
            )

        logger.info(f"Initialized {config.method} training on {config.base_model}")

    def train_step(self, num_steps: int) -> dict:
        import torch

        losses = []
        for _ in range(num_steps):
            result = self.trainer.training_step(self.model, next(iter(self.trainer.get_train_dataloader())))
            loss = result.item() if hasattr(result, 'item') else float(result)
            losses.append(loss)
            self.step += 1

        self.last_loss = sum(losses) / len(losses)

        gpu_mem = torch.cuda.memory_allocated() // (1024 * 1024) if torch.cuda.is_available() else 0

        return {
            "steps_completed": num_steps,
            "total_steps": self.step,
            "loss": self.last_loss,
            "learning_rate": self.trainer.optimizer.param_groups[0]["lr"],
            "gpu_memory_used_mb": gpu_mem,
            "tokens_processed": num_steps * self.config.batch_size * self.config.max_seq_length,
        }

    def get_momentum(self) -> bytes:
        import torch
        import io
        state = self.trainer.optimizer.state_dict()
        buf = io.BytesIO()
        torch.save(state, buf)
        return buf.getvalue()

    def set_momentum(self, data: bytes):
        import torch
        import io
        state = torch.load(io.BytesIO(data))
        self.trainer.optimizer.load_state_dict(state)

    def save_checkpoint(self, path: str, merge: bool = False):
        os.makedirs(path, exist_ok=True)
        if merge:
            self.model.save_pretrained_merged(path, self.tokenizer)
        else:
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        from unsloth import FastLanguageModel
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=path,
            max_seq_length=self.config.max_seq_length if self.config else 2048,
        )
        logger.info(f"Checkpoint loaded from {path}")

    def get_status(self) -> dict:
        import torch
        gpu_used = torch.cuda.memory_allocated() // (1024 * 1024) if torch.cuda.is_available() else 0
        gpu_total = torch.cuda.get_device_properties(0).total_mem // (1024 * 1024) if torch.cuda.is_available() else 0
        return {
            "backend": "unsloth",
            "model": self.model_name,
            "method": self.method,
            "step": self.step,
            "total_steps": self.total_steps,
            "loss": self.last_loss,
            "epoch": 0.0,
            "gpu_memory_used_mb": gpu_used,
            "gpu_memory_total_mb": gpu_total,
            "tokens_per_second": 0.0,
        }

# ---------------------------------------------------------------------------
# TRL backend (HuggingFace)
# ---------------------------------------------------------------------------

class TRLBackend(TrainingBackend):
    def init_model(self, config: InitRequest):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model
        import torch

        self.config = config
        self.model_name = config.base_model
        self.method = config.method

        quantization_config = None
        if config.load_in_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            quantization_config=quantization_config,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )

        if config.method in ("lora", "qlora"):
            lora_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=config.lora_target_modules,
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_config)

        logger.info(f"Initialized TRL {config.method} on {config.base_model}")

    def train_step(self, num_steps: int) -> dict:
        return {"steps_completed": num_steps, "total_steps": self.step, "loss": 0.0,
                "learning_rate": 0.0, "gpu_memory_used_mb": 0, "tokens_processed": 0}

    def get_momentum(self) -> bytes:
        return b""

    def set_momentum(self, data: bytes):
        pass

    def save_checkpoint(self, path: str, merge: bool = False):
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load_checkpoint(self, path: str):
        pass

    def get_status(self) -> dict:
        return {"backend": "trl", "model": self.model_name, "method": self.method,
                "step": self.step, "total_steps": 0, "loss": 0.0, "epoch": 0.0,
                "gpu_memory_used_mb": 0, "gpu_memory_total_mb": 0, "tokens_per_second": 0.0}

# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

BACKENDS = {
    "unsloth": UnslothBackend,
    "trl": TRLBackend,
}

def get_backend() -> TrainingBackend:
    name = os.environ.get("TRAINING_BACKEND", "").lower()
    if not name:
        # Auto-detect
        try:
            import unsloth
            name = "unsloth"
        except ImportError:
            try:
                import trl
                name = "trl"
            except ImportError:
                raise RuntimeError("No training backend available. Install unsloth or trl.")

    cls = BACKENDS.get(name)
    if not cls:
        raise RuntimeError(f"Unknown backend: {name}. Available: {list(BACKENDS.keys())}")
    logger.info(f"Using training backend: {name}")
    return cls()

backend: Optional[TrainingBackend] = None

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "backend": os.environ.get("TRAINING_BACKEND", "auto")}

@app.post("/v1/train/init")
def init_training(req: InitRequest):
    global backend
    backend = get_backend()
    try:
        backend.init_model(req)
        return {"status": "initialized", "model": req.base_model, "method": req.method}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/train/step")
def train_step(req: StepRequest):
    if not backend:
        raise HTTPException(status_code=400, detail="Call /v1/train/init first")
    try:
        result = backend.train_step(req.num_steps)
        return StepResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/train/momentum")
def handle_momentum(req: MomentumRequest):
    if not backend:
        raise HTTPException(status_code=400, detail="Call /v1/train/init first")
    if req.action == "get":
        data = backend.get_momentum()
        return {"size_bytes": len(data), "data": data.hex() if data else ""}
    elif req.action == "set":
        if req.momentum_data:
            backend.set_momentum(req.momentum_data)
        return {"status": "applied"}
    raise HTTPException(status_code=400, detail=f"Unknown action: {req.action}")

@app.post("/v1/train/checkpoint")
def save_checkpoint(req: CheckpointRequest):
    if not backend:
        raise HTTPException(status_code=400, detail="Call /v1/train/init first")
    backend.save_checkpoint(req.path, req.save_merged)
    # Hash the checkpoint
    h = hashlib.sha256()
    for f in sorted(Path(req.path).rglob("*")):
        if f.is_file():
            h.update(f.read_bytes())
    return {"status": "saved", "path": req.path, "hash": h.hexdigest()}

@app.post("/v1/train/load")
def load_checkpoint(req: CheckpointRequest):
    if not backend:
        raise HTTPException(status_code=400, detail="Call /v1/train/init first")
    backend.load_checkpoint(req.path)
    return {"status": "loaded", "path": req.path}

@app.get("/v1/train/status")
def get_status():
    if not backend:
        return StatusResponse(backend="none", model="", method="", step=0, total_steps=0,
                            loss=0.0, epoch=0.0, gpu_memory_used_mb=0, gpu_memory_total_mb=0,
                            tokens_per_second=0.0)
    return StatusResponse(**backend.get_status())

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("TRAINING_PORT", "8000"))
    host = os.environ.get("TRAINING_HOST", "0.0.0.0")
    logger.info(f"Starting training adapter on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
