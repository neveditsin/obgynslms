import logging
from typing import List, Dict
from transformers import pipeline

try:
    import torch
    _DEVICE = 0 if torch.cuda.is_available() else -1
except Exception:
    torch = None
    _DEVICE = -1

from .experiment_logger import get_logger

NORMALIZE_PROMPT = (
    "You are a medical assistant specialized in obstetric ultrasound. "
    "Given an ultrasound report, produce a concise summary that retains only "
    "pregnancy-related findings. Omit unrelated details.\n\n"
    "Report:\n\"{report_text}\"\n\n"
    "Summary:"
)

JSON_PROMPT = (
    "You are a medical assistant specialized in obstetric ultrasound. "
    "Given an ultrasound report, extract all information in JSON format.\n\n"
    "Report:\n\"{report_text}\"\n\n"
    "Structured report:"
)


def _load_pipeline(model_name: str):
    logger = get_logger(f"norm_{model_name}")
    if torch is None:
        logger.error("PyTorch is not installed; skipping model %s", model_name)
        return None
    try:
        pipe = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=model_name,
            model_kwargs={"torch_dtype": torch.bfloat16},
            #device=_DEVICE,
        )
        logger.info("Loaded model %s for normalization", model_name)
        return pipe
    except Exception as exc:
        logger.error("Failed to load %s: %s", model_name, exc)
        return None


import os
from typing import List, Dict
import torch

import os, pickle, tempfile, time
from typing import List, Dict
import torch

def _atomic_pickle_dump(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".ckpt_", suffix=".pkl")
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)  # atomic on POSIX
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass

def _load_checkpoint(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None

def normalize_reports(
    docs: List[Dict[str, str]],
    model_name: str,
    PROMPT=NORMALIZE_PROMPT,
    checkpoint_path: str | None = None,
) -> List[Dict[str, str]]:
    """Return normalized versions of documents using ``model_name`` with special handling for thinking models."""
    logger = get_logger(f"norm_{model_name}")
    is_gemma = "gemma" in model_name.lower()

    # Default checkpoint location (per-model) if not provided
    if checkpoint_path is None:
        safe_model = "".join(c if c.isalnum() or c in "-._" else "_" for c in model_name)
        checkpoint_path = os.path.join("checkpoints", f"normalize_{safe_model}.pkl")

    # Try to resume
    ckpt = _load_checkpoint(checkpoint_path) or {}
    # ckpt schema: {"model": str, "created": float, "items": List[Dict[file,text]]}
    normalized: List[Dict[str, str]] = ckpt.get("items", [])
    done_files = {it["file"] for it in normalized}

    # Extra guard for Gemma
    if is_gemma:
        os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
        try:
            import torch._dynamo as dynamo
            dynamo.config.suppress_errors = True
        except Exception:
            pass

    pipe = _load_pipeline(model_name)
    if pipe is None:
        return normalized  # return whatever we had

    # Belt-and-suspenders: disable Dynamo on model
    if is_gemma and hasattr(pipe, "model"):
        try:
            import torch._dynamo as dynamo
            pipe.model = dynamo.disable(pipe.model)
        except Exception:
            pass

    # Ensure pad token to reduce shape churn
    try:
        if getattr(pipe, "tokenizer", None) and pipe.tokenizer.pad_token_id is None:
            eos_id = getattr(getattr(pipe, "model", None), "config", None).eos_token_id
            if eos_id is not None:
                pipe.tokenizer.pad_token_id = eos_id
    except Exception:
        pass

    # ---- Special model handling + helpers ----
    SPECIAL_THINKING_MODEL = "Intelligent-Internet/II-Medical-8B"
    # If you want to force "thinking off" for some models in this function, add them here:
    SPECIAL_MODELS_THINKING_OFF = {"Qwen/Qwen3-8B"}

    import re

    def strip_think_block(s: str) -> str:
        # Remove <think>...</think> if present; keep remaining content
        return re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL).strip()

    def after_think_or_strip(s: str) -> str:
        # If explicit closing tag present, keep only the portion after </think>, else strip any block
        if "</think>" in s:
            return s.split("</think>", 1)[1].strip()
        return strip_think_block(s)

    def build_messages(name: str, prompt: str):
        SYSTEM_TXT = "You are a helpful assistant for obstetric ultrasound summarization."
        name_l = name.lower()
        needs_inline_system = any(k in name_l for k in [
            "biomistral",  # BioMistral/BioMistral-7B (doesn't like separate 'system')
        ])
        if needs_inline_system:
            return [{"role": "user", "content": f"{SYSTEM_TXT}\n\n{prompt}"}]
        else:
            return [
                {"role": "system", "content": SYSTEM_TXT},
                {"role": "user", "content": prompt},
            ]

    # Check chat template availability once
    has_chat_template = bool(getattr(getattr(pipe, "tokenizer", None), "chat_template", None))

    try:
        total = len(docs)
        for idx, doc in enumerate(docs, start=1):
            if doc["file"] in done_files:
                logger.info("[resume %s] Skipping already done: %s", model_name, doc["file"])
                continue

            logger.info("[%s] Normalizing file %d/%d: %s", model_name, idx, total, doc["file"])
            prompt = PROMPT.format(report_text=doc["text"])
            response = ""

            # Base gen kwargs; we’ll tweak per-model
            gen_kwargs = dict(max_new_tokens=8192, do_sample=False, use_cache=True)

            try:
                with torch.no_grad(), torch.inference_mode():
                    if model_name == SPECIAL_THINKING_MODEL:
                        # II-Medical-8B: thinking is unavoidable -> allow it, then keep only post-</think>
                        # Also add +256 tokens to your base setting
                        gen_kwargs_local = {**gen_kwargs, "max_new_tokens": gen_kwargs["max_new_tokens"] + 256}
                        messages = build_messages(model_name, prompt)

                        if has_chat_template and hasattr(pipe.tokenizer, "apply_chat_template"):
                            # Build chat string with thinking enabled
                            chat_str = pipe.tokenizer.apply_chat_template(
                                messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
                            )
                            outputs = pipe(
                                chat_str,
                                return_full_text=False,  # only continuation
                                **{k: v for k, v in gen_kwargs_local.items() if k != "use_cache"}  # pipeline ignores use_cache
                            )
                            out_text = outputs[0]["generated_text"]
                        else:
                            # Fallback: plain prompt (in case template missing)
                            plain = f"You are a helpful assistant for obstetric ultrasound summarization.\n\n{prompt}"
                            outputs = pipe(
                                plain,
                                return_full_text=False,
                                **{k: v for k, v in gen_kwargs_local.items() if k != "use_cache"}
                            )
                            out_text = outputs[0]["generated_text"]

                        response = after_think_or_strip(out_text)

                    else:
                        # Other models: prefer chat template if available
                        if has_chat_template and hasattr(pipe.tokenizer, "apply_chat_template"):
                            messages = build_messages(model_name, prompt)

                            # Guard enable_thinking for models that support it; force OFF for specified models
                            add_kwargs = dict(tokenize=False, add_generation_prompt=True)
                            if model_name in SPECIAL_MODELS_THINKING_OFF:
                                try:
                                    chat_str = pipe.tokenizer.apply_chat_template(
                                        messages, enable_thinking=False, **add_kwargs
                                    )
                                except TypeError:
                                    chat_str = pipe.tokenizer.apply_chat_template(messages, **add_kwargs)
                            else:
                                # No explicit thinking flag (some tokenizers don't accept it)
                                chat_str = pipe.tokenizer.apply_chat_template(messages, **add_kwargs)

                            outputs = pipe(
                                chat_str,
                                return_full_text=False,
                                **{k: v for k, v in gen_kwargs.items() if k != "use_cache"}
                            )
                            out_text = outputs[0]["generated_text"]
                            # In case the model still emitted a think block, strip it
                            response = strip_think_block(out_text)

                        else:
                            # No chat template → plain text prompt
                            plain = f"You are a helpful assistant for obstetric ultrasound summarization.\n\n{prompt}"
                            outputs = pipe(
                                plain,
                                return_full_text=False,
                                **{k: v for k, v in gen_kwargs.items() if k != "use_cache"}
                            )
                            out_text = outputs[0]["generated_text"]
                            response = strip_think_block(out_text)

            except Exception as exc:
                logger.error("Normalization failed for %s on %s: %s", model_name, doc["file"], exc)
                response = doc["text"]  # fallback

            item = {"file": doc["file"], "text": response}
            normalized.append(item)
            done_files.add(doc["file"])

            # Save checkpoint after each success
            _atomic_pickle_dump(
                {"model": model_name, "created": ckpt.get("created", time.time()), "updated": time.time(), "items": normalized},
                checkpoint_path,
            )
            logger.info("Checkpoint saved: %s (items=%d)", checkpoint_path, len(normalized))

    finally:
        try:
            del pipe
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        logger.info("Released model and cleared CUDA cache for normalization %s", model_name)

    return normalized


