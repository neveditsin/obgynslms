import logging
from pathlib import Path
from typing import List, Dict

import pandas as pd
try:
    import torch
    _DEVICE = 0 if torch.cuda.is_available() else -1
except Exception:
    torch = None
    _DEVICE = -1

from .utils import load_checkpoint, save_checkpoint
from .experiment_logger import get_logger


PROMPT_TEMPLATE = (
    "You are a medical assistant specialized in Obstetric and Gynecologic Ultrasound.\n"
    "\n"
    "Task: Classify the document-level ACTIVE pregnancy status at the time of this report, using ONLY what is stated or strongly implied in the report (no outside chart review).\n"
    "\n"
    "Choose exactly ONE label from:\n"
    "- early active pregnancy\n"
    "- late active pregnancy\n"
    "- no active pregnancy\n"
    "\n"
    "Key definition (ACTIVE pregnancy):\n"
    "\"Active\" means pregnancy tissue/trophoblastic tissue is currently present or reasonably suspected to be present such that pregnancy-specific evaluation/management is ongoing, regardless of viability. This includes:\n"
    "- confirmed/suspected intrauterine pregnancy (IUP), including nonviable IUP if management is not documented as complete,\n"
    "- confirmed/suspected ectopic pregnancy,\n"
    "- pregnancy of unknown location (PUL) when ongoing pregnancy/ectopic is still being worked up,\n"
    "- vascularized retained products of conception (RPOC) with suspected persistent trophoblastic activity.\n"
    "Do NOT switch to \"no active pregnancy\" at the moment nonviability is diagnosed; switch only when the report documents resolution/completion (e.g., confirmed passage, completed procedure, or follow-up showing no persistent active pregnancy).\n"
    "\n"
    "Decision rules (apply in order):\n"
    "1) Decide ACTIVE vs NO ACTIVE first:\n"
    "   - If the report raises ongoing pregnancy/ectopic/PUL as an active diagnostic possibility (even \"cannot exclude\", \"correlate with beta-hCG\", serial follow-up), classify as ACTIVE (usually early active pregnancy).\n"
    "   - If the report describes resolved/post-management state with no ongoing pregnancy concern (postpartum; completed miscarriage with passage or completed treatment; post-treatment ectopic follow-up without persistent concern; non-obstetric gyn imaging), classify as NO ACTIVE.\n"
    "\n"
    "2) If ACTIVE, split early vs late using best GA evidence:\n"
    "   - early active pregnancy: GA < 14 weeks\n"
    "   - late active pregnancy: GA >= 14 weeks\n"
    "   GA may come from explicit GA, EDD inference, LMP estimate, fetal biometry, or clear exam context.\n"
    "\n"
    "3) If ACTIVE but GA cannot be determined:\n"
    "   - Default to early active pregnancy unless the report clearly indicates second-trimester/later pregnancy (e.g., anatomy survey, amniocentesis, established pregnancy procedures).\n"
    "\n"
    "RPOC rule:\n"
    "- Vascularized RPOC with suspected persistent trophoblastic activity / ongoing pregnancy-related management concern -> ACTIVE (typically early).\n"
    "- Non-vascularized RPOC with no gestational sac in post-miscarriage/post-procedure follow-up -> NO ACTIVE.\n"
    "\n"
    "Only output the label (exactly one of the three). Do not output any other text.\n"
    "\n"
    "Report:\n\"{report_text}\"\n\n"
    "Label: "
)


import os

import os


def _load_pipeline(model_name: str):
    logger = get_logger(model_name)
    if torch is None:
        logger.error("PyTorch is not installed; skipping model %s", model_name)
        return None

    # Imported here rather than at module scope so that importing this module
    # (and therefore collecting the test suite) does not require transformers.
    from transformers import pipeline

    try:
        # Default fast path
        if _DEVICE >= 0:
            # Prefer GPU with auto placement when available.
            pipe = pipeline(
                "text-generation",
                model=model_name,
                tokenizer=model_name,
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map="auto",
            )
        else:
            pipe = pipeline(
                "text-generation",
                model=model_name,
                tokenizer=model_name,
                model_kwargs={"torch_dtype": "auto"},
                device=-1,
            )
        logger.info("Loaded model %s", model_name)
        return pipe

    except Exception as exc:
        logger.error("Failed to load %s: %s", model_name, exc)
        return None



def run_zero_shot_classification(
    docs: List[Dict[str, str]],
    model_name: str,
    results_dir: str = "results",
    PROMPT_TEMPLATE = PROMPT_TEMPLATE,
) -> pd.DataFrame:
    """
    Zero-shot run over the configured models:
      - Uses text-generation pipeline (loaded once)
      - If chat template exists: format with apply_chat_template
      - Else: fallback to plain text
      - Uses return_full_text=False everywhere
    """
    logger = get_logger(model_name)
    results_path = Path(results_dir) / f"results_{model_name.replace('/', '_')}.pkl"
    df = load_checkpoint(results_path)
    processed = set(df["file"].tolist())

    SYSTEM_TXT = "You are a helpful assistant for obstetric ultrasound classification."

    # --- helpers ---
    def build_messages(name: str, prompt: str):
        name_l = name.lower()
        needs_inline_system = any(k in name_l for k in ["biomistral", "gemma"])
        if needs_inline_system:
            return [{"role": "user", "content": f"{SYSTEM_TXT}\n\n{prompt}"}]
        else:
            return [
                {"role": "system", "content": SYSTEM_TXT},
                {"role": "user", "content": prompt},
            ]

    # --- pipeline loading (once) ---
    pipe = _load_pipeline(model_name)
    if pipe is None:
        return df

    # Cache this for the pipeline path
    has_chat_template = False
    if pipe is not None and hasattr(pipe, "tokenizer"):
        has_chat_template = bool(getattr(pipe.tokenizer, "chat_template", None))

    try:
        for idx, doc in enumerate(docs, start=1):
            if doc["file"] in processed:
                continue

            logger.info("[%s] Processing file %d/%d: %s", model_name, idx, len(docs), doc["file"])
            if callable(PROMPT_TEMPLATE):
                try:
                    prompt = PROMPT_TEMPLATE(doc)
                except Exception as exc:
                    logger.error("%s: prompt builder failed for %s: %s", model_name, doc.get("file"), exc)
                    continue
            else:
                prompt = PROMPT_TEMPLATE.format(report_text=doc["text"])
            messages = build_messages(model_name, prompt)

            if has_chat_template and hasattr(pipe.tokenizer, "apply_chat_template"):
                add_kwargs = dict(tokenize=False, add_generation_prompt=True)

                try:
                    formatted = pipe.tokenizer.apply_chat_template(messages, **add_kwargs)
                except TypeError as e:
                    # Final fallback if tokenizer insists on different kwargs
                    formatted = pipe.tokenizer.apply_chat_template(messages, tokenize=False)

                outputs = pipe(
                    formatted,
                    max_new_tokens=200,
                    max_length=None,
                    do_sample=False,
                    return_full_text=False,
                )
                response = outputs[0]["generated_text"]
            else:
                # No chat template -> plain text prompt
                plain = f"{SYSTEM_TXT}\n\n{prompt}"
                outputs = pipe(
                    plain,
                    max_new_tokens=200,
                    max_length=None,
                    do_sample=False,
                    return_full_text=False,
                )
                response = outputs[0]["generated_text"]

            # Append and checkpoint
            new_row = {
                "file": doc["file"],
                "model": model_name,
                "raw_output": response,
            }
            prompt_value = None
            if callable(PROMPT_TEMPLATE):
                try:
                    # Rebuild prompt again for logging; safe because builder should be deterministic
                    prompt_value = PROMPT_TEMPLATE(doc)
                except Exception:
                    prompt_value = None
            else:
                prompt_value = PROMPT_TEMPLATE.format(report_text=doc["text"])
            if prompt_value is not None:
                new_row["prompt"] = prompt_value
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            save_checkpoint(df, results_path)

    finally:
        # Cleanup
        if pipe is not None:
            del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
        logger.info("Released model and cleared CUDA cache for %s", model_name)

    return df
