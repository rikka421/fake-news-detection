from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_analysis_progress.data import load_news_dataset, split_dataset

LABELS = [
    "fake",
    "satire",
    "bias",
    "conspiracy",
    "junksci",
    "hate",
    "clickbait",
    "unreliable",
    "political",
    "reliable",
]


def _prompt_prefix() -> str:
    labels = ", ".join(LABELS)
    return (
        "You are a news reliability classifier. "
        "Read the text and output exactly one label from this set: "
        f"{labels}.\n"
        "Only output the label word, nothing else.\n\n"
        "News text:\n"
    )


def _prompt_suffix() -> str:
    return "\n\nLabel:"


def _extract_label(text: str) -> tuple[str, bool]:
    normalized = text.strip().lower()
    for label in LABELS:
        if re.search(rf"\b{re.escape(label)}\b", normalized):
            return label, True
    return "unreliable", False


def _build_prompt(text: str) -> str:
    return _prompt_prefix() + text + _prompt_suffix()


def _truncate_text_for_prompt(
    tokenizer: AutoTokenizer,
    text: str,
    max_length: int,
) -> str:
    overhead_ids = tokenizer(_build_prompt(""), add_special_tokens=False).input_ids
    # Keep a small token cushion for tokenizer variations.
    max_text_tokens = max(16, max_length - len(overhead_ids) - 8)
    text_ids = tokenizer(text, add_special_tokens=False).input_ids[:max_text_tokens]
    return tokenizer.decode(text_ids, skip_special_tokens=True)


def _predict_label_by_likelihood(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    device: str,
    max_length: int = 512,
) -> str:
    prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).input_ids.to(device)
    prompt_len = int(prompt_ids.shape[1])
    best_label = LABELS[0]
    best_score = float("-inf")

    for label in LABELS:
        label_ids = tokenizer(" " + label, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        input_ids = torch.cat([prompt_ids, label_ids], dim=1)
        if input_ids.shape[1] > max_length:
            input_ids = input_ids[:, -max_length:]

        with torch.no_grad():
            logits = model(input_ids).logits

        start = min(prompt_len, int(input_ids.shape[1]) - int(label_ids.shape[1]))
        target_ids = input_ids[:, start:]
        if target_ids.numel() == 0 or start == 0:
            continue

        pred_logits = logits[:, start - 1 : input_ids.shape[1] - 1, :]
        token_log_probs = F.log_softmax(pred_logits, dim=-1)
        gathered = token_log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        score = float(gathered.mean().item())

        if score > best_score:
            best_score = score
            best_label = label

    return best_label


def run_eval(
    model_name: str,
    dataset: str,
    text_mode: str,
    split_mode: str,
    mask_label_tokens: bool,
    max_samples: int,
    method: str,
    sample_mode: str,
    seed: int,
    precision: str,
    output: Path | None,
) -> dict:
    startup_begin = time.perf_counter()
    frame = load_news_dataset(dataset, text_mode=text_mode, mask_label_tokens=mask_label_tokens)
    _, _, test_frame = split_dataset(frame, random_state=42, split_mode=split_mode)

    total_test_samples = int(len(test_frame))

    if max_samples > 0:
        if sample_mode == "random":
            test_frame = test_frame.sample(n=min(max_samples, len(test_frame)), random_state=seed)
        else:
            test_frame = test_frame.head(max_samples)
    test_frame = test_frame.reset_index(drop=True)

    tokenizer_begin = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer_load_sec = time.perf_counter() - tokenizer_begin
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if precision == "auto":
        chosen_precision = "fp16" if device == "cuda" else "fp32"
    else:
        chosen_precision = precision

    dtype_map = {
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    torch_dtype = dtype_map[chosen_precision]

    model_begin = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch_dtype)
    model_load_sec = time.perf_counter() - model_begin
    model.to(device)
    model.eval()
    startup_runtime_sec = time.perf_counter() - startup_begin

    y_true: list[str] = []
    y_pred: list[str] = []
    sample_times_sec: list[float] = []
    extraction_hits = 0

    start_time = time.perf_counter()

    for _, row in test_frame.iterrows():
        one_start = time.perf_counter()
        trimmed_text = _truncate_text_for_prompt(tokenizer, str(row["text"]), max_length=512)
        prompt = _build_prompt(trimmed_text)
        if method == "generation":
            encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            encoded = {k: v.to(device) for k, v in encoded.items()}

            with torch.no_grad():
                generated = model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=6,
                    num_beams=1,
                    pad_token_id=tokenizer.eos_token_id,
                )

            output_text = tokenizer.decode(generated[0][encoded["input_ids"].shape[1]:], skip_special_tokens=True)
            pred, matched = _extract_label(output_text)
            if matched:
                extraction_hits += 1
        else:
            pred = _predict_label_by_likelihood(model, tokenizer, prompt, device=device)
            extraction_hits += 1

        y_true.append(str(row["label"]))
        y_pred.append(pred)
        sample_times_sec.append(time.perf_counter() - one_start)

    total_runtime_sec = time.perf_counter() - start_time
    end_to_end_runtime_sec = startup_runtime_sec + total_runtime_sec
    avg_sample_time_sec = total_runtime_sec / max(1, len(y_true))
    throughput_samples_per_sec = len(y_true) / max(total_runtime_sec, 1e-9)
    projected_runtime_10k_sec = avg_sample_time_sec * 10000

    result = {
        "model": model_name,
        "dataset": dataset,
        "samples": int(len(y_true)),
        "total_test_samples": total_test_samples,
        "text_mode": text_mode,
        "split_mode": split_mode,
        "mask_label_tokens": bool(mask_label_tokens),
        "method": method,
        "sample_mode": sample_mode,
        "seed": seed,
        "device": device,
        "precision": chosen_precision,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "tokenizer_load_sec": float(tokenizer_load_sec),
        "model_load_sec": float(model_load_sec),
        "startup_runtime_sec": float(startup_runtime_sec),
        "end_to_end_runtime_sec": float(end_to_end_runtime_sec),
        "total_runtime_sec": float(total_runtime_sec),
        "avg_sample_time_sec": float(avg_sample_time_sec),
        "throughput_samples_per_sec": float(throughput_samples_per_sec),
        "projected_runtime_10k_sec": float(projected_runtime_10k_sec),
        "projected_runtime_10k_min": float(projected_runtime_10k_sec / 60.0),
        "projected_runtime_10k_hours": float(projected_runtime_10k_sec / 3600.0),
        "sample_time_p50_sec": float(sorted(sample_times_sec)[len(sample_times_sec) // 2]) if sample_times_sec else 0.0,
        "extraction_hit_rate": float(extraction_hits / max(1, len(y_true))),
    }

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate <=0.5B LLM zero-shot fake-news classification")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--dataset", default="balanced_large")
    parser.add_argument("--text-mode", default="title_content", choices=["all_fields", "title_content", "content_only"])
    parser.add_argument("--split-mode", default="grouped_title_content", choices=["random", "grouped_content", "grouped_title_content"])
    parser.add_argument("--mask-label-tokens", action="store_true")
    parser.add_argument("--max-samples", type=int, default=300)
    parser.add_argument("--method", default="likelihood", choices=["likelihood", "generation"])
    parser.add_argument("--sample-mode", default="random", choices=["random", "head"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", default="auto", choices=["auto", "fp16", "fp32"])
    parser.add_argument("--output", default="artifacts/results/llm_qwen_0p5b_zero_shot_balanced_large.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_eval(
        model_name=args.model,
        dataset=args.dataset,
        text_mode=args.text_mode,
        split_mode=args.split_mode,
        mask_label_tokens=args.mask_label_tokens,
        max_samples=args.max_samples,
        method=args.method,
        sample_mode=args.sample_mode,
        seed=args.seed,
        precision=args.precision,
        output=Path(args.output) if args.output else None,
    )


if __name__ == "__main__":
    main()
