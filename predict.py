"""
AutoLLM — CLI Inference Script
================================
Run predictions with a trained AutoLLM model.

You can point to either:
  --experiment  PATH   The experiment folder (experiments/<run>).
                       Automatically finds the best model via config.pkl.
  --model       PATH   A specific saved model folder (models/<run>_<name>).
                       You must also provide --experiment for the label encoder,
                       or the raw numeric class IDs will be shown.

Usage examples
--------------
  # Single text prediction
  python predict.py --experiment experiments/20260422_123456 --text "Great product!"

  # Batch prediction from a file (auto-detect text column)
  python predict.py --experiment experiments/20260422_123456 --file new_reviews.csv

  # Batch prediction with explicit text column, save output
  python predict.py --experiment experiments/20260422_123456 \\
                    --file new_reviews.xlsx --text-column review_body \\
                    --output predictions.csv

  # Show top-2 classes with confidence scores
  python predict.py --experiment experiments/20260422_123456 \\
                    --text "Not sure about this!" --top-k 2

Run `python predict.py --help` for the full argument reference.
"""

import argparse
import sys
import os
import pickle
import time
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# Constants — same tokenizer overrides as model_trainer.py
# ──────────────────────────────────────────────────────────────────────────────

_TOKENIZER_OVERRIDES = {
    "prajjwal1/bert-tiny":   "bert-base-uncased",
    "prajjwal1/bert-mini":   "bert-base-uncased",
    "prajjwal1/bert-small":  "bert-base-uncased",
    "prajjwal1/bert-medium": "bert-base-uncased",
}


# ──────────────────────────────────────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="predict.py",
        description="AutoLLM — run inference with a trained classification model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py --experiment experiments/20260422_123456 --text "Amazing product!"
  python predict.py --experiment experiments/20260422_123456 --file emails.csv
  python predict.py --experiment experiments/20260422_123456 --file data.xlsx --text-column body --output out.csv
        """,
    )

    # ── Source ──
    src = p.add_argument_group("model source (at least one required)")
    src.add_argument(
        "--experiment", "-e",
        metavar="DIR",
        default=None,
        help=(
            "Path to an AutoLLM experiment folder (e.g. experiments/20260422_123456). "
            "Loads the best model and label encoder saved by train.py."
        ),
    )
    src.add_argument(
        "--model", "-m",
        metavar="DIR",
        default=None,
        help=(
            "Path to a specific model folder (e.g. models/20260422_distilbert-base-uncased). "
            "When used without --experiment, integer class IDs are returned."
        ),
    )

    # ── Input ──
    inp = p.add_argument_group("input (provide --text OR --file)")
    inp_grp = inp.add_mutually_exclusive_group(required=True)
    inp_grp.add_argument(
        "--text", "-t",
        metavar="TEXT",
        help="Single text string to classify.",
    )
    inp_grp.add_argument(
        "--file", "-f",
        metavar="PATH",
        help="CSV or XLSX file for batch prediction. Text column is auto-detected.",
    )

    # ── Batch options ──
    bat = p.add_argument_group("batch options (when using --file)")
    bat.add_argument(
        "--text-column",
        metavar="COLUMN",
        default=None,
        help="Name of the text column in the input file. Auto-detected if omitted.",
    )
    bat.add_argument(
        "--output", "-o",
        metavar="PATH",
        default=None,
        help="Save predictions to this CSV file. Defaults to printing to the terminal.",
    )
    bat.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="Batch size for model inference (default: 32).",
    )

    # ── Display ──
    disp = p.add_argument_group("display options")
    disp.add_argument(
        "--top-k",
        type=int,
        default=1,
        metavar="K",
        help="Show top-K predicted classes with confidence scores (default: 1).",
    )

    return p


# ──────────────────────────────────────────────────────────────────────────────
# Config loading
# ──────────────────────────────────────────────────────────────────────────────

def _load_config(experiment_dir: Path) -> dict:
    """Load config.pkl from an experiment directory."""
    cfg_path = experiment_dir / "config.pkl"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"config.pkl not found in {experiment_dir}. "
            "Make sure this is a completed AutoLLM experiment folder."
        )
    with open(cfg_path, "rb") as f:
        return pickle.load(f)


def _resolve_model_path(args, config: dict) -> Path:
    """
    Return the model directory to load.
    Priority: --model > best_model_path stored in config.
    """
    if args.model:
        p = Path(args.model)
        if not p.exists():
            raise FileNotFoundError(f"Model directory not found: {args.model}")
        return p

    # Find best model path from training_results stored in config
    training_results = config.get("training_results")
    if not training_results:
        raise ValueError(
            "No training_results found in config.pkl. "
            "Cannot determine which model to load. "
            "Use --model to specify the model path directly."
        )

    # The best model is the first one in training_results that still exists on disk.
    # training_results is ordered: best first (as saved by pipeline).
    # We pick the first one whose path exists.
    for tr in training_results:
        mp = Path(tr.get("model_path", ""))
        if mp.exists():
            return mp

    # Fallback: pick by highest F1 from the model dirs in experiment folder
    raise FileNotFoundError(
        "None of the trained model paths recorded in config.pkl exist on disk. "
        "Use --model to provide the path manually."
    )


def _detect_model_name(model_path: Path) -> str:
    """
    Infer the HuggingFace model ID from the model folder name.
    Folder names follow the pattern: <timestamp>_<model-name>
    """
    # e.g. "20260422_123456_distilbert-base-uncased" → "distilbert-base-uncased"
    # e.g. "20260422_123456_prajjwal1/bert-tiny" won't happen (/ is not in folder names)
    # prajjwal1 models are saved as "bert-tiny", so we expand them back.
    name = model_path.name
    # Strip timestamp prefix  (format: YYYYMMDD_HHMMSS_)
    parts = name.split("_", 2)
    if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
        base = parts[2]
    else:
        base = name  # couldn't parse; use as-is

    # Expand tiny/mini back to prajjwal1/ scope
    _expand = {
        "bert-tiny":         "prajjwal1/bert-tiny",
        "bert-mini":         "prajjwal1/bert-mini",
        "bert-small":        "prajjwal1/bert-small",
        "bert-medium":       "prajjwal1/bert-medium",
        "mobilebert-uncased": "google/mobilebert-uncased",
    }
    return _expand.get(base, base)


# ──────────────────────────────────────────────────────────────────────────────
# Model + tokenizer loading
# ──────────────────────────────────────────────────────────────────────────────

def _load_model_and_tokenizer(model_path: Path, num_labels: int, device):
    """Load the saved model weights and the associated tokenizer."""
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        BertForSequenceClassification,
        BertTokenizer,
    )

    model_name = _detect_model_name(model_path)
    tokenizer_name = _TOKENIZER_OVERRIDES.get(model_name, model_name)

    # Load tokenizer — try from model_path first (it's saved there by the trainer),
    # then fall back to hub if any files are missing.
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    except Exception:
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        except Exception as e:
            raise RuntimeError(
                f"Could not load tokenizer for '{model_name}' from '{model_path}' "
                f"or from the HuggingFace hub ('{tokenizer_name}'). "
                f"Ensure you have an internet connection or cached models. Error: {e}"
            )

    # prajjwal1 models need explicit BertForSequenceClassification
    _bert_models = {
        "prajjwal1/bert-tiny", "prajjwal1/bert-mini",
        "prajjwal1/bert-small", "prajjwal1/bert-medium",
    }
    if model_name in _bert_models:
        model = BertForSequenceClassification.from_pretrained(
            str(model_path), num_labels=num_labels
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path), num_labels=num_labels
        )

    model.eval()
    model.to(device)
    return model, tokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ──────────────────────────────────────────────────────────────────────────────

def _predict_batch(
    texts: list,
    model,
    tokenizer,
    max_length: int,
    device,
    batch_size: int = 32,
    top_k: int = 1,
    is_multi_label: bool = False,
) -> list:
    """
    Run inference on a list of texts.
    Returns a list of dicts: {predicted_class_id, confidence, top_k: [...]}
    For multi-label returns {predicted_labels_ids: [...], confidences: [...]}
    """
    import torch
    import numpy as np

    results = []
    model.eval()

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = [str(t) for t in texts[start : start + batch_size]]
            enc = tokenizer(
                batch_texts,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            if is_multi_label:
                probs = torch.sigmoid(outputs.logits).cpu().numpy()
                for row_probs in probs:
                    predicted_ids = [int(i) for i, p in enumerate(row_probs) if p >= 0.5]
                    results.append(
                        {
                            "predicted_labels_ids": predicted_ids,
                            "confidences": [float(row_probs[i]) for i in predicted_ids],
                            "all_probs": row_probs.tolist(),
                        }
                    )
            else:
                probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
                for row_probs in probs:
                    top_indices = np.argsort(row_probs)[::-1][:top_k]
                    results.append(
                        {
                            "predicted_class_id": int(top_indices[0]),
                            "confidence": float(row_probs[top_indices[0]]),
                            "top_k": [
                                {"class_id": int(i), "confidence": float(row_probs[i])}
                                for i in top_indices
                            ],
                        }
                    )

    return results


def _decode_results(raw_results: list, label_encoder=None, is_multi_label: bool = False) -> list:
    """Replace integer class IDs with human-readable label strings (if encoder available)."""
    decoded = []
    for r in raw_results:
        entry = dict(r)
        if is_multi_label:
            # label_encoder is a MultiLabelBinarizer
            if label_encoder is not None:
                try:
                    # Build binary indicator row and inverse-transform
                    n_classes = len(label_encoder.classes_)
                    indicator = [0] * n_classes
                    for idx in r.get("predicted_labels_ids", []):
                        if idx < n_classes:
                            indicator[idx] = 1
                    labels_tuple = label_encoder.inverse_transform([indicator])[0]
                    entry["predicted_label"] = ", ".join(labels_tuple) if labels_tuple else "(none)"
                    # Per-label confidences
                    entry["top_k"] = [
                        {"label": label_encoder.classes_[i], "confidence": c}
                        for i, c in zip(r.get("predicted_labels_ids", []), r.get("confidences", []))
                    ]
                except Exception:
                    entry["predicted_label"] = str(r.get("predicted_labels_ids", []))
            else:
                entry["predicted_label"] = str(r.get("predicted_labels_ids", []))
        else:
            if label_encoder is not None:
                try:
                    entry["predicted_label"] = label_encoder.inverse_transform(
                        [r["predicted_class_id"]]
                    )[0]
                    entry["top_k"] = [
                        {
                            "label": label_encoder.inverse_transform([t["class_id"]])[0],
                            "confidence": t["confidence"],
                        }
                        for t in r["top_k"]
                    ]
                except Exception:
                    entry["predicted_label"] = str(r["predicted_class_id"])
            else:
                entry["predicted_label"] = str(r["predicted_class_id"])
        decoded.append(entry)
    return decoded


# ──────────────────────────────────────────────────────────────────────────────
# File loading (mirrors DataValidator, without the training-focused checks)
# ──────────────────────────────────────────────────────────────────────────────

def _load_file(path: Path):
    """Load CSV or XLSX into a DataFrame."""
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        try:
            import openpyxl  # noqa: F401
        except ImportError:
            print("ERROR: Reading .xlsx requires 'openpyxl'. Install with: pip install openpyxl")
            sys.exit(1)
        return __import__("pandas").read_excel(str(path))
    else:
        import pandas as pd
        try:
            return pd.read_csv(str(path), encoding="utf-8")
        except UnicodeDecodeError:
            return pd.read_csv(str(path), encoding="latin-1")


def _detect_text_column(df, hint: str = None) -> str:
    """Return the column name most likely to contain text."""
    if hint:
        if hint not in df.columns:
            print(f"ERROR: --text-column '{hint}' not found. Available: {list(df.columns)}")
            sys.exit(1)
        return hint

    # Heuristic: pick non-numeric column with highest average string length
    candidates = []
    for col in df.columns:
        avg_len = df[col].astype(str).str.len().mean()
        candidates.append((col, avg_len))

    candidates.sort(key=lambda x: x[1], reverse=True)
    if not candidates:
        print("ERROR: No columns found in the input file.")
        sys.exit(1)

    chosen = candidates[0][0]
    print(f"  Auto-detected text column: '{chosen}'")
    return chosen


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = build_parser()
    args = parser.parse_args()

    # ── Validate sources ───────────────────────────────────────────────────────
    if args.experiment is None and args.model is None:
        parser.error("Provide at least one of --experiment or --model.")

    experiment_dir = Path(args.experiment) if args.experiment else None
    if experiment_dir and not experiment_dir.exists():
        parser.error(f"Experiment directory not found: {args.experiment}")

    print("\n" + "=" * 70)
    print("  AutoLLM — Inference")
    print("=" * 70)

    # ── Load config ──────────────────────────────────────────────────────────
    config = {}
    label_encoder = None
    is_multi_label = False
    max_length = 128  # safe default; overridden from config if available

    if experiment_dir:
        print(f"\n  Loading config from: {experiment_dir}")
        try:
            config = _load_config(experiment_dir)
            label_encoder = config.get("label_encoder")
            is_multi_label = bool(config.get("is_multi_label", False))
            analysis = config.get("analysis") or {}
            max_length = (
                analysis.get("text_info", {}).get("p95_length", 128)
            )
            print(f"  max_length       : {max_length}")
            print(f"  multi-label      : {is_multi_label}")
            if label_encoder is not None:
                classes = list(label_encoder.classes_)
                print(f"  Classes ({len(classes)})      : {classes}")
        except FileNotFoundError as e:
            print(f"\n  WARNING: {e}")
            print("  Continuing without label encoder — raw class IDs will be shown.\n")

    # ── Resolve model path ────────────────────────────────────────────────────
    try:
        model_path = _resolve_model_path(args, config)
    except (FileNotFoundError, ValueError) as e:
        print(f"\n  ERROR: {e}")
        sys.exit(1)

    print(f"  Model path       : {model_path}")
    model_name = _detect_model_name(model_path)
    print(f"  Model name       : {model_name}")

    # ── Determine number of labels ────────────────────────────────────────────
    if label_encoder is not None:
        num_labels = len(label_encoder.classes_)
    else:
        # Try to read from the model's config.json
        import json
        cfg_json = model_path / "config.json"
        if cfg_json.exists():
            with open(cfg_json) as fj:
                num_labels = json.load(fj).get("num_labels", 2)
        else:
            num_labels = 2

    # ── Load model ────────────────────────────────────────────────────────────
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device           : {device}")
    print()

    try:
        model, tokenizer = _load_model_and_tokenizer(model_path, num_labels, device)
    except RuntimeError as e:
        print(f"\n  ERROR loading model: {e}")
        sys.exit(1)

    # ──────────────────────────────────────────────────────────────────────────
    # SINGLE TEXT MODE
    # ──────────────────────────────────────────────────────────────────────────
    if args.text:
        texts = [args.text]
        t0 = time.perf_counter()
        raw = _predict_batch(
            texts, model, tokenizer, max_length, device,
            batch_size=1, top_k=args.top_k,
            is_multi_label=is_multi_label,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        decoded = _decode_results(raw, label_encoder, is_multi_label=is_multi_label)

        r = decoded[0]
        print("-" * 70)
        print(f"  Input text    : {args.text[:120]}")
        print(f"  Prediction    : {r['predicted_label']}")
        if is_multi_label:
            top_k_items = r.get("top_k", [])
            if top_k_items:
                print(f"  Per-label scores:")
                for t in top_k_items:
                    print(f"    {t['label']:<25}  {t['confidence']*100:.1f}%")
        else:
            print(f"  Confidence    : {r['confidence']*100:.1f}%")
            if args.top_k > 1:
                print(f"\n  Top-{args.top_k} predictions:")
                for rank, t in enumerate(r["top_k"], 1):
                    label_str = t.get("label", t.get("class_id"))
                    conf = t["confidence"] * 100
                    print(f"    {rank}. {label_str:<25}  {conf:.1f}%")

        print(f"\n  Inference time: {elapsed_ms:.2f} ms  (tokenization + model forward)")
        print("=" * 70 + "\n")
        sys.exit(0)

    # ──────────────────────────────────────────────────────────────────────────
    # BATCH FILE MODE
    # ──────────────────────────────────────────────────────────────────────────
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"\n  ERROR: File not found: {args.file}")
        sys.exit(1)

    print(f"  Loading file: {file_path}")
    import pandas as pd
    df = _load_file(file_path)
    print(f"  Rows: {len(df)}")

    text_col = _detect_text_column(df, args.text_column)
    texts = df[text_col].astype(str).tolist()

    print(f"  Running inference on {len(texts)} texts  (batch_size={args.batch_size}) ...")
    t0 = time.perf_counter()
    raw = _predict_batch(
        texts, model, tokenizer, max_length, device,
        batch_size=args.batch_size, top_k=args.top_k,
        is_multi_label=is_multi_label,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    decoded = _decode_results(raw, label_encoder, is_multi_label=is_multi_label)

    # ── Build output DataFrame ────────────────────────────────────────────────
    df_out = df.copy()
    df_out["predicted_label"] = [r["predicted_label"] for r in decoded]
    if is_multi_label:
        # Multi-label: no single confidence; store per-label scores as a string
        df_out["predicted_labels"] = [r.get("predicted_label", "") for r in decoded]
        df_out["label_confidences"] = [
            "; ".join(
                f"{t['label']}={t['confidence']*100:.1f}%"
                for t in r.get("top_k", [])
            )
            for r in decoded
        ]
    else:
        df_out["confidence"] = [round(r["confidence"] * 100, 2) for r in decoded]

    if not is_multi_label and args.top_k > 1:
        for k in range(args.top_k):
            df_out[f"top{k+1}_label"] = [
                r["top_k"][k].get("label", r["top_k"][k].get("class_id"))
                if k < len(r["top_k"]) else ""
                for r in decoded
            ]
            df_out[f"top{k+1}_confidence"] = [
                round(r["top_k"][k]["confidence"] * 100, 2)
                if k < len(r["top_k"]) else 0.0
                for r in decoded
            ]

    avg_ms = elapsed_ms / max(len(texts), 1)
    print(f"\n  Done. Avg latency: {avg_ms:.2f} ms/sample  (total: {elapsed_ms/1000:.1f}s)")

    # ── Display or save ───────────────────────────────────────────────────────
    if args.output:
        out_path = Path(args.output)
        df_out.to_csv(str(out_path), index=False)
        print(f"\n  Predictions saved to: {out_path}")
        # Quick summary
        print("\n  Label distribution in predictions:")
        vc = df_out["predicted_label"].value_counts()
        for label, count in vc.items():
            pct = 100 * count / len(df_out)
            print(f"    {label:<30} {count:>6}  ({pct:.1f}%)")
    else:
        # Print a preview (first 20 rows)
        display_cols = [text_col, "predicted_label", "confidence"] if not is_multi_label else [text_col, "predicted_label", "label_confidences"]
        n_preview = min(20, len(df_out))
        print(f"\n  Preview (first {n_preview} rows):")
        print("-" * 70)
        with pd.option_context(
            "display.max_columns", None,
            "display.width", 120,
            "display.max_colwidth", 60,
        ):
            print(df_out[display_cols].head(n_preview).to_string(index=False))

        if len(df_out) > n_preview:
            print(f"\n  ... {len(df_out) - n_preview} more rows not shown. Use --output to save all.")

        print("\n  Label distribution:")
        vc = df_out["predicted_label"].value_counts()
        for label, count in vc.items():
            pct = 100 * count / len(df_out)
            print(f"    {label:<30} {count:>6}  ({pct:.1f}%)")

    print("\n" + "=" * 70 + "\n")
    sys.exit(0)


if __name__ == "__main__":
    main()
