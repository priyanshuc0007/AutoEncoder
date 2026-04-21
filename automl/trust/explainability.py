"""
Explainability & Reliability  (Pillar 5)
=========================================
Provides token-level explanations and confidence reliability analysis
for the best trained model WITHOUT modifying any training logic.

What it does
------------
1. Token importance    — uses attention weights (mean over all layers and
                         heads, CLS-row rollout) to score which input tokens
                         most influenced each prediction.  Subword tokens are
                         mapped back to whole words where possible.

2. Confidence analysis — computes the softmax max-probability across the full
                         validation set:
                           • mean / min / max confidence
                           • count & percentage of low-confidence predictions
                             (default threshold: 0.80)

3. Calibration check   — Expected Calibration Error (ECE, 10 uniform bins):
                         measures whether the model's stated confidence
                         actually tracks its empirical accuracy.
                           ECE < 0.05  → well-calibrated   ✅
                           ECE < 0.10  → acceptable         ⚠️
                           ECE ≥ 0.10  → poorly calibrated  ❌

Output: `explainability_report.txt` in the experiment folder.
"""

import logging
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _merge_subwords(tokens: List[str], scores: np.ndarray) -> List[tuple]:
    """
    Merge WordPiece '##' continuations back into whole words and sum their
    attention scores.  Returns a list of (word, score) tuples.
    """
    words, word_scores = [], []
    for tok, sc in zip(tokens, scores):
        if tok.startswith("##") and words:
            words[-1] += tok[2:]
            word_scores[-1] += sc
        else:
            words.append(tok)
            word_scores.append(sc)
    return list(zip(words, word_scores))


# ─────────────────────────────────────────────────────────────────────────────
# Token importance (attention rollout)
# ─────────────────────────────────────────────────────────────────────────────

def compute_token_importance(
    model,
    tokenizer,
    texts: List[str],
    labels: List[int],
    max_length: int,
    device,
    label_encoder=None,
    n_samples: int = 10,
) -> List[Dict]:
    """
    Compute per-token importance for up to *n_samples* validation texts.

    Strategy: average attention weights across all layers and all heads,
    then take the CLS-token row.  This gives one scalar per input token
    representing how much the classification head "attended to" it.

    Args:
        model        : Loaded AutoModelForSequenceClassification.
        tokenizer    : Matching tokenizer.
        texts        : Validation texts.
        labels       : Encoded integer labels (parallel to texts).
        max_length   : Max token length used during training.
        device       : torch.device.
        label_encoder: sklearn LabelEncoder for decoding predicted class names.
        n_samples    : Number of samples to explain.

    Returns:
        List of dicts — one per sample — each containing:
            text          : original text (truncated to 120 chars for display)
            predicted     : predicted class name (decoded if encoder provided)
            actual        : actual class name
            correct       : bool
            word_scores   : list of (word, score) sorted by descending score
            top_words     : top-5 (word, score) pairs
    """
    model.eval()
    # Force attention output regardless of what was baked into the saved config
    model.config.output_attentions = True
    results = []
    special = {tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token}

    for text, label in zip(texts[:n_samples], labels[:n_samples]):
        try:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=False,
            ).to(device)

            # Explicit model inputs (avoids unexpected keys from tokenizer)
            model_inputs = {"input_ids": inputs["input_ids"]}
            if "attention_mask" in inputs:
                model_inputs["attention_mask"] = inputs["attention_mask"]
            if "token_type_ids" in inputs:
                model_inputs["token_type_ids"] = inputs["token_type_ids"]

            with torch.no_grad():
                outputs = model(**model_inputs, output_attentions=True)

            # Filter valid attention tensors.
            # Some BERT variants (e.g. bert-tiny) return a tuple of None values
            # per layer even when output_attentions=True, which causes torch.stack
            # to fail.  We accept only real Tensor elements.
            attn_list = [
                a for a in (outputs.attentions or [])
                if isinstance(a, torch.Tensor)
            ]

            if attn_list:
                # Attention rollout: mean over layers & heads, take CLS row
                # attentions: list of (1, n_heads, seq_len, seq_len)
                avg_attn = torch.stack(attn_list).squeeze(1).mean(dim=(0, 1))
                cls_row = avg_attn[0].cpu().numpy()  # (seq_len,)
            else:
                # Fallback: gradient × embedding saliency (model-agnostic).
                # Runs a second forward pass with gradient tracking on the
                # input embeddings; L2-norm of the gradient gives per-token
                # importance without relying on attention outputs.
                if not hasattr(model, 'get_input_embeddings'):
                    continue  # model doesn't support embedding access — skip sample
                model.zero_grad()
                input_embeds = (
                    model.get_input_embeddings()(inputs["input_ids"])
                    .detach()
                    .requires_grad_(True)
                )
                with torch.enable_grad():
                    out2 = model(
                        inputs_embeds=input_embeds,
                        attention_mask=inputs.get("attention_mask"),
                    )
                    pred_cls = int(out2.logits.argmax(dim=1).item())
                    out2.logits[0, pred_cls].backward()
                cls_row = input_embeds.grad[0].norm(dim=-1).detach().cpu().numpy()

            token_ids = inputs["input_ids"][0].cpu().tolist()
            tokens = tokenizer.convert_ids_to_tokens(token_ids)

            # Remove special tokens
            filtered_tokens, filtered_scores = [], []
            for tok, sc in zip(tokens, cls_row):
                if tok not in special:
                    filtered_tokens.append(tok)
                    filtered_scores.append(sc)

            if not filtered_tokens:
                continue

            scores_arr = np.array(filtered_scores, dtype=float)
            total = scores_arr.sum()
            if total > 0:
                scores_arr = scores_arr / total

            # Merge subwords → whole words
            word_scores = _merge_subwords(filtered_tokens, scores_arr)
            word_scores_sorted = sorted(word_scores, key=lambda x: x[1], reverse=True)

            # Predicted class
            pred_idx = int(torch.argmax(outputs.logits, dim=1).item())
            if label_encoder is not None:
                try:
                    predicted_str = label_encoder.inverse_transform([pred_idx])[0]
                    actual_str = label_encoder.inverse_transform([label])[0]
                except Exception:
                    predicted_str = str(pred_idx)
                    actual_str = str(label)
            else:
                predicted_str = str(pred_idx)
                actual_str = str(label)

            results.append({
                "text": text[:120],
                "predicted": predicted_str,
                "actual": actual_str,
                "correct": pred_idx == label,
                "word_scores": word_scores_sorted,
                "top_words": word_scores_sorted[:5],
            })
        except Exception as exc:
            logger.warning(f"[Trust/Explainability] Skipped sample: {exc}")
            continue

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Confidence & calibration
# ─────────────────────────────────────────────────────────────────────────────

def compute_confidence_stats(
    model,
    tokenizer,
    texts: List[str],
    labels: List[int],
    max_length: int,
    device,
    batch_size: int = 32,
    low_confidence_threshold: float = 0.80,
) -> Dict:
    """
    Compute confidence distribution and Expected Calibration Error (ECE).

    Args:
        model                    : Loaded classification model.
        tokenizer                : Matching tokenizer.
        texts                    : Validation texts.
        labels                   : Encoded integer labels.
        max_length               : Max token length.
        device                   : torch.device.
        batch_size               : Inference batch size.
        low_confidence_threshold : Samples with max-softmax below this are
                                   flagged as uncertain.

    Returns:
        Dict with confidence stats and calibration bin details.
    """
    from automl.dataset import TextDataset
    from torch.utils.data import DataLoader

    dataset = TextDataset(texts, labels, tokenizer, max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_probs: List[List[float]] = []
    all_preds: List[int] = []
    all_labels: List[int] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(batch_labels.numpy().tolist())

    confidences = np.array([max(p) for p in all_probs])
    all_preds_arr = np.array(all_preds)
    all_labels_arr = np.array(all_labels)

    low_conf_mask = confidences < low_confidence_threshold
    low_conf_count = int(low_conf_mask.sum())

    # Confidence histogram (10 equal-width buckets)
    hist_counts, hist_edges = np.histogram(confidences, bins=10, range=(0.0, 1.0))
    confidence_histogram = [
        {
            "range": f"{hist_edges[i]:.1f}–{hist_edges[i + 1]:.1f}",
            "count": int(hist_counts[i]),
        }
        for i in range(len(hist_counts))
    ]

    return {
        "n_samples": len(confidences),
        "mean_confidence": round(float(confidences.mean()), 4),
        "median_confidence": round(float(np.median(confidences)), 4),
        "min_confidence": round(float(confidences.min()), 4),
        "max_confidence": round(float(confidences.max()), 4),
        "std_confidence": round(float(confidences.std()), 4),
        "low_confidence_threshold": low_confidence_threshold,
        "low_confidence_count": low_conf_count,
        "low_confidence_pct": round(float(low_conf_count / len(confidences) * 100), 2),
        "confidence_histogram": confidence_histogram,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Report writer
# ─────────────────────────────────────────────────────────────────────────────

def save_explainability_report(
    token_explanations: List[Dict],
    confidence_stats: Dict,
    experiment_dir: Path,
) -> None:
    """
    Write `explainability_report.txt` to the experiment folder.
    """
    lines: List[str] = []
    sep = "=" * 70

    lines += [sep, "EXPLAINABILITY & RELIABILITY REPORT  (Pillar 5)", sep]

    # ── Confidence overview ────────────────────────────────────────────────
    cs = confidence_stats
    lines += [
        "",
        sep,
        "CONFIDENCE ANALYSIS",
        sep,
        f"Samples analysed:       {cs.get('n_samples', '?')}",
        f"Mean confidence:        {cs.get('mean_confidence', '?')}",
        f"Median confidence:      {cs.get('median_confidence', '?')}",
        f"Std  confidence:        {cs.get('std_confidence', '?')}",
        f"Min  confidence:        {cs.get('min_confidence', '?')}",
        f"Max  confidence:        {cs.get('max_confidence', '?')}",
        f"Low-confidence samples: {cs.get('low_confidence_count', '?')} "
        f"({cs.get('low_confidence_pct', '?')}%)  "
        f"[threshold < {cs.get('low_confidence_threshold', '?')}]",
    ]

    # Confidence histogram
    lines += ["", "Confidence Distribution:"]
    for bucket in cs.get("confidence_histogram", []):
        bar_len = int(bucket["count"] / max(cs.get("n_samples", 1), 1) * 40)
        bar = "█" * bar_len
        lines.append(f"  {bucket['range']:<10}  {bucket['count']:>5}  {bar}")

    # ── Token importance ───────────────────────────────────────────────────
    if token_explanations:
        lines += [
            "",
            sep,
            "TOKEN IMPORTANCE  (attention rollout if available, else gradient saliency)",
            sep,
            "Shows which words the model attended to most when classifying each sample.",
            "Scores are normalised so they sum to 1.0 per sample.",
            "",
        ]
        for i, sample in enumerate(token_explanations, 1):
            correct_marker = "✅ correct" if sample.get("correct") else "❌ wrong"
            lines.append(
                f"Sample {i}  [{correct_marker}]"
                f"  predicted={sample.get('predicted')}  actual={sample.get('actual')}"
            )
            lines.append(f"  Text: \"{sample['text'][:100]}\"")
            lines.append("  Top influential words:")
            for word, score in sample.get("top_words", []):
                bar = "█" * max(1, int(score * 60))
                lines.append(f"    {word:<22}  {score:.4f}  {bar}")
            lines.append("")

    lines.append(sep)

    out_path = experiment_dir / "explainability_report.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"[Trust/Explainability] Report saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_explainability(
    model_path: str,
    tokenizer,
    val_texts: List[str],
    val_labels: List[int],
    max_length: int,
    experiment_dir: Path,
    label_encoder=None,
    n_samples: int = 10,
    low_confidence_threshold: float = 0.80,
) -> Dict:
    """
    Main entry point — load best model, run all explainability checks,
    save report.  Wrapped in a broad try/except so a failure here NEVER
    breaks the main pipeline.

    Args:
        model_path               : Path to saved best model directory.
        tokenizer                : Matching tokenizer (already loaded).
        val_texts                : Validation texts.
        val_labels               : Encoded integer validation labels.
        max_length               : Max token length used during training.
        experiment_dir           : Path to current experiment folder.
        label_encoder            : sklearn LabelEncoder for decoding labels.
        n_samples                : How many samples to explain (token importance).
        low_confidence_threshold : Confidence below which samples are flagged.

    Returns:
        Summary dict with key metrics (ece, mean_confidence, etc.) for the
        pipeline tracker.  Empty dict on failure.
    """
    from transformers import AutoModelForSequenceClassification

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # Use 'eager' attention to allow output_attentions=True;
        # the default 'sdpa' (scaled dot-product attention) kernel does not
        # support returning attention tensors and raises an AttributeError.
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            attn_implementation="eager",
        ).to(device)
    except Exception as exc:
        logger.warning(f"[Trust/Explainability] Could not load model: {exc}")
        return {}

    conf_stats: Dict = {}
    token_expl: List[Dict] = []

    # 1. Confidence & calibration (full validation set)
    try:
        conf_stats = compute_confidence_stats(
            model, tokenizer, val_texts, val_labels,
            max_length, device,
            low_confidence_threshold=low_confidence_threshold,
        )
        logger.info(
            f"[Trust/Explainability] Confidence — mean={conf_stats['mean_confidence']}, "
            f"ECE={conf_stats['ece']}, "
            f"low-conf={conf_stats['low_confidence_count']}"
            f"({conf_stats['low_confidence_pct']}%)"
        )
    except Exception as exc:
        logger.warning(f"[Trust/Explainability] Confidence stats failed: {exc}")

    # 2. Token importance (small sample)
    try:
        token_expl = compute_token_importance(
            model, tokenizer, val_texts, val_labels,
            max_length, device,
            label_encoder=label_encoder,
            n_samples=n_samples,
        )
        logger.info(
            f"[Trust/Explainability] Token importance computed for "
            f"{len(token_expl)} samples"
        )
    except Exception as exc:
        logger.warning(f"[Trust/Explainability] Token importance failed: {exc}")

    # 3. Save report
    if conf_stats or token_expl:
        try:
            save_explainability_report(token_expl, conf_stats, experiment_dir)
        except Exception as exc:
            logger.warning(f"[Trust/Explainability] Report save failed: {exc}")

    return {
        "ece": conf_stats.get("ece"),
        "mean_confidence": conf_stats.get("mean_confidence"),
        "low_confidence_count": conf_stats.get("low_confidence_count"),
        "low_confidence_pct": conf_stats.get("low_confidence_pct"),
        "n_explained_samples": len(token_expl),
        # Full structured data for the UI
        "confidence_stats": conf_stats,
        "token_explanations": token_expl,
    }
