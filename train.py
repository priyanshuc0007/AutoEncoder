"""
AutoLLM — CLI Training Script
==============================
Fine-tunes multiple encoder models on any text classification CSV/XLSX,
compares their performance and latency, and saves the best model.

Usage examples
--------------
  # Minimal — auto-detect text column
  python train.py --file data.csv --label sentiment

  # Explicit text column
  python train.py --file data.xlsx --label category --text review_body

  # Merge multiple text columns into one
  python train.py --file data.csv --label label --text-columns title description

  # Pin specific models
  python train.py --file data.csv --label label --models distilbert-base-uncased prajjwal1/bert-tiny

  # Full options
  python train.py --file data.csv --label label --optuna --optuna-trials 15 --cv --cv-folds 5

Run `python train.py --help` for the full argument reference.
"""

import argparse
import sys
import os
import logging
from pathlib import Path

# Console logging configured once for the CLI entry point.
# Per-experiment FILE logging is handled inside AutoLLMPipeline.run() and
# writes to experiments/<run>/pipeline.log automatically.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# ──────────────────────────────────────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train.py",
        description="AutoLLM — fine-tune & compare encoder models for text classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --file reviews.csv --label sentiment
  python train.py --file emails.xlsx --label category --text body
  python train.py --file data.csv --label label --text-columns title body --optuna
  python train.py --file data.csv --label label --models distilbert-base-uncased prajjwal1/bert-tiny

After training, use predict.py to run inference:
  python predict.py --experiment experiments/<run> --text "This is amazing!"
  python predict.py --experiment experiments/<run> --file new_data.csv
        """,
    )

    # ── Required ──
    req = p.add_argument_group("required arguments")
    req.add_argument(
        "--file", "-f",
        required=True,
        metavar="PATH",
        help="Path to CSV or XLSX file containing your dataset",
    )
    req.add_argument(
        "--label", "-l",
        required=True,
        metavar="COLUMN",
        help="Name of the column that contains class labels",
    )

    # ── Text column(s) ──
    txt = p.add_argument_group("text column options (mutually exclusive)")
    txt_grp = txt.add_mutually_exclusive_group()
    txt_grp.add_argument(
        "--text", "-t",
        metavar="COLUMN",
        default=None,
        help="Single text column to use. Omit to auto-detect the longest text column.",
    )
    txt_grp.add_argument(
        "--text-columns",
        nargs="+",
        metavar="COLUMN",
        default=None,
        help="Two or more text columns to merge (space-separated). "
             "Example: --text-columns title body description",
    )

    # ── Model selection ──
    mod = p.add_argument_group("model options")
    mod.add_argument(
        "--models",
        nargs="+",
        metavar="MODEL",
        default=None,
        help=(
            "HuggingFace model IDs to train (space-separated). "
            "Omit to use auto-selection based on dataset size. "
            "Available: prajjwal1/bert-tiny  prajjwal1/bert-mini  "
            "google/mobilebert-uncased  distilbert-base-uncased  bert-base-uncased"
        ),
    )

    # ── Hyperparameter search ──
    opt = p.add_argument_group("Optuna hyperparameter search")
    opt.add_argument(
        "--optuna",
        action="store_true",
        help="Run Bayesian hyperparameter search (learning_rate + weight_decay + epochs). "
             "Adds ~N× training time per model where N = --optuna-trials.",
    )
    opt.add_argument(
        "--optuna-trials",
        type=int,
        default=10,
        metavar="N",
        help="Number of Optuna trials per model (clamped 3–20, default: 10)",
    )

    # ── Cross-validation ──
    cv = p.add_argument_group("cross-validation")
    cv.add_argument(
        "--cv",
        action="store_true",
        help="Run K-fold cross-validation on the best model after training.",
    )
    cv.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        metavar="K",
        help="Number of CV folds (default: 5)",
    )

    # ── Output ──
    out = p.add_argument_group("output")
    out.add_argument(
        "--output-dir",
        default="experiments",
        metavar="DIR",
        help="Directory to save experiments (default: experiments/)",
    )
    out.add_argument(
        "--name",
        default=None,
        metavar="NAME",
        help="Experiment name / subfolder. Defaults to a timestamp.",
    )

    return p


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _print_banner():
    print("\n" + "=" * 70)
    print("  AutoLLM — Automated Text Classification Fine-Tuning")
    print("=" * 70 + "\n")


def _print_summary(result: dict):
    """Pretty-print the pipeline result after a successful run."""
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)

    m = result["best_model_metrics"]
    print(f"\n  Best Model : {result['best_model_name']}")
    print(f"  F1 Score   : {m['f1_score']:.4f}")
    print(f"  Accuracy   : {m['accuracy']:.4f}")
    print(f"  Latency    : {m['latency_ms']:.2f} ms  (single-sample, tokenization included)")

    print(f"\n  Model path      : {result['best_model_path']}")
    print(f"  Experiment dir  : {result['experiment_dir']}")
    print(f"  Report          : {result['experiment_dir']}/best_model_report.txt")

    df_cmp = result.get("comparison_df")
    if df_cmp is not None and not df_cmp.empty:
        print("\n" + "-" * 70)
        print("  ALL MODELS COMPARED")
        print("-" * 70)
        # Widen display so it doesn't wrap
        import pandas as pd
        with pd.option_context(
            "display.max_columns", None,
            "display.width", 120,
            "display.float_format", "{:.4f}".format,
        ):
            print(df_cmp.to_string(index=False))

    cv = result.get("cv_results")
    if cv:
        summary = cv.get("summary", {})
        f1_stats  = summary.get("f1", {})
        acc_stats = summary.get("accuracy", {})
        print("\n" + "-" * 70)
        print("  CROSS-VALIDATION RESULTS")
        print("-" * 70)
        print(f"  F1    : {f1_stats.get('mean', 0):.4f}  \u00b1  {f1_stats.get('std', 0):.4f}")
        print(f"  Acc   : {acc_stats.get('mean', 0):.4f}  \u00b1  {acc_stats.get('std', 0):.4f}")

    print("\n" + "-" * 70)
    print("  NEXT STEP — run inference:")
    print(f"    python predict.py --experiment {result['experiment_dir']} --text \"Your text here\"")
    print(f"    python predict.py --experiment {result['experiment_dir']} --file new_data.csv")
    print("=" * 70 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = build_parser()
    args = parser.parse_args()

    # ── Pre-flight checks ──────────────────────────────────────────────────────
    file_path = Path(args.file)
    if not file_path.exists():
        parser.error(f"File not found: {args.file}")

    suffix = file_path.suffix.lower()
    if suffix not in (".csv", ".xlsx", ".xls"):
        parser.error(
            f"Unsupported file type '{suffix}'. "
            f"Provide a CSV (.csv) or Excel (.xlsx / .xls) file."
        )

    if args.optuna_trials < 1:
        parser.error("--optuna-trials must be at least 1")
    if args.cv_folds < 2:
        parser.error("--cv-folds must be at least 2")

    _print_banner()

    # ── Print run config ───────────────────────────────────────────────────────
    print("  Run configuration")
    print("  " + "-" * 40)
    print(f"  File          : {args.file}")
    print(f"  Label column  : {args.label}")
    if args.text:
        print(f"  Text column   : {args.text}")
    elif args.text_columns:
        print(f"  Text columns  : {', '.join(args.text_columns)}  (will be merged)")
    else:
        print(f"  Text column   : auto-detect")
    if args.models:
        print(f"  Models        : {', '.join(args.models)}")
    else:
        print(f"  Models        : auto-select based on dataset size")
    print(f"  Optuna search : {'yes  (' + str(args.optuna_trials) + ' trials)' if args.optuna else 'no'}")
    print(f"  Cross-val     : {'yes  (' + str(args.cv_folds) + ' folds)' if args.cv else 'no'}")
    print(f"  Output dir    : {args.output_dir}")
    print()

    # ── Run pipeline ───────────────────────────────────────────────────────────
    from automl import AutoLLMPipeline

    pipeline = AutoLLMPipeline(output_dir=args.output_dir)

    result = pipeline.run(
        csv_path=str(file_path),
        label_column=args.label,
        text_column=args.text,
        text_columns=args.text_columns,
        experiment_name=args.name,
        use_cv=args.cv,
        cv_folds=args.cv_folds,
        use_optuna=args.optuna,
        optuna_trials=args.optuna_trials,
        model_names=args.models,
    )

    # ── Report outcome ─────────────────────────────────────────────────────────
    if result["status"] == "success":
        _print_summary(result)
        sys.exit(0)
    else:
        print("\n" + "=" * 70)
        print("  TRAINING FAILED")
        print("=" * 70)
        print(f"\n  Error: {result.get('error', 'Unknown error')}")
        print("\n  Troubleshooting tips:")
        print("  - Make sure --label column name is spelled correctly")
        print("  - Your CSV needs at least 100 rows and 2 distinct label values")
        print("  - Make sure each class has at least 2 samples")
        print("=" * 70 + "\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
