"""
Streamlit Web Application for AutoML Pipeline
Interactive UI for testing and running AutoML
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import os
from datetime import datetime
import time

# Import AutoLLM components
from automl import AutoLLMPipeline

# Page config
st.set_page_config(
    page_title="🤖 AutoLLM Text Classification",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 1.1em;
        padding: 0.5rem 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'pipeline_results' not in st.session_state:
    st.session_state.pipeline_results = None
if 'experiment_history' not in st.session_state:
    st.session_state.experiment_history = []
if 'current_experiment' not in st.session_state:
    st.session_state.current_experiment = None


def save_experiment_to_history(result):
    """Save experiment to history"""
    st.session_state.experiment_history.append({
        'timestamp': datetime.now(),
        'experiment_name': result.get('experiment_name'),
        'best_model': result.get('best_model_name'),
        'f1_score': result.get('best_model_metrics', {}).get('f1_score'),
        'accuracy': result.get('best_model_metrics', {}).get('accuracy'),
        'result': result
    })


def display_data_preview(df, label_col, text_col):
    """Display data preview with statistics"""
    st.subheader("📋 Data Preview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Samples", len(df))
    with col2:
        # For multi-label data (comma-separated), count unique individual labels
        _sample = df[label_col].dropna().astype(str)
        _is_multi = _sample.str.contains(',').any()
        if _is_multi:
            _unique_labels = set(lbl.strip() for row in _sample for lbl in row.split(','))
            _class_count = len(_unique_labels)
        else:
            _class_count = df[label_col].nunique()
        st.metric("Unique Classes", _class_count)
    with col3:
        st.metric("Avg Text Length", int(df[text_col].astype(str).str.len().mean()))
    with col4:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    st.write("First few rows:")
    st.dataframe(df.head(10), width='stretch')
    
    # Class distribution
    st.subheader("📊 Class Distribution")
    class_dist = df[label_col].value_counts()
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.bar_chart(class_dist)
    
    with col2:
        st.write("Distribution Details:")
        for cls, count in class_dist.items():
            pct = 100 * count / len(df)
            st.write(f"- **{cls}**: {count} samples ({pct:.1f}%)")


def display_pipeline_results(result):
    """Display pipeline results"""
    st.success("✅ Pipeline Completed Successfully!")

    # ── Best model headline ──────────────────────────────────────────────────
    st.subheader("🏆 Best Model Results")
    metrics = result.get('best_model_metrics', {})
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Model", result.get('best_model_name', 'Unknown')[:30])
    with col2:
        st.metric("F1 Score", f"{metrics.get('f1_score', 0):.4f}")
    with col3:
        st.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
    with col4:
        st.metric("Single-Sample Latency (ms)", f"{metrics.get('single_sample_latency_ms', metrics.get('latency_ms', 0)):.2f}")

    # ── Full model comparison ────────────────────────────────────────────────
    st.subheader("📊 Model Comparison")
    comparison_df = result.get('comparison_df')

    if comparison_df is not None and not comparison_df.empty:
        n_models = len(comparison_df)
        best_name = result.get('best_model_name', '')

        # ── Styled table ────────────────────────────────────────────────────
        def _highlight_best(row):
            """Highlight the winning model row in green."""
            is_best = str(row.get('Model', '')) == str(best_name)
            bg = 'background-color: #1a4a2a; color: #06D6A0; font-weight: bold' if is_best else ''
            return [bg] * len(row)

        display_cols = [c for c in [
            'Model', 'F1 Score', 'Accuracy', 'Precision', 'Recall',
            'Batch Latency (ms)', 'Single-Sample Latency (ms)', 'Composite Score'
        ] if c in comparison_df.columns]

        styled = (
            comparison_df[display_cols]
            .style
            .apply(_highlight_best, axis=1)
            .format({
                'F1 Score': '{:.4f}',
                'Accuracy': '{:.4f}',
                'Precision': '{:.4f}',
                'Recall': '{:.4f}',
                'Batch Latency (ms)': '{:.2f}',
                'Single-Sample Latency (ms)': '{:.2f}',
                'Composite Score': '{:.4f}',
            })
        )
        st.dataframe(styled, width='stretch')

        # ── Charts (only useful when more than 1 model) ──────────────────────
        if n_models > 1:
            st.markdown("#### Visual Comparison")

            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                # F1 / Accuracy / Composite bar chart
                bar_data = comparison_df.set_index('Model')[
                    [c for c in ['F1 Score', 'Accuracy', 'Composite Score']
                     if c in comparison_df.columns]
                ]
                st.markdown("**Performance Metrics**")
                st.bar_chart(bar_data)

            with chart_col2:
                # Latency bar chart (lower = better)
                lat_col = next((c for c in ['Single-Sample Latency (ms)', 'Batch Latency (ms)'] if c in comparison_df.columns), None)
                if lat_col:
                    lat_data = comparison_df.set_index('Model')[[lat_col]]
                    st.markdown(f"**{lat_col} — lower is better**")
                    st.bar_chart(lat_data)

            # Detailed per-model cards
            st.markdown("#### Per-Model Detail")
            card_cols = st.columns(n_models)
            for i, (_, row) in enumerate(comparison_df.iterrows()):
                model_name = str(row.get('Model', 'Model'))
                is_best = model_name == str(best_name)
                with card_cols[i]:
                    header = f"🏆 {model_name}" if is_best else f"🔹 {model_name}"
                    st.markdown(f"**{header}**")
                    if is_best:
                        st.success("Best model")
                    for col_name in ['F1 Score', 'Accuracy', 'Precision', 'Recall',
                                     'Batch Latency (ms)', 'Single-Sample Latency (ms)',
                                     'Composite Score']:
                        if col_name in row:
                            val = row[col_name]
                            try:
                                is_latency = 'Latency' in col_name
                                fmt = f"{val:.2f} ms" if is_latency else f"{val:.4f}"
                            except (TypeError, ValueError):
                                fmt = str(val) if val is not None else 'N/A'
                            st.write(f"- **{col_name}**: {fmt}")
        else:
            st.info(
                "Only one model was trained in this run. "
                "Select multiple models in Step 2.5 and re-run the pipeline to get a full comparison."
            )
    else:
        st.warning("No comparison data available.")

    # ── Experiment info ──────────────────────────────────────────────────────
    st.subheader("ℹ️ Experiment Information")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Experiment Name**: {result.get('experiment_name')}")
        st.write(f"**Model Path**: `{result.get('best_model_path')}`")
    with col2:
        st.write(f"**Output Directory**: `{result.get('experiment_dir')}`")
        if os.path.exists(result.get('experiment_dir', '')):
            st.info("🔍 Model files are saved to the experiment directory")

    # ── Data intelligence report ─────────────────────────────────────────────
    st.subheader("🧠 Data Intelligence Report")
    analysis = result.get('data_analysis', {})

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Task Information**")
        task_info = analysis.get('task_info', {})
        st.write(f"- Task Type: **{task_info.get('task_type', 'N/A').upper()}**")
        st.write(f"- Number of Classes: **{task_info.get('num_classes', 'N/A')}**")

        st.write("\n**Imbalance Information**")
        imbal = analysis.get('imbalance_info', {})
        imbalance_ratio = imbal.get('imbalance_ratio')
        st.write(f"- Imbalance Ratio: **{imbalance_ratio:.2f}**" if isinstance(imbalance_ratio, (int, float)) else "- Imbalance Ratio: **N/A**")
        st.write(f"- Use Class Weights: **{'✓ YES' if imbal.get('use_class_weights') else '✗ NO'}**")
        st.write(f"- Use Focal Loss: **{'✓ YES' if imbal.get('use_focal_loss') else '✗ NO'}**")

    with col2:
        st.write("**Text Information**")
        text_info = analysis.get('text_info', {})
        avg_length = text_info.get('avg_length')
        st.write(f"- Average Length: **{avg_length:.0f}**" if isinstance(avg_length, (int, float)) else "- Average Length: **N/A**")
        st.write(f"- Max Length (95%): **{text_info.get('p95_length', 'N/A')}**")

        st.write("\n**Training Configuration**")
        config = analysis.get('training_config', {})
        st.write(f"- Batch Size: **{config.get('batch_size', 'N/A')}**")
        st.write(f"- Learning Rate: **{config.get('learning_rate', 'N/A')}**")

    # ── Explainability & Reliability ─────────────────────────────────────────
    st.subheader("🔍 Explainability & Reliability")

    expl = result.get('explainability') or {}
    conf_stats = expl.get('confidence_stats') or {}
    token_expl = expl.get('token_explanations') or []

    # Fallback: read the .txt report if structured data is missing (e.g. older run)
    if not conf_stats and not token_expl:
        exp_dir = result.get('experiment_dir', '')
        report_path = os.path.join(exp_dir, 'explainability_report.txt')
        if os.path.exists(report_path):
            with open(report_path, 'r', encoding='utf-8') as _f:
                st.text(_f.read())
        else:
            st.info("No explainability data available. Re-run the pipeline to generate it.")
    else:
        # ── Confidence headline metrics ──────────────────────────────────────
        mean_conf = conf_stats.get('mean_confidence')
        low_count = conf_stats.get('low_confidence_count', 0)
        low_pct = conf_stats.get('low_confidence_pct', 0.0)
        threshold = conf_stats.get('low_confidence_threshold', 0.80)

        ec1, ec2 = st.columns(2)
        with ec1:
            st.metric(
                "Mean Confidence",
                f"{mean_conf:.4f}" if isinstance(mean_conf, float) else "N/A",
                help=(
                    "Average softmax probability the model assigned to its "
                    "chosen class across the whole validation set. "
                    "0.73 means the model said 'I'm 73% sure' on average."
                ),
            )
        with ec2:
            st.metric(
                f"Low-Conf Samples (<{threshold})",
                f"{low_count} ({low_pct:.1f}%)",
                help=(
                    f"Samples where the model's confidence was below {threshold}. "
                    "High counts here don't mean wrong predictions — just that "
                    "the model is uncertain. Check Token Importance below for these cases."
                ),
            )

        # ── Confidence distribution bar chart ────────────────────────────────
        hist = conf_stats.get('confidence_histogram', [])
        if hist:
            hist_df = pd.DataFrame(hist).set_index('range').rename(
                columns={'count': 'Samples'}
            )
            st.markdown(
                "**Confidence Distribution** — how many samples fell into each probability bucket"
            )
            st.caption(
                "Ideal: most samples in 0.9–1.0. "
                "All bars in 0.6–0.8 = under-confident (correct but not bold). "
                "Bars spread across 0.5–0.7 on wrong predictions = over-confident."
            )
            st.bar_chart(hist_df)

        # ── Token importance samples ─────────────────────────────────────────
        if token_expl:
            st.markdown("**Token Importance** — which words most influenced each prediction")
            st.caption(
                "Scores use attention rollout (1 forward pass) with gradient "
                "saliency fallback. Both are fast and model-agnostic."
            )
            for i, sample in enumerate(token_expl):
                correct = sample.get('correct', False)
                badge = "✅ Correct" if correct else "❌ Wrong"
                pred = sample.get('predicted', '?')
                actual = sample.get('actual', '?')
                with st.expander(f"Sample {i+1} — predicted: **{pred}** | actual: **{actual}** | {badge}"):
                    st.write(f"**Text:** {sample.get('text', '')}")
                    top_words = sample.get('top_words', [])
                    if top_words:
                        # Normalise scores to [0,1] for progress bars
                        max_score = max(sc for _, sc in top_words) or 1.0
                        for word, score in top_words:
                            col_w, col_b = st.columns([1, 4])
                            with col_w:
                                st.write(f"`{word}`")
                            with col_b:
                                st.progress(min(1.0, float(score / max_score)))
                    all_words = sample.get('word_scores', [])
                    if len(all_words) > len(top_words):
                        with st.expander("See all word scores"):
                            max_all = max(sc for _, sc in all_words) or 1.0
                            for word, score in all_words:
                                col_w, col_b = st.columns([1, 4])
                                with col_w:
                                    st.write(f"`{word}`")
                                with col_b:
                                    st.progress(min(1.0, float(score / max_all)))

        # ── Raw report fallback ──────────────────────────────────────────────
        exp_dir = result.get('experiment_dir', '')
        report_path = os.path.join(exp_dir, 'explainability_report.txt')
        if os.path.exists(report_path):
            with st.expander("📄 View Full Explainability Report (text)"):
                with open(report_path, 'r', encoding='utf-8') as _f:
                    st.text(_f.read())

    # ── Cross-Validation Results ─────────────────────────────────────────────
    cv = result.get('cv_results')
    if cv:
        st.subheader("🔁 Cross-Validation Results")
        summary = cv.get('summary', {})
        n_splits = cv.get('n_splits', '?')
        n_ok = cv.get('n_successful_folds', 0)

        if n_ok < n_splits:
            st.warning(f"{n_splits - n_ok} fold(s) failed — summary is based on {n_ok} successful fold(s).")

        if summary:
            f1_s = summary.get('f1', {})
            acc_s = summary.get('accuracy', {})
            pre_s = summary.get('precision', {})
            rec_s = summary.get('recall', {})

            cv1, cv2, cv3, cv4 = st.columns(4)
            with cv1:
                st.metric(
                    "CV F1 (mean ± std)",
                    f"{f1_s.get('mean', 0):.4f}",
                    delta=f"± {f1_s.get('std', 0):.4f}",
                    delta_color="off",
                    help=(
                        "Mean F1 across all folds. Low std = stable model. "
                        "High std = model is sensitive to which samples land in the test fold."
                    ),
                )
            with cv2:
                st.metric(
                    "CV Accuracy (mean)",
                    f"{acc_s.get('mean', 0):.4f}",
                    delta=f"± {acc_s.get('std', 0):.4f}",
                    delta_color="off",
                )
            with cv3:
                st.metric("F1 min", f"{f1_s.get('min', 0):.4f}")
            with cv4:
                st.metric("F1 max", f"{f1_s.get('max', 0):.4f}")

            # Interpret stability
            std = f1_s.get('std', 0)
            mean = f1_s.get('mean', 0)
            if std <= 0.01:
                st.success(f"**Stable model** — F1 std={std:.4f}. "
                           "Performance is consistent across all data splits.")
            elif std <= 0.03:
                st.info(f"**Reasonably stable** — F1 std={std:.4f}. "
                        "Minor variation across folds, acceptable for most tasks.")
            else:
                st.warning(
                    f"**High variance** — F1 std={std:.4f}. "
                    "Performance varies a lot between folds. "
                    "Consider collecting more data or using stronger regularisation."
                )

            # Single-split vs CV comparison
            single_f1 = result.get('best_model_metrics', {}).get('f1_score')
            if single_f1 is not None:
                diff = single_f1 - mean
                if abs(diff) <= 0.02:
                    st.success(
                        f"**Single-split F1 ({single_f1:.4f}) ≈ CV mean ({mean:.4f})** — "
                        "the original 80/20 split gave a representative estimate."
                    )
                elif diff > 0.02:
                    st.warning(
                        f"**Single-split F1 ({single_f1:.4f}) > CV mean ({mean:.4f})** — "
                        "the original split was lucky. "
                        f"The CV mean ({mean:.4f}) is the more reliable estimate."
                    )
                else:
                    st.info(
                        f"**Single-split F1 ({single_f1:.4f}) < CV mean ({mean:.4f})** — "
                        "the original split was conservative. "
                        f"CV suggests the model is actually stronger ({mean:.4f})."
                    )

        # Per-fold table
        fold_data = cv.get('fold_results', [])
        if fold_data:
            fold_rows = []
            for r in fold_data:
                fold_rows.append({
                    'Fold':      r['fold'],
                    'Train Samples': r['n_train'],
                    'Val Samples':   r['n_val'],
                    'F1 Score':  f"{r['f1']:.4f}"  if r['f1']  is not None else 'Failed',
                    'Accuracy':  f"{r['accuracy']:.4f}" if r['accuracy'] is not None else 'Failed',
                    'Precision': f"{r['precision']:.4f}" if r['precision'] is not None else 'Failed',
                    'Recall':    f"{r['recall']:.4f}" if r['recall'] is not None else 'Failed',
                    'Status':    '✅' if r['error'] is None else f"❌ {r.get('error','')[:40]}",
                })
            st.markdown("**Per-Fold Breakdown**")
            st.dataframe(pd.DataFrame(fold_rows), use_container_width=True)

        # Raw text report
        exp_dir = result.get('experiment_dir', '')
        cv_report = os.path.join(exp_dir, 'cross_validation_report.txt')
        if os.path.exists(cv_report):
            with st.expander("📄 View Full CV Report (text)"):
                with open(cv_report, 'r', encoding='utf-8') as _f:
                    st.text(_f.read())

    # ── Training Log ─────────────────────────────────────────────────────────
    exp_dir = result.get('experiment_dir', '')
    log_path = os.path.join(exp_dir, 'pipeline.log')
    if os.path.exists(log_path):
        st.subheader("📋 Training Log")
        with st.expander("View full pipeline.log (per-epoch loss, early stopping, errors)", expanded=False):
            with open(log_path, 'r', encoding='utf-8', errors='replace') as _lf:
                log_text = _lf.read()
            # Show last 400 lines so very long logs don't freeze the browser
            lines = log_text.splitlines()
            if len(lines) > 400:
                st.caption(f"Showing last 400 of {len(lines)} lines. Full log at: `{log_path}`")
                log_text = "\n".join(lines[-400:])
            st.code(log_text, language=None)


# Main app
st.title("🚀 AutoLLM Text Classification")
st.markdown("**Automated machine learning for text classification tasks**")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["🏃 Run Pipeline", "📊 Results", "📜 History", "❓ Help"])

with tab1:
    st.header("Run AutoLLM Pipeline")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📥 Step 1: Upload CSV File")
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="CSV file with text and label columns"
        )
    
    with col2:
        st.subheader("⚙️ Options")
        experiment_name = st.text_input(
            "Experiment Name",
            value=datetime.now().strftime("%Y%m%d_%H%M%S"),
            help="Name for this experiment run"
        )
    
    if uploaded_file is not None:
        # Clear stale analysis from a previous run when the user uploads a different file.
        # Without this, the Advanced Options expander shows recommendations from the old dataset.
        _prev_file = st.session_state.get("_uploaded_file_name")
        if _prev_file != uploaded_file.name:
            st.session_state["_uploaded_file_name"] = uploaded_file.name
            st.session_state.pop("analysis", None)   # invalidate stale recommendations

        # Load and preview data — handle Windows (cp1252/latin-1) encoded files
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='latin-1')
        except Exception:
            st.error(
                "❌ Could not parse the uploaded file. "
                "Please make sure it is a valid CSV (not Excel, JSON, etc.)."
            )
            st.stop()
        
        st.subheader("📋 Step 2: Select Columns")
        
        # Select label column
        col1, col2 = st.columns([1, 1])
        with col1:
            label_column = st.selectbox(
                "Label Column",
                options=df.columns,
                help="Column containing class labels"
            )
        
        # Detect text columns
        try:
            from automl.data_validator import DataValidator  # autollm package
            validator = DataValidator()
            detected_info = validator.detect_text_columns(df, label_column)
            text_columns_available = detected_info['text_columns']
            column_stats = detected_info['column_stats']
            
            st.success(f"✓ Detected {len(text_columns_available)} text column(s)")
            
            # Display detected text columns with stats
            st.write("**Available Text Columns:**")
            for i, col in enumerate(text_columns_available, 1):
                stats = column_stats[col]
                col_marker = "🔵 PRIMARY" if i == 1 else "⚪ SECONDARY"
                st.write(f"{col_marker} **{col}**")
                st.write(f"   Avg Length: {stats['avg_length']:.0f} chars | "
                        f"Min: {stats['min_length']} | Max: {stats['max_length']}")
            
            # Multi-column selection
            st.write("\n**Select Text Columns to Use:**")
            selected_text_columns = st.multiselect(
                "Choose one or more text columns (order matters for merging)",
                options=text_columns_available,
                default=[text_columns_available[0]],  # Default to primary
                help="Select multiple columns to merge them with [SEP] separator. Order determines priority."
            )
            
            if not selected_text_columns:
                st.warning("Please select at least one text column")
                selected_text_columns = None
            
        except Exception as e:
            st.warning(f"Could not auto-detect text columns: {str(e)}")
            st.write("**Manual Text Column Selection:**")
            selected_text_columns = st.multiselect(
                "Select text columns",
                options=[col for col in df.columns if col != label_column],
                help="Manually select text columns to use"
            )
        
        # Display preview
        if selected_text_columns:
            # Show preview with selected columns
            st.write("\n**Data Preview:**")
            preview_df = df[[label_column] + selected_text_columns].head(5)
            st.dataframe(preview_df, width='stretch')
            display_data_preview(df, label_column, selected_text_columns[0])
        
        # ── Step 2.5: Model selector ─────────────────────────────────────────
        st.subheader("🤖 Step 2.5: Select Models to Train")

        _MODEL_CATALOG = [
            {
                "Model ID": "prajjwal1/bert-tiny",
                "Params": "4.4 M",
                "Speed": "⚡⚡⚡⚡",
                "Best For": "Very small datasets, fastest inference",
            },
            {
                "Model ID": "prajjwal1/bert-mini",
                "Params": "11 M",
                "Speed": "⚡⚡⚡",
                "Best For": "Small datasets, fast iteration",
            },
            {
                "Model ID": "google/mobilebert-uncased",
                "Params": "25 M",
                "Speed": "⚡⚡⚡",
                "Best For": "Latency-critical / mobile deployment",
            },
            {
                "Model ID": "distilbert-base-uncased",
                "Params": "66 M",
                "Speed": "⚡⚡",
                "Best For": "Balanced accuracy and speed",
            },
            {
                "Model ID": "bert-base-uncased",
                "Params": "110 M",
                "Speed": "⚡",
                "Best For": "Best accuracy, large datasets (≥ 10 K samples)",
            },
        ]
        _ALL_MODEL_IDS = [m["Model ID"] for m in _MODEL_CATALOG]

        import pandas as _pd
        st.dataframe(_pd.DataFrame(_MODEL_CATALOG), use_container_width=True, hide_index=True)

        # Compute auto-selected defaults so they appear pre-checked
        _auto_selected: list = []
        if "analysis" in st.session_state and st.session_state["analysis"]:
            _auto_selected = st.session_state["analysis"].get("model_selection", [])
        if not _auto_selected:
            # Fall back to showing all models pre-checked
            _auto_selected = _ALL_MODEL_IDS

        selected_models = st.multiselect(
            "Choose which models to train (leave empty to use auto-selection)",
            options=_ALL_MODEL_IDS,
            default=[m for m in _auto_selected if m in _ALL_MODEL_IDS],
            help=(
                "Auto-selection picks models based on dataset size. "
                "Pick specific models if you want to control the trade-off between speed and accuracy."
            ),
        )

        # Warn if user picks large models for small datasets
        if selected_models and df is not None:
            _large = [m for m in selected_models if "bert-base" in m or "distilbert" in m]
            if _large and len(df) < 2000:
                st.warning(
                    f"⚠️ {_large} are large models — your dataset has only {len(df):,} rows. "
                    "These models may overfit. Consider bert-tiny or bert-mini for small datasets."
                )

        # Start button
        st.subheader("🚀 Step 3: Run Pipeline")

        with st.expander("⚙️ Advanced Options"):
            # ── Pull auto-recommendations if a previous analysis exists ──────
            _prior_analysis = st.session_state.get("analysis", {})
            _rec = _prior_analysis.get("run_recommendations", {})
            _has_rec = bool(_rec)

            if _has_rec:
                st.markdown("**🤖 Auto-Recommendations** *(based on your dataset)*")
                _cv_icon  = "✅" if _rec.get("use_cv")     else "⏭️"
                _opt_icon = "✅" if _rec.get("use_optuna") else "⏭️"
                st.info(
                    f"{_cv_icon}  **Cross-Validation:** "
                    f"{'ON — ' + str(_rec.get('cv_folds', 5)) + '-fold' if _rec.get('use_cv') else 'OFF'}"
                    f"  —  *{_rec.get('cv_reason', '')}*\n\n"
                    f"{_opt_icon}  **Optuna tuning:** "
                    f"{'ON — ' + str(_rec.get('optuna_trials', 10)) + ' trials' if _rec.get('use_optuna') else 'OFF'}"
                    f"  —  *{_rec.get('optuna_reason', '')}*"
                )
                _override = st.checkbox(
                    "Override recommendations and set manually",
                    value=False,
                    key="adv_override",
                )
            else:
                # No analysis for this file yet — pipeline will auto-decide after run
                st.info(
                    "🤖 **Auto mode** — CV and Optuna settings will be decided automatically "
                    "based on your dataset when the pipeline runs. "
                    "Tick below to set them manually instead."
                )
                _override = st.checkbox(
                    "Set CV / Optuna manually",
                    value=False,
                    key="adv_override",
                )

            # ── Manual / override controls ────────────────────────────────────
            if _override:
                st.markdown("**Evaluation Strategy**")
                eval_strategy = st.radio(
                    "Choose how to evaluate model performance:",
                    options=["Train/Test Split (80/20)", "Cross-Validation (K-Fold)"],
                    index=0,
                    help=(
                        "Train/Test Split: fast, splits 80% training / 20% validation. "
                        "Cross-Validation: retrains the best model K times on different folds "
                        "and reports mean ± std metrics — more reliable but K× slower."
                    ),
                )
                use_cv = eval_strategy == "Cross-Validation (K-Fold)"

                if use_cv:
                    cv_folds = st.number_input(
                        "Number of folds",
                        min_value=3,
                        max_value=10,
                        value=5,
                        step=1,
                        help="3 folds = faster. 5 folds = more reliable (default). 10 = slowest.",
                    )
                    st.info(
                        f"⏱️ CV will retrain the best model **{cv_folds} times** on different data splits."
                    )
                else:
                    cv_folds = 5
                    st.info("Data will be split **80% training / 20% validation**.")

                st.divider()
                st.markdown("**Hyperparameter Optimization (Optuna)**")
                use_optuna = st.checkbox(
                    "Optimize learning rate & weight decay with Optuna",
                    value=False,
                    key="adv_use_optuna",
                    help=(
                        "Runs N short proxy trials per model to find the best learning rate "
                        "and weight decay. Each trial trains for 2 epochs on a small data slice."
                    ),
                )
                if use_optuna:
                    optuna_trials = st.number_input(
                        "Number of trials per model",
                        min_value=3,
                        max_value=20,
                        value=10,
                        step=1,
                        help="More trials = better search, but slower. 10 is a good default.",
                    )
                    st.info(
                        f"⏱️ Optuna will run **{optuna_trials} proxy trials** per model before full training."
                    )
                else:
                    optuna_trials = 10
            else:
                # Auto mode: pass None so pipeline uses _recommend_run_options()
                use_cv        = None
                cv_folds      = None
                use_optuna    = None
                optuna_trials = None
        
        if st.button("🚀 Start Pipeline", type="primary"):
            if selected_text_columns and label_column:
                import threading
                import re as _re

                # Sanitize filename to prevent path traversal (e.g. ../../module.py)
                safe_name = _re.sub(r'[^\w.\-]', '_', uploaded_file.name) or "upload.csv"
                safe_name = safe_name.lstrip('.')
                temp_path = f"temp_{safe_name}"
                with open(temp_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                    f.flush()
                    import os as _os
                    _os.fsync(f.fileno())

                # ── Thread-safe result container ─────────────────────────────
                _run = {"result": None, "done": False, "error": None}

                def _pipeline_thread():
                    try:
                        _pipeline = AutoLLMPipeline(output_dir="experiments")
                        _run["result"] = _pipeline.run(
                            csv_path=temp_path,
                            label_column=label_column,
                            text_columns=selected_text_columns,
                            experiment_name=experiment_name,
                            use_cv=use_cv,
                            cv_folds=int(cv_folds) if cv_folds is not None else None,
                            use_optuna=use_optuna,
                            optuna_trials=int(optuna_trials) if optuna_trials is not None else None,
                            model_names=selected_models if selected_models else None,
                        )
                    except Exception as _e:
                        _run["error"] = str(_e)
                    finally:
                        _run["done"] = True

                _t = threading.Thread(target=_pipeline_thread, daemon=True)
                _start_time = time.time()
                _t.start()

                # ── Progress UI placeholders ──────────────────────────────────
                st.markdown("---")
                st.markdown("### ⚙️ Pipeline Running")
                _prog       = st.progress(0)
                _status_ph  = st.empty()
                _step_ph    = st.empty()
                _log_ph     = st.empty()

                # ── Step map: log keyword → (progress %, step label, detail) ──
                # Ordered from first to last so we pick the highest matched step.
                _STEP_MAP = [
                    ("AUTOLLM PIPELINE STARTED",    3,
                     "🚀 Pipeline started",
                     "Initialising components and seed..."),
                    ("STEP 1",                       8,
                     "📂 Step 1/7 — Data Validation",
                     "Loading CSV, checking columns, removing duplicates..."),
                    ("STEP 2",                      20,
                     "🧠 Step 2/7 — Data Intelligence",
                     "Measuring class balance, token lengths, picking strategy..."),
                    ("CRITICAL STRATEGY",            22,
                     "🧠 Step 2/7 — Strategy: CRITICAL",
                     "< 50 samples/class — aggressive regularisation mode"),
                    ("SMALL DATA STRATEGY",          22,
                     "🧠 Step 2/7 — Strategy: SMALL",
                     "50–200 samples/class — moderate regularisation"),
                    ("MODERATE DATA STRATEGY",       22,
                     "🧠 Step 2/7 — Strategy: MODERATE",
                     "200–500 samples/class — light regularisation"),
                    ("GOOD DATA STRATEGY",           22,
                     "🧠 Step 2/7 — Strategy: GOOD",
                     ">= 500 samples/class — standard fine-tuning"),
                    ("STEP 3",                      30,
                     "🔄 Step 3/7 — Data Preparation",
                     "Splitting train/val, encoding labels, computing class weights..."),
                    ("STEP 4",                      36,
                     "🤖 Step 4/7 — Model Training",
                     "Loading models and starting fine-tuning..."),
                    ("Optuna skipped",               37,
                     "🤖 Step 4/7 — Optuna skipped",
                     "Too few samples for reliable tuning — using rule-based config"),
                    ("OPTUNA HYPERPARAMETER SEARCH", 37,
                     "🔍 Step 4/7 — Optuna search",
                     "Running short proxy trials to find best lr/weight_decay..."),
                    ("Using Optuna params",           40,
                     "🔍 Step 4/7 — Optuna complete",
                     "Best hyperparameters found, starting full training..."),
                    ("Starting training",             42,
                     "🤖 Step 4/7 — Training in progress",
                     "Fine-tuning model weights epoch by epoch..."),
                    ("Epoch ",                        None,   # handled specially
                     None, None),
                    ("Training completed in",         80,
                     "🤖 Step 4/7 — Training complete",
                     "Model saved, moving to next model or evaluation..."),
                    ("STEP 5",                        85,
                     "📊 Step 5/7 — Model Evaluation",
                     "Computing F1, accuracy, latency on validation set..."),
                    ("STEP 6",                        92,
                     "🏆 Step 6/7 — Best Model Selection",
                     "Comparing all models by composite score..."),
                    ("STEP 7",                        95,
                     "📋 Step 7/7 — Report Generation",
                     "Evaluating on train set, checking overfitting, writing report..."),
                    ("PIPELINE COMPLETED",            99,
                     "✅ Pipeline complete",
                     "Wrapping up..."),
                ]

                _current_pct = 0
                _current_label = "🚀 Starting pipeline..."
                _current_detail = "Waiting for first log output..."
                _log_path = None

                # ── Polling loop — updates UI every 1.5 s until thread ends ──
                while not _run["done"]:
                    # Find log file once the experiment directory appears
                    if _log_path is None:
                        try:
                            _exp_base = Path("experiments")
                            if _exp_base.exists():
                                _dirs = sorted(
                                    [d for d in _exp_base.iterdir() if d.is_dir()],
                                    key=lambda d: d.stat().st_ctime,
                                    reverse=True,
                                )
                                if _dirs and _dirs[0].stat().st_ctime >= _start_time - 3:
                                    _log_path = _dirs[0] / "pipeline.log"
                        except Exception:
                            pass

                    _epoch_line = ""
                    if _log_path and _log_path.exists():
                        try:
                            _log_text  = _log_path.read_text(encoding="utf-8", errors="replace")
                            _log_lines = _log_text.splitlines()

                            # Walk step map to find highest matched progress
                            for _keyword, _pct, _label, _detail in _STEP_MAP:
                                if _keyword == "Epoch ":
                                    continue  # handled below
                                if _keyword in _log_text and _pct is not None and _pct > _current_pct:
                                    _current_pct    = _pct
                                    _current_label  = _label
                                    _current_detail = _detail

                            # Interpolate epoch progress (42 % → 80 %)
                            # Each "Epoch X/Y" line moves the needle inside training range
                            for _line in reversed(_log_lines):
                                if "Epoch " in _line and "/" in _line:
                                    _epoch_line = _line.strip()
                                    try:
                                        # Robust epoch parse: match "N/M" anywhere after "Epoch"
                                        import re as _epoch_re
                                        _m = _epoch_re.search(r'Epoch\s+(\d+)/(\d+)', _line)
                                        if _m:
                                            _cur_ep = int(_m.group(1))
                                            _tot_ep = int(_m.group(2))
                                            _ep_pct = 42 + int((_cur_ep / max(_tot_ep, 1)) * 38)
                                            if _ep_pct > _current_pct:
                                                _current_pct    = _ep_pct
                                                _current_label  = f"🤖 Training — epoch {_cur_ep}/{_tot_ep}"
                                                _current_detail = _line.strip()
                                    except Exception:
                                        pass
                                    break

                            # Last non-HTTP log line for live feed
                            _last_line = ""
                            for _line in reversed(_log_lines):
                                _s = _line.strip()
                                if _s and "HTTP Request" not in _s and "httpx" not in _s:
                                    _last_line = _s
                                    break

                            # Show last 12 lines in the log box (skip HTTP noise)
                            _clean_lines = [
                                l for l in _log_lines
                                if "HTTP Request" not in l and "httpx" not in l
                            ]
                            _log_ph.code(
                                "\n".join(_clean_lines[-12:]) if _clean_lines else "...",
                                language=None,
                            )
                        except Exception:
                            pass

                    _elapsed = int(time.time() - _start_time)
                    _prog.progress(min(_current_pct, 99))
                    _status_ph.markdown(
                        f"**{_current_label}** &nbsp;|&nbsp; ⏱️ {_elapsed}s elapsed"
                    )
                    _step_ph.caption(f"ℹ️ {_current_detail}")
                    time.sleep(1.5)

                # ── Thread finished ───────────────────────────────────────────
                result = _run["result"]
                if _run["error"] and result is None:
                    result = {"status": "failed", "error": _run["error"]}

                _prog.progress(100)
                _status_ph.empty()
                _step_ph.empty()
                _log_ph.empty()

                # Cleanup temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)

                # Store and display results
                st.session_state.pipeline_results = result
                st.session_state.current_experiment = result
                if result.get('status') == 'success':
                    st.session_state["analysis"] = result.get('data_analysis', {})
                    save_experiment_to_history(result)
                    display_pipeline_results(result)
                else:
                    st.error(f"❌ Pipeline failed: {result.get('error', 'Unknown error')}")

            else:
                st.warning("⚠️ Please select label column and at least one text column")

with tab2:
    st.header("📊 Results & Analysis")
    
    if st.session_state.pipeline_results:
        result = st.session_state.pipeline_results
        
        if result.get('status') == 'success':
            display_pipeline_results(result)
        else:
            st.error(f"❌ Pipeline Error: {result.get('error')}")
    else:
        st.info("📌 No results yet. Run a pipeline in the 'Run Pipeline' tab to see results.")

with tab3:
    st.header("📜 Experiment History")
    
    if st.session_state.experiment_history:
        history_df = pd.DataFrame([
            {
                'Timestamp': h['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
                'Experiment': h['experiment_name'],
                'Best Model': h['best_model'],
                'F1 Score': f"{h['f1_score']:.4f}" if h['f1_score'] else 'N/A',
                'Accuracy': f"{h['accuracy']:.4f}" if h['accuracy'] else 'N/A',
            }
            for h in st.session_state.experiment_history
        ])
        
        st.dataframe(history_df, width='stretch')
        
        # Select experiment to view
        if len(st.session_state.experiment_history) > 0:
            selected_idx = st.selectbox(
                "View Experiment",
                range(len(st.session_state.experiment_history)),
                format_func=lambda i: st.session_state.experiment_history[i]['experiment_name']
            )
            
            if st.button("📊 View Details"):
                st.session_state.pipeline_results = st.session_state.experiment_history[selected_idx]['result']
                st.rerun()
    else:
        st.info("📌 No experiments run yet.")

with tab4:
    st.header("❓ Help & Documentation")
    
    st.subheader("🚀 Quick Start")
    st.markdown("""
    1. **Upload CSV**: Choose a file with text and label columns
    2. **Select Columns**: Pick which columns contain text and labels
    3. **Run Pipeline**: Click start and wait for results
    
    The AutoLLM system will automatically:
    - Analyze your data
    - Detect class imbalance
    - Select optimal models
    - Train and evaluate
    - Report the best results
    """)
    
    st.subheader("📋 Data Format")
    st.markdown("""
    Your CSV should have at least 2 columns:
    
    | Column | Type | Example |
    |--------|------|---------|
    | Text | string | "This is great!" |
    | Label | string/int | "positive" |
    
    **Example CSV**:
    ```
    text,label
    "I love this product!",positive
    "Terrible experience.",negative
    "Amazing quality!",positive
    ```
    """)
    
    st.subheader("💡 Tips")
    st.markdown("""
    - **Minimum samples**: At least 100 samples total
    - **Balanced data**: Try to have similar samples per class
    - **Clean text**: Remove special characters if possible
    - **Avoid duplicates**: The system removes them automatically
    """)
    
    st.subheader("🎯 Understanding Results")
    st.markdown("""
    - **F1 Score**: Balance between precision and recall (0-1, higher is better)
    - **Accuracy**: Percentage of correct predictions (0-1, higher is better)
    - **Latency**: Time to make prediction (ms, lower is better)
    - **Imbalance Ratio**: Class distribution (1.0 = balanced)
    """)
    
    st.subheader("🛠️ Troubleshooting")
    st.markdown("""
    **Q: "CUDA out of memory"**
    - Use a smaller model or reduce batch size
    
    **Q: "Very low F1 score"**
    - Check data quality and balance
    - Ensure sufficient training samples
    
    **Q: "Pipeline very slow"**
    - Use a smaller model like MiniLM
    - Reduce number of epochs
    """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9em;'>
    🤖 AutoLLM Text Classification System | Built with Streamlit + PyTorch + HuggingFace
</div>
""", unsafe_allow_html=True)
