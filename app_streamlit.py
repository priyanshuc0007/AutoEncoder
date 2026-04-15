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

# Import AutoML components
from automl import AutoMLPipeline

# Page config
st.set_page_config(
    page_title="🤖 AutoML Text Classification",
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
        st.metric("Unique Classes", df[label_col].nunique())
    with col3:
        st.metric("Avg Text Length", int(df[text_col].astype(str).str.len().mean()))
    with col4:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    st.write("First few rows:")
    st.dataframe(df.head(10), use_container_width=True)
    
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
    
    # Key metrics
    st.subheader("🏆 Best Model Results")
    
    metrics = result.get('best_model_metrics', {})
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Model", result.get('best_model_name', 'Unknown')[:25])
    with col2:
        st.metric("F1 Score", f"{metrics.get('f1_score', 0):.4f}")
    with col3:
        st.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
    with col4:
        st.metric("Latency (ms)", f"{metrics.get('latency_ms', 0):.2f}")
    
    # Model comparison
    st.subheader("📊 Model Comparison")
    comparison_df = result.get('comparison_df')
    if comparison_df is not None:
        st.dataframe(comparison_df, use_container_width=True)
    
    # Experiment info
    st.subheader("ℹ️ Experiment Information")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Experiment Name**: {result.get('experiment_name')}")
        st.write(f"**Model Path**: `{result.get('best_model_path')}`")
    with col2:
        st.write(f"**Output Directory**: `{result.get('experiment_dir')}`")
        
        # Download best model button
        if os.path.exists(result.get('experiment_dir')):
            st.info("🔍 Model files are saved to the experiment directory")
    
    # Data analysis
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
        st.write(f"- Imbalance Ratio: **{imbal.get('imbalance_ratio', 'N/A'):.2f}**")
        st.write(f"- Use Class Weights: **{'✓ YES' if imbal.get('use_class_weights') else '✗ NO'}**")
        st.write(f"- Use Focal Loss: **{'✓ YES' if imbal.get('use_focal_loss') else '✗ NO'}**")
    
    with col2:
        st.write("**Text Information**")
        text_info = analysis.get('text_info', {})
        st.write(f"- Average Length: **{text_info.get('avg_length', 'N/A'):.0f}**")
        st.write(f"- Max Length (95%): **{text_info.get('p95_length', 'N/A')}**")
        
        st.write("\n**Training Configuration**")
        config = analysis.get('training_config', {})
        st.write(f"- Batch Size: **{config.get('batch_size', 'N/A')}**")
        st.write(f"- Learning Rate: **{config.get('learning_rate', 'N/A')}**")


# Main app
st.title("🚀 AutoML Text Classification")
st.markdown("**Automated machine learning for text classification tasks**")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["🏃 Run Pipeline", "📊 Results", "📜 History", "❓ Help"])

with tab1:
    st.header("Run AutoML Pipeline")
    
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
        # Load and preview data
        df = pd.read_csv(uploaded_file)
        
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
            from automl.data_validator import DataValidator
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
            st.dataframe(preview_df, use_container_width=True)
            display_data_preview(df, label_column, selected_text_columns[0])
        
        # Start button
        st.subheader("🚀 Step 3: Run Pipeline")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            use_gpu = st.checkbox("Use GPU", value=True, help="Use GPU if available")
        
        with col2:
            simulate = st.checkbox("Simulate Run", value=False, help="Skip training (demo mode)")
        
        with col3:
            pass
        
        if st.button("🚀 Start Pipeline", type="primary"):
            if selected_text_columns and label_column:
                # Save uploaded file temporarily
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                
                st.info("⏳ Pipeline starting... This may take a few minutes...")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Initialize pipeline
                    pipeline = AutoMLPipeline(output_dir="experiments")
                    
                    # Update progress
                    progress_bar.progress(10)
                    status_text.text("📂 Validating data...")
                    
                    # Run pipeline with selected text columns
                    result = pipeline.run(
                        csv_path=temp_path,
                        label_column=label_column,
                        text_columns=selected_text_columns,
                        experiment_name=experiment_name
                    )
                    
                    # Clean up temp file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    
                    # Update progress
                    progress_bar.progress(100)
                    status_text.text("✅ Complete!")
                    
                    # Store results
                    st.session_state.pipeline_results = result
                    st.session_state.current_experiment = result
                    
                    # Save to history
                    if result.get('status') == 'success':
                        save_experiment_to_history(result)
                    
                    # Display results
                    display_pipeline_results(result)
                    
                except Exception as e:
                    st.error(f"❌ Pipeline failed: {str(e)}")
                    progress_bar.progress(0)
                    status_text.text("Failed")
            
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
        
        st.dataframe(history_df, use_container_width=True)
        
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
    
    The AutoML system will automatically:
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
    🤖 AutoML Text Classification System | Built with Streamlit + PyTorch + HuggingFace
</div>
""", unsafe_allow_html=True)
