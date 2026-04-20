"""
Quick Start - Test AutoLLM Pipeline Immediately
Run this script to test the pipeline with sample data
"""

from automl import AutoLLMPipeline
import sys
import os


def main():
    print("\n" + "="*70)
    print("🚀 AutoLLM Pipeline - Quick Start Test")
    print("="*70 + "\n")
    
    # Check if sample data exists
    sample_data_path = "data/sample_dataset.csv"
    if not os.path.exists(sample_data_path):
        print(f"❌ Sample data not found at {sample_data_path}")
        print("Please run from the project root directory")
        return
    
    print(f"✓ Found sample data at {sample_data_path}\n")
    
    try:
        # Initialize pipeline
        print("📌 Step 1: Initializing AutoLLM Pipeline...")
        pipeline = AutoLLMPipeline(output_dir="experiments")
        print("✓ Pipeline initialized\n")
        
        # Run pipeline
        print("📌 Step 2: Running AutoLLM Pipeline...")
        print("This will:")
        print("  • Analyze your data")
        print("  • Select optimal models")
        print("  • Train and evaluate")
        print("  • Generate reports\n")
        print("⏳ This may take a few minutes (first time setup)...\n")
        
        result = pipeline.run(
            csv_path=sample_data_path,
            label_column="label",
            text_column="text",
            experiment_name="quickstart_test"
        )
        
        # Display results
        print("\n" + "="*70)
        print("📊 RESULTS")
        print("="*70 + "\n")
        
        if result['status'] == 'success':
            print(f"✓ Pipeline completed successfully!\n")
            print(f"Best Model: {result['best_model_name']}")
            print(f"F1 Score: {result['best_model_metrics']['f1_score']:.4f}")
            print(f"Accuracy: {result['best_model_metrics']['accuracy']:.4f}")
            print(f"Latency: {result['best_model_metrics']['latency_ms']:.2f}ms\n")
            print(f"📁 Experiment saved to: {result['experiment_dir']}")
            print(f"📋 Report: {result['experiment_dir']}/best_model_report.txt\n")
            
            print("="*70)
            print("✅ SUCCESS! Your AutoLLM system is working!")
            print("="*70 + "\n")
            
            print("📚 Next Steps:")
            print("1. Try with your own CSV file:")
            print("   - Prepare CSV with 'text' and 'label' columns")
            print("   - Run: pipeline.run(csv_path='your_file.csv', ...)\n")
            
            print("2. Check the comparison table:")
            print(result['comparison_df'].to_string())
            
        else:
            print(f"❌ Pipeline failed: {result['error']}")
            return
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
