"""
Example: Run AutoLLM Pipeline
Demonstrates how to use the AutoLLM system
"""

from automl import AutoMLPipeline
import sys


def main():
    """
    Example usage of AutoLLM pipeline
    """
    
    # Initialize pipeline
    pipeline = AutoLLMPipeline(output_dir="experiments")
    
    # Example 1: With explicit column names
    # Uncomment and modify with your CSV path
    # result = pipeline.run(
    #     csv_path="data/your_data.csv",
    #     label_column="label",
    #     text_column="text",
    #     experiment_name="my_first_experiment"
    # )
    
    # Example 2: With auto-detection of text column
    # result = pipeline.run(
    #     csv_path="data/your_data.csv",
    #     label_column="category",
    #     experiment_name="auto_detected_text"
    # )
    
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "AutoLLM Pipeline Example" + " "*30 + "║")
    print("╚" + "="*68 + "╝")
    
    print("\n📋 To get started:\n")
    print("1. Prepare your CSV file with:")
    print("   - A column with text data (messages, reviews, descriptions, etc.)")
    print("   - A column with labels (class names or categories)")
    print()
    print("2. Run the pipeline:")
    print()
    print("   from automl import AutoLLMPipeline")
    print()
    print("   pipeline = AutoLLMPipeline(output_dir='experiments')")
    print()
    print("   result = pipeline.run(")
    print("       csv_path='data/your_data.csv',")
    print("       label_column='label',")
    print("       text_column='text'  # or use None for auto-detection")
    print("   )")
    print()
    print("3. The pipeline will:")
    print("   ✓ Validate and analyze your data")
    print("   ✓ Select appropriate models (MiniLM, DistilBERT, etc.)")
    print("   ✓ Automatically tune hyperparameters")
    print("   ✓ Train and evaluate multiple models")
    print("   ✓ Select and save the best model")
    print("   ✓ Generate evaluation reports")
    print()
    print("📁 Output will be saved to experiments/[timestamp]/")
    print()
    print("="*70)
    print()
    
    # Example dataset structure:
    print("📄 Example CSV Format:\n")
    example_csv = """
    text,label
    "This movie is amazing!",positive
    "Terrible experience here.",negative
    "Best product ever bought.",positive
    "Would not recommend.",negative
    "Absolutely loved it!",positive
    """
    print(example_csv)
    
    print("\n" + "="*70)
    print("💡 Need help? Check README.md for detailed instructions")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
