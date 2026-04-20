"""
Data Validation Module
Validates and loads CSV data for AutoLLM pipeline
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataValidator:
    """Validates and loads CSV data"""
    
    def __init__(self, min_samples: int = 100):
        """
        Initialize validator
        
        Args:
            min_samples: Minimum number of samples required
        """
        self.min_samples = min_samples
    
    def load_and_validate(
        self, 
        csv_path: str, 
        label_column: str,
        text_column: Optional[str] = None
    ) -> Tuple[pd.DataFrame, str, str]:
        """
        Load and validate CSV file
        
        Args:
            csv_path: Path to CSV file
            label_column: Name of label column
            text_column: Name of text column (auto-detected if None)
            
        Returns:
            Tuple of (dataframe, label_column, text_column)
            
        Raises:
            ValueError: If validation fails
        """
        # Load CSV
        logger.info(f"Loading CSV from {csv_path}")
        try:
            try:
                df = pd.read_csv(csv_path, encoding='utf-8')
            except UnicodeDecodeError:
                logger.warning("UTF-8 decode failed, retrying with latin-1 encoding")
                df = pd.read_csv(csv_path, encoding='latin-1')
        except Exception as e:
            raise ValueError(f"Failed to load CSV: {str(e)}")
        
        # Validate shape
        if len(df) < self.min_samples:
            raise ValueError(f"Dataset has {len(df)} samples, minimum required: {self.min_samples}")
        
        logger.info(f"Loaded {len(df)} samples")
        
        # Validate label column exists
        if label_column not in df.columns:
            raise ValueError(f"Label column '{label_column}' not found in CSV")

        # Regression guard — run BEFORE dtype conversion so the original numeric
        # dtype is still visible.  High-cardinality numeric columns (age, price, etc.)
        # are almost certainly regression targets, not discrete class labels.
        if pd.api.types.is_numeric_dtype(df[label_column]):
            num_unique_raw = df[label_column].nunique()
            cardinality_ratio_raw = num_unique_raw / max(len(df), 1)
            if num_unique_raw > 20 and cardinality_ratio_raw > 0.2:
                raise ValueError(
                    f"Label column '{label_column}' looks like a continuous/regression target: "
                    f"{num_unique_raw} unique numeric values out of {len(df)} rows "
                    f"({cardinality_ratio_raw*100:.1f}% cardinality). "
                    f"This pipeline only supports classification. "
                    f"If these are genuinely discrete classes, cast the column to string before running."
                )

        # Normalise label column dtype:
        # - Float labels (1.0, 2.0) that are actually integer class IDs → "1", "2"
        #   so they don't get encoded as different classes from "1" / "2" string datasets.
        # - Int labels (0, 1, 2) → "0", "1", "2" (string) for consistent LabelEncoder behaviour.
        if pd.api.types.is_float_dtype(df[label_column]):
            non_null = df[label_column].dropna()
            if not non_null.empty and (non_null == non_null.astype(int)).all():
                # Convert the FULL column so NaN rows keep NaN and valid rows become strings.
                # Using .where() preserves NaN positions; the NaN rows are dropped later.
                df[label_column] = df[label_column].where(
                    df[label_column].isna(),
                    df[label_column].fillna(0).astype(int).astype(str)
                )
                logger.info(
                    f"Label column '{label_column}': converted float values to integer strings "
                    f"(e.g. 1.0 → '1') for consistent encoding."
                )
        elif pd.api.types.is_integer_dtype(df[label_column]):
            df[label_column] = df[label_column].astype(str)
            logger.info(
                f"Label column '{label_column}': converted integer values to strings "
                f"for consistent encoding."
            )
        
        # Auto-detect text column if not provided
        if text_column is None:
            text_column = self._detect_text_column(df, label_column)
            logger.info(f"Auto-detected text column: {text_column}")
        else:
            if text_column not in df.columns:
                raise ValueError(f"Text column '{text_column}' not found in CSV")
        
        # Check for missing values in key columns
        label_missing = df[label_column].isna().sum()
        text_missing = df[text_column].isna().sum()
        
        if label_missing > 0:
            logger.warning(f"Found {label_missing} missing values in label column, dropping rows")
            df = df.dropna(subset=[label_column])
        
        if text_missing > 0:
            logger.warning(f"Found {text_missing} missing values in text column, dropping rows")
            df = df.dropna(subset=[text_column])
        
        if len(df) < self.min_samples:
            raise ValueError(f"After removing NaN values, only {len(df)} samples remain")

        # Drop whitespace-only texts (they tokenize to just [CLS][SEP] and add noise)
        whitespace_mask = df[text_column].astype(str).str.strip() == ""
        whitespace_count = int(whitespace_mask.sum())
        if whitespace_count > 0:
            logger.warning(
                f"Dropping {whitespace_count} rows whose text column is whitespace-only "
                f"(they would tokenize to empty sequences)"
            )
            df = df[~whitespace_mask]

        if len(df) < self.min_samples:
            raise ValueError(
                f"After removing whitespace-only texts, only {len(df)} samples remain"
            )

        # Remove duplicates
        initial_len = len(df)
        df = df.drop_duplicates(subset=[text_column])
        duplicates_removed = initial_len - len(df)
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate samples ({100*duplicates_removed/initial_len:.2f}%)")

        # Guard: need at least 2 classes to do classification
        num_classes = df[label_column].nunique()
        if num_classes < 2:
            raise ValueError(
                f"Label column '{label_column}' contains only 1 unique class. "
                f"Classification requires at least 2 classes."
            )

        # After deduplication, verify each class still has enough samples for a stratified split
        min_class_count = df[label_column].value_counts().min()
        if min_class_count < 2:
            raise ValueError(
                f"After deduplication, at least one class has only {min_class_count} sample(s). "
                f"Stratified train/val split requires at least 2 samples per class. "
                f"Please collect more data or reduce the number of classes."
            )
        
        # Validate text column is string type
        df[text_column] = df[text_column].astype(str)
        
        logger.info(f"Final dataset: {len(df)} samples")
        logger.info(f"Label column: {label_column}")
        logger.info(f"Text column: {text_column}")
        
        return df, label_column, text_column
    
    def detect_text_columns(self, df: pd.DataFrame, label_column: str) -> dict:
        """
        Detect ALL text columns in the dataframe
        Returns both single column and multiple columns info
        
        Args:
            df: DataFrame
            label_column: Label column name
            
        Returns:
            Dictionary with:
            - text_columns: list of all text column names (sorted by avg length, descending)
            - primary_text_column: longest text column
            - column_stats: dict with stats for each column
        """
        text_columns_info = []

        for col in df.columns:
            if col == label_column:
                continue
            # Try ALL columns regardless of dtype — cast to str to measure text length.
            # This handles columns stored as int/float that actually contain text IDs or
            # short category strings (e.g. a text column accidentally inferred as numeric).
            try:
                text_series = df[col].astype(str)
                avg_len = text_series.str.len().mean()

                # Primary threshold: avg length > 10 chars indicates real text content
                if avg_len > 10:
                    # Warn if > 50% of rows share the same value (near-constant column)
                    top_freq = text_series.value_counts(normalize=True).iloc[0]
                    if top_freq > 0.5:
                        logger.warning(
                            f"Column '{col}' has {top_freq*100:.0f}% of rows with the same "
                            f"value — likely a near-constant column. It will be included but "
                            f"may not carry useful signal."
                        )
                    text_columns_info.append({
                        'name': col,
                        'avg_length': avg_len,
                        'min_length': text_series.str.len().min(),
                        'max_length': text_series.str.len().max(),
                        'non_null_count': text_series.notna().sum(),
                    })
            except Exception:
                logger.warning(f"Could not analyse column '{col}' for text detection, skipping")

        # Fallback: if no column cleared the avg_length > 10 bar, pick the column with the
        # highest average length among all non-label columns.  This handles datasets where
        # all texts happen to be very short (e.g. single-word labels in the wrong column).
        if not text_columns_info:
            best_col, best_avg = None, 0.0
            for col in df.columns:
                if col == label_column:
                    continue
                try:
                    avg_len = float(df[col].astype(str).str.len().mean())
                    if avg_len > best_avg:
                        best_avg = avg_len
                        best_col = col
                except Exception:
                    pass
            if best_col is not None:
                logger.warning(
                    f"No column has avg text length > 10. "
                    f"Falling back to '{best_col}' (avg {best_avg:.1f} chars) as the text column. "
                    f"Verify this is the correct column."
                )
                text_series = df[best_col].astype(str)
                text_columns_info.append({
                    'name': best_col,
                    'avg_length': best_avg,
                    'min_length': text_series.str.len().min(),
                    'max_length': text_series.str.len().max(),
                    'non_null_count': text_series.notna().sum(),
                })
            else:
                raise ValueError(
                    f"Could not find any text column. The dataset has only the label column '{label_column}'. "
                    f"Please provide a CSV with at least one text/feature column."
                )

        
        # Sort by average length (descending)
        text_columns_info.sort(key=lambda x: x['avg_length'], reverse=True)
        
        logger.info(f"\n{'='*70}")
        logger.info("📝 TEXT COLUMNS DETECTED")
        logger.info(f"{'='*70}")
        for i, col_info in enumerate(text_columns_info, 1):
            primary_marker = "🔵 PRIMARY" if i == 1 else "⚪ SECONDARY"
            logger.info(f"{primary_marker}: {col_info['name']}")
            logger.info(f"   Avg Length: {col_info['avg_length']:.0f} | "
                       f"Min: {col_info['min_length']} | Max: {col_info['max_length']}")
        logger.info(f"{'='*70}\n")
        
        return {
            'text_columns': [col['name'] for col in text_columns_info],
            'primary_text_column': text_columns_info[0]['name'],
            'column_stats': {col['name']: col for col in text_columns_info},
        }
    
    def _detect_text_column(self, df: pd.DataFrame, label_column: str) -> str:
        """
        Auto-detect text column by finding the longest average string length
        (Kept for backward compatibility)
        
        Args:
            df: DataFrame
            label_column: Label column name
            
        Returns:
            Name of detected text column
        """
        result = self.detect_text_columns(df, label_column)
        return result['primary_text_column']
    
    def merge_text_columns(
        self, 
        df: pd.DataFrame, 
        text_columns: list,
        max_tokens: int = 512,
        tokenizer_name: str = "bert-base-uncased"
    ) -> Tuple[pd.DataFrame, str]:
        """
        Merge multiple text columns into a single combined text column
        with intelligent truncation to stay under token limit
        
        Args:
            df: DataFrame
            text_columns: List of column names to merge (in priority order)
            max_tokens: Maximum tokens (BERT limit = 512)
            tokenizer_name: Tokenizer to use for token counting
            
        Returns:
            Tuple of (modified_dataframe, combined_column_name)
        """
        from transformers import AutoTokenizer
        
        logger.info(f"\n{'='*70}")
        logger.info("🔗 MERGING TEXT COLUMNS")
        logger.info(f"{'='*70}")
        logger.info(f"Columns to merge: {text_columns}")
        logger.info(f"Max tokens: {max_tokens}")
        
        # Load tokenizer for token counting
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        except Exception:
            logger.warning(f"Could not load tokenizer {tokenizer_name}, using approximate token counting")
            tokenizer = None
        
        # Create combined column — use a unique name to avoid clobbering any
        # existing column the user may already have named 'combined_text'.
        combined_column_name = "combined_text"
        while combined_column_name in df.columns:
            combined_column_name += "_merged"
        
        def merge_row(row):
            """Merge multiple text columns into one with [SEP] separator"""
            parts = []
            for col in text_columns:
                text = str(row[col]).strip()
                if text and text.lower() != 'nan':
                    # Add column name as prefix: "column_name: text"
                    parts.append(f"{col}: {text}")
            
            combined = " [SEP] ".join(parts)
            return combined
        
        logger.info("Concatenating columns...")
        df[combined_column_name] = df[text_columns].apply(merge_row, axis=1)
        
        # Check token counts
        if tokenizer:
            logger.info("Analyzing token distribution...")
            token_counts = df[combined_column_name].apply(
                lambda x: len(tokenizer.encode(x, truncation=False))
            )
        else:
            # Rough estimate: 1 token ≈ 4 characters
            token_counts = df[combined_column_name].str.len() / 4
        
        avg_tokens = token_counts.mean()
        max_text_tokens = token_counts.max()
        num_over_limit = (token_counts > max_tokens).sum()
        pct_over_limit = 100 * num_over_limit / len(df)
        
        logger.info(f"Token Distribution:")
        logger.info(f"  Avg: {avg_tokens:.0f} tokens")
        logger.info(f"  Max: {max_text_tokens:.0f} tokens")
        logger.info(f"  Over limit ({max_tokens}): {num_over_limit} texts ({pct_over_limit:.1f}%)")
        
        # If many texts exceed limit, truncate smart ly
        if num_over_limit > 0:
            logger.warning(f"⚠️  {num_over_limit} texts exceed token limit, applying smart truncation...")
            
            def smart_truncate(text):
                """Truncate while preserving main content"""
                if tokenizer is None:
                    # Fallback: rough char-based truncation (≈4 chars per token),
                    # keeping 80% of the limit to match the tokenizer-based path.
                    _KEEP_RATIO = 0.8
                    max_chars = int(max_tokens * 4 * _KEEP_RATIO)
                    return text[:max_chars] if len(text) > max_chars else text

                tokens = tokenizer.encode(text, truncation=False)
                if len(tokens) > max_tokens:
        # Keep first 80% of tokens to preserve primary content over secondary
                    _KEEP_RATIO = 0.8
                    truncated = tokenizer.decode(tokens[:int(max_tokens * _KEEP_RATIO)])
                    return truncated
                return text
            
            df[combined_column_name] = df[combined_column_name].apply(smart_truncate)
            logger.info("✓ Smart truncation applied")
        else:
            logger.info("✓ All texts within token limit")
        
        logger.info(f"✓ Combined text column created: {combined_column_name}")
        logger.info(f"{'='*70}\n")
        
        return df, combined_column_name
    
    def get_statistics(self, df: pd.DataFrame, label_column: str, text_column: str) -> dict:
        """
        Get basic dataset statistics
        
        Args:
            df: DataFrame
            label_column: Label column name
            text_column: Text column name
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            'num_samples': len(df),
            'num_classes': df[label_column].nunique(),
            'class_distribution': df[label_column].value_counts().to_dict(),
            'avg_text_length': df[text_column].astype(str).str.len().mean(),
            'min_text_length': df[text_column].astype(str).str.len().min(),
            'max_text_length': df[text_column].astype(str).str.len().max(),
        }
        return stats
