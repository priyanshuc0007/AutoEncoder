"""
Data Validation Module
Validates and loads CSV data for AutoLLM pipeline
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataValidator:
    """Validates and loads CSV data"""

    # ── Multi-label helpers ────────────────────────────────────────────────

    @staticmethod
    def _is_multi_label_column(col: pd.Series) -> bool:
        """Heuristic: detect if a label column contains multiple labels per row.

        Looks for comma or pipe delimiters in a sample of values.  Also
        handles cells that already contain Python list objects.
        A column is considered multi-label when ≥ 10 % of its values carry
        more than one label (to avoid false positives like 'Smith, John').
        """
        sample = col.dropna().head(500)
        if sample.apply(lambda x: isinstance(x, list)).any():
            return True
        str_sample = sample.astype(str)
        has_delim = str_sample.str.contains(r'[,|]', regex=True)
        if not has_delim.any():
            return False
        # Fraction of rows that have a delimiter
        return float(has_delim.mean()) > 0.10

    @staticmethod
    def _parse_multi_labels(series: pd.Series) -> List[List[str]]:
        """Split multi-label strings into lists of label strings.

        Handles:
          - Python list objects: ["pos", "urgent"] → ["pos", "urgent"]
          - Pipe-separated: "pos|urgent" → ["pos", "urgent"]
          - Comma-separated: "pos,urgent" → ["pos", "urgent"]
        """
        result = []
        for val in series:
            if isinstance(val, list):
                result.append([str(v).strip() for v in val if str(v).strip()])
            else:
                parts = [
                    v.strip()
                    for v in str(val).replace('|', ',').split(',')
                    if v.strip()
                ]
                result.append(parts)
        return result

    # ──────────────────────────────────────────────────────────────────────

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
    ) -> Tuple[pd.DataFrame, str, str, bool]:
        """
        Load and validate CSV file.

        Args:
            csv_path: Path to CSV file
            label_column: Name of label column
            text_column: Name of text column (auto-detected if None)

        Returns:
            Tuple of (dataframe, label_column, text_column, is_multi_label)

        Raises:
            ValueError: If validation fails
        """
        # Load CSV or Excel file
        file_path = Path(csv_path)
        suffix = file_path.suffix.lower()
        logger.info(f"Loading {'Excel' if suffix in ('.xlsx', '.xls') else 'CSV'} from {csv_path}")
        try:
            if suffix in ('.xlsx', '.xls'):
                try:
                    df = pd.read_excel(csv_path, engine='openpyxl' if suffix == '.xlsx' else 'xlrd')
                except ImportError as ie:
                    pkg = 'openpyxl' if suffix == '.xlsx' else 'xlrd'
                    raise ValueError(
                        f"Reading {suffix} files requires '{pkg}'. "
                        f"Install it with: pip install {pkg}"
                    ) from ie
            else:
                try:
                    df = pd.read_csv(csv_path, encoding='utf-8')
                except UnicodeDecodeError:
                    logger.warning("UTF-8 decode failed, retrying with latin-1 encoding")
                    df = pd.read_csv(csv_path, encoding='latin-1')
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Failed to load file: {str(e)}")
        
        # Validate shape
        if len(df) < self.min_samples:
            raise ValueError(f"Dataset has {len(df)} samples, minimum required: {self.min_samples}")
        
        logger.info(f"Loaded {len(df)} samples")
        
        # Validate label column exists
        if label_column not in df.columns:
            # Hint: if none of the columns look like proper names, the file
            # probably has no header row and the first data row was parsed as
            # column names — give the user an actionable message.
            cols = list(df.columns)
            looks_headerless = all(
                str(c).strip() == str(c).strip().lstrip('0123456789.-') == '' or
                (pd.to_numeric(str(c), errors='coerce') is not np.nan
                 and pd.notna(pd.to_numeric(str(c), errors='coerce')))
                for c in cols
            )
            hint = (
                " The CSV column names look like data values — the file may be "
                "missing a header row. Add a header row with column names and retry."
                if looks_headerless else ""
            )
            raise ValueError(
                f"Label column '{label_column}' not found in CSV. "
                f"Available columns: {cols}.{hint}"
            )

        # ── Detect multi-label BEFORE any dtype conversion ────────────────────
        is_multi_label = self._is_multi_label_column(df[label_column])
        if is_multi_label:
            logger.info(
                f"🏷️  Multi-label format detected in '{label_column}' "
                f"(comma/pipe separated values or list objects). "
                f"Pipeline will use BCEWithLogitsLoss + sigmoid threshold."
            )

        # Regression guard — only meaningful for single-label numeric columns.
        if not is_multi_label and pd.api.types.is_numeric_dtype(df[label_column]):
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

        # Normalise label column dtype (single-label only — multi-label keeps raw str values).
        if not is_multi_label and pd.api.types.is_float_dtype(df[label_column]):
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
        elif not is_multi_label and pd.api.types.is_integer_dtype(df[label_column]):
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

        # Guard: need at least 2 distinct atomic labels to do classification
        if is_multi_label:
            parsed = self._parse_multi_labels(df[label_column])
            all_atomic = {lbl for row in parsed for lbl in row}
            if len(all_atomic) < 2:
                raise ValueError(
                    f"Multi-label column '{label_column}' contains fewer than 2 unique labels "
                    f"across all rows. Found: {all_atomic}. "
                    f"Classification requires at least 2 distinct label values."
                )
            logger.info(f"Multi-label: {len(all_atomic)} unique atomic labels found: {sorted(all_atomic)}")
        else:
            num_classes = df[label_column].nunique()
            if num_classes < 2:
                raise ValueError(
                    f"Label column '{label_column}' contains only 1 unique class. "
                    f"Classification requires at least 2 classes."
                )
            # Verify each class still has enough samples for a stratified split
            # Skip this check for multi-label: value_counts counts combination strings
            # (e.g. "cat,dog"), not atomic labels — rare combos would falsely trigger.
            if not is_multi_label:
                min_class_count = df[label_column].value_counts().min()
                if min_class_count < 2:
                    raise ValueError(
                        f"After deduplication, at least one class has only {min_class_count} sample(s). "
                        f"Stratified train/val split requires at least 2 samples per class. "
                        f"Please collect more data or reduce the number of classes."
                    )
                # Warn when class count is high relative to data — challenging setup
                if num_classes > 20:
                    avg_spc = len(df) / num_classes
                    if avg_spc < 50:
                        logger.warning(
                            f"⚠️  {num_classes} classes with avg {avg_spc:.0f} samples/class. "
                            f"High class count + limited data is very challenging. "
                            f"Only CRITICAL/SMALL strategies will apply — consider merging "
                            f"rare classes or collecting more data."
                        )
                    elif num_classes > 50:
                        logger.warning(
                            f"⚠️  {num_classes}-class problem detected. "
                            f"BERT models can handle this but need sufficient samples per class. "
                            f"Micro-F1 will be used as the primary evaluation metric."
                        )

        # Validate text column is string type
        df[text_column] = df[text_column].astype(str)
        
        logger.info(f"Final dataset: {len(df)} samples")
        logger.info(f"Label column: {label_column}  |  multi_label={is_multi_label}")
        logger.info(f"Text column: {text_column}")

        return df, label_column, text_column, is_multi_label
    
    def detect_text_columns(self, df: pd.DataFrame, label_column: str) -> dict:
        """
        Detect ALL usable columns in the dataframe and classify them.

        Column categories:
        - text_columns: long natural-language columns (avg_length > 20, low cardinality)
        - categorical_columns: short low-cardinality string/numeric features
          (avg_length ≤ 20, ≤ 100 unique values) — e.g. region, tier, status
        - id_columns: high-cardinality columns excluded to prevent data leakage
          (uniqueness_ratio > 0.8 for short strings, or > 0.95 for any length)

        Args:
            df: DataFrame
            label_column: Label column name

        Returns:
            Dictionary with:
            - text_columns: list of text column names (sorted by avg length, desc)
            - primary_text_column: longest text column
            - categorical_columns: list of short categorical feature column names
            - id_columns: list of excluded high-cardinality column names
            - column_stats: dict with stats per included column
        """
        text_columns_info = []
        categorical_columns_info = []
        id_columns = []
        n_rows = max(len(df), 1)

        for col in df.columns:
            if col == label_column:
                continue
            # Cast all columns to str to measure character-level properties uniformly.
            try:
                text_series = df[col].astype(str)
                avg_len = float(text_series.str.len().mean())
                num_unique = int(df[col].nunique())
                uniqueness_ratio = num_unique / n_rows

                # ── HIGH-CARDINALITY EXCLUSION (data leakage guard) ──────────────
                # Flag short single-token strings with very high uniqueness as
                # ID/key columns (UUIDs, user IDs, emails, SKU codes, etc.).
                # Real text almost always contains spaces (multiple words), so we
                # require avg_words ≤ 1.3 to avoid excluding short but valid text
                # columns (e.g. a 10-row dataset where each short review is unique).
                avg_words = float(
                    text_series.str.split().str.len().mean()
                ) if len(text_series) > 0 else 1.0
                is_id_column = uniqueness_ratio > 0.85 and avg_len < 40 and avg_words <= 1.3
                if is_id_column:
                    id_columns.append(col)
                    logger.warning(
                        f"⚠️  Column '{col}' excluded — high-cardinality identifier "
                        f"({num_unique} unique values, {uniqueness_ratio*100:.0f}% uniqueness). "
                        f"Including it would risk data leakage."
                    )
                    continue

                # ── LONG TEXT COLUMNS ────────────────────────────────────────────
                if avg_len > 20:
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
                        'non_null_count': int(text_series.notna().sum()),
                        'num_unique': num_unique,
                        'kind': 'text',
                    })

                # ── SHORT CATEGORICAL COLUMNS ────────────────────────────────────
                # avg_len ≤ 20 AND at least 2 distinct values AND not near-constant
                elif avg_len > 1 and num_unique >= 2:
                    top_freq = text_series.value_counts(normalize=True).iloc[0]
                    if top_freq > 0.95:
                        logger.warning(
                            f"Column '{col}' is near-constant ({top_freq*100:.0f}% same value) "
                            f"— skipping as categorical feature (no signal)."
                        )
                        continue
                    categorical_columns_info.append({
                        'name': col,
                        'avg_length': avg_len,
                        'min_length': text_series.str.len().min(),
                        'max_length': text_series.str.len().max(),
                        'non_null_count': int(text_series.notna().sum()),
                        'num_unique': num_unique,
                        'kind': 'categorical',
                    })

            except Exception:
                logger.warning(f"Could not analyse column '{col}' for text detection, skipping")

        # ── FALLBACK: no long-text column found ──────────────────────────────────
        # Pick the column with the highest average length among non-excluded columns.
        if not text_columns_info:
            # Promote the longest categorical column (if any) to text
            if categorical_columns_info:
                categorical_columns_info.sort(key=lambda x: x['avg_length'], reverse=True)
                promoted = categorical_columns_info.pop(0)
                promoted['kind'] = 'text'
                text_columns_info.append(promoted)
                logger.warning(
                    f"No column has avg text length > 20. "
                    f"Promoted '{promoted['name']}' (avg {promoted['avg_length']:.1f} chars) "
                    f"as the primary text column. Verify this is correct."
                )
            else:
                # Absolute fallback: scan all non-label columns
                best_col, best_avg = None, 0.0
                for col in df.columns:
                    if col == label_column or col in id_columns:
                        continue
                    try:
                        avg_len = float(df[col].astype(str).str.len().mean())
                        if avg_len > best_avg:
                            best_avg = avg_len
                            best_col = col
                    except Exception:
                        pass
                if best_col is not None:
                    if best_avg < 5:
                        raise ValueError(
                            f"All columns in the dataset have very short average text length "
                            f"(best candidate '{best_col}': avg {best_avg:.1f} chars). "
                            f"A BERT encoder needs natural language or meaningful string features. "
                            f"Please provide a CSV with at least one text or descriptive column."
                        )
                    logger.warning(
                        f"No column has avg text length > 20. "
                        f"Falling back to '{best_col}' (avg {best_avg:.1f} chars) as the text column. "
                        f"Verify this is the correct column."
                    )
                    text_series = df[best_col].astype(str)
                    text_columns_info.append({
                        'name': best_col,
                        'avg_length': best_avg,
                        'min_length': text_series.str.len().min(),
                        'max_length': text_series.str.len().max(),
                        'non_null_count': int(text_series.notna().sum()),
                        'num_unique': int(df[best_col].nunique()),
                        'kind': 'text',
                    })
                else:
                    raise ValueError(
                        f"Could not find any text column. The dataset has only the label column "
                        f"'{label_column}'. Please provide a CSV with at least one text/feature column."
                    )

        # Sort results
        text_columns_info.sort(key=lambda x: x['avg_length'], reverse=True)
        categorical_columns_info.sort(key=lambda x: x['avg_length'])  # short first

        # ── Logging ──────────────────────────────────────────────────────────────
        logger.info(f"\n{'='*70}")
        logger.info("📝 COLUMN DETECTION RESULTS")
        logger.info(f"{'='*70}")
        for i, col_info in enumerate(text_columns_info, 1):
            marker = "🔵 PRIMARY TEXT" if i == 1 else "⚪ TEXT"
            logger.info(f"{marker}: '{col_info['name']}'  "
                        f"avg {col_info['avg_length']:.0f} chars | "
                        f"{col_info['num_unique']} unique values")
        for col_info in categorical_columns_info:
            logger.info(f"🟢 CATEGORICAL: '{col_info['name']}'  "
                        f"avg {col_info['avg_length']:.0f} chars | "
                        f"{col_info['num_unique']} unique values")
        if id_columns:
            logger.info(f"🔴 EXCLUDED (ID/leakage risk): {id_columns}")
        logger.info(f"{'='*70}\n")

        all_included = text_columns_info + categorical_columns_info
        return {
            'text_columns': [col['name'] for col in text_columns_info],
            'primary_text_column': text_columns_info[0]['name'],
            'categorical_columns': [col['name'] for col in categorical_columns_info],
            'id_columns': id_columns,
            'column_stats': {col['name']: col for col in all_included},
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

        # Warn about any high-cardinality columns that slipped through
        n_rows = max(len(df), 1)
        for col in text_columns:
            if col not in df.columns:
                continue
            col_series = df[col].astype(str)
            num_unique = df[col].nunique()
            uniqueness_ratio = num_unique / n_rows
            avg_len = float(col_series.str.len().mean())
            if uniqueness_ratio > 0.85 and avg_len < 40:
                logger.warning(
                    f"⚠️  Column '{col}' has {uniqueness_ratio*100:.0f}% unique values — "
                    f"merging near-unique identifiers risks data leakage. "
                    f"Remove it from text_columns if it is an ID field."
                )
        
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
                    truncated = tokenizer.decode(
                        tokens[:int(max_tokens * _KEEP_RATIO)],
                        skip_special_tokens=True,
                    )
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
