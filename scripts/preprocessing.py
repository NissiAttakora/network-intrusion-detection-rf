"""
============================================================
  IDS Project — Preprocessing
  Prepares the NSL-KDD dataset for model training
============================================================

HOW TO RUN:
    source ~/IDS_Project/ids_env/bin/activate
    python scripts/preprocessing.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# ─────────────────────────────────────────────────────────
# FEATURES TO DROP — decided during EDA
# ─────────────────────────────────────────────────────────
# Reason 1 — constant or near constant
CONSTANT_FEATURES = [
    "land",                # 100% zeros
    "urgent",              # 100% zeros
    "num_shells",          # 100% zeros
    "num_outbound_cmds",   # 100% zeros — truly constant
    "is_host_login",       # 100% zeros
    "num_failed_logins",   # 99.9% zeros
    "root_shell",          # 99.9% zeros — Step 8 confirmed weak
    "su_attempted",        # 99.9% zeros
    "num_file_creations",  # 99.8% zeros
    "num_access_files",    # 99.7% zeros
    "num_root",            # 99.5% zeros
    "wrong_fragment",      # 99.1% zeros
    "is_guest_login",      # 99.1% zeros
    "difficulty_level",    # NSL-KDD metadata
]

# Reason 2 — highly correlated duplicates
CORRELATED_FEATURES = [
    "srv_serror_rate",          # duplicate of serror_rate (0.993)
    "dst_host_serror_rate",     # duplicate of serror_rate (0.979)
    "dst_host_srv_serror_rate", # duplicate of serror_rate (0.981)
    "srv_rerror_rate",          # duplicate of rerror_rate (0.989)
    "dst_host_rerror_rate",     # duplicate of rerror_rate (0.927)
    "dst_host_srv_rerror_rate", # duplicate of rerror_rate (0.970)
]

# Reason 3 — weak label correlation
WEAK_FEATURES = [
    "hot",             # -0.013
    "num_compromised", # -0.010
    "src_bytes",       # +0.006
    "dst_bytes",       # +0.004
    "srv_count",       # +0.001
]

# Combined drop list
ALL_FEATURES_TO_DROP = CONSTANT_FEATURES + CORRELATED_FEATURES + WEAK_FEATURES

# Categorical features that need encoding
CATEGORICAL_FEATURES = ["protocol_type", "service", "flag"]


#function to save the cleaned dataset to disk
def save_clean_dataset(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    base_dir: Path
) -> None:
    """
    Save the cleaned and preprocessed datasets to disk
    ready for model training.
    """

    save_dir = base_dir / "dataset" / "clean_dataset"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save all 4 files
    X_train.to_csv(save_dir / "X_train.csv", index=False)
    y_train.to_csv(save_dir / "y_train.csv", index=False)
    X_test.to_csv(save_dir / "X_test.csv",   index=False)
    y_test.to_csv(save_dir / "y_test.csv",   index=False)

    print("\n=== CLEAN DATASET SAVED ===")
    print(f"  Location : {save_dir}")
    print(f"\n  Files saved:")
    print(f"    X_train.csv → {X_train.shape[0]:,} rows × {X_train.shape[1]} columns")
    print(f"    y_train.csv → {y_train.shape[0]:,} labels")
    print(f"    X_test.csv  → {X_test.shape[0]:,} rows  × {X_test.shape[1]} columns")
    print(f"    y_test.csv  → {y_test.shape[0]:,} labels")

#function to display a preview of the dataset in VS Code
def preview_dataset(df: pd.DataFrame, filename: str, base_dir: Path,
                    sample: int = 1000) -> None:
    """
    Save a sample of the dataset as Excel for viewing as a formatted table.
    Always overwrites existing files.
    """
    preview_dir = base_dir / "logs" / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    # Save as Excel instead of CSV
    filepath = preview_dir / filename.replace(".csv", ".xlsx")

    # Delete existing file first to force overwrite
    if filepath.exists():
        filepath.unlink()
        print(f"  Deleted existing file: {filepath.name}")

    # Take a sample to keep file size manageable
    df_sample = df.head(sample)
    df_sample.to_excel(filepath, index=False)

    print(f"\n=== PREVIEW SAVED ===")
    print(f"  File     : {filepath}")
    print(f"  Showing  : {sample} rows out of {df.shape[0]:,} total")
    print(f"  Columns  : {df.shape[1]}")
    print(f"  → Open this file in Excel or VS Code to view as a table")

#function to load the dataset
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the processed NSL-KDD training and test datasets.
    """
    base_dir = Path(__file__).resolve().parent.parent
    processed_dir = base_dir / "dataset" / "processed"

    train_path = processed_dir / "train_with_headers.csv"
    test_path  = processed_dir / "test_with_headers.csv"

    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    print("=== DATA LOADED ===")
    print(f"Training set : {train_df.shape[0]:,} rows × {train_df.shape[1]} columns")
    print(f"Test set     : {test_df.shape[0]:,} rows × {test_df.shape[1]} columns")

    return train_df, test_df

#function to drop features from the dataset
def drop_features(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Drop all features identified during EDA as useless,
    redundant or weakly correlated with the label.
    """
    # Only drop columns that actually exist in the dataframe
    cols_to_drop = [col for col in ALL_FEATURES_TO_DROP if col in df.columns]
    
    df = df.drop(columns=cols_to_drop)
    
    print(f"\n=== DROP FEATURES ({name}) ===")
    print(f"  Dropped  : {len(cols_to_drop)} features")
    print(f"  Remaining: {df.shape[1]} columns")
    print(f"  Shape    : {df.shape}")
    
    return df

#fuction to create binary label for the dataset
def create_binary_label(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Convert the multi-class label column into a binary label.
    normal = 0
    attack = 1
    """
    df["label"] = df["label"].apply(lambda x: 0 if x == "normal" else 1)

    print(f"\n=== BINARY LABEL ({name}) ===")
    counts = df["label"].value_counts()
    for label, count in counts.items():
        meaning = "normal" if label == 0 else "attack"
        print(f"  {meaning} ({label}) : {count:,}  ({count/len(df)*100:.1f}%)")

    return df

#function to display categorical features and their unique values
def inspect_categorical_features(df: pd.DataFrame, name: str) -> None:
    """
    Display all categorical features, their unique values
    and how many times each value appears.
    """
    print(f"\n=== CATEGORICAL FEATURES ({name}) ===")

    cat_cols = df.select_dtypes(include="object").columns.tolist()
    print(f"  Categorical columns found: {len(cat_cols)} → {cat_cols}\n")

    for col in cat_cols:
        val_counts = df[col].value_counts()
        print(f"  [{col}] — {val_counts.nunique()} unique values:")
        for val, count in val_counts.items():
            pct = count / len(df) * 100
            print(f"    {str(val):<25} : {count:>7,}  ({pct:.1f}%)")
        print()

#function to encode categorical features using One Hot Encoding
def encode_categorical_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    One Hot Encode categorical features.
    Fit on training set and apply to both train and test.
    Missing columns in test set are filled with 0.
    """

    print("\n=== ENCODE CATEGORICAL FEATURES ===")
    print(f"  Columns before encoding: {train_df.shape[1]}")

    # Apply One Hot Encoding to both datasets
    train_df = pd.get_dummies(train_df, columns=CATEGORICAL_FEATURES)
    test_df  = pd.get_dummies(test_df,  columns=CATEGORICAL_FEATURES)

    # Align test set to match training set exactly
    # missing columns → filled with 0
    # extra columns   → dropped
    test_df = test_df.reindex(columns=train_df.columns, fill_value=0)

    print(f"  Columns after encoding : {train_df.shape[1]}")
    print(f"  New columns added      : {train_df.shape[1] - 18}")
    print(f"\n  Training set shape     : {train_df.shape}")
    print(f"  Test set shape         : {test_df.shape}")
    print(f"\n Both datasets now have identical columns")

    return train_df, test_df

#function to scale numeric features using MinMaxScaler
def scale_numeric_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scale all numeric features to range 0-1 using MinMaxScaler.
    Fit scaler on training set only, apply to both train and test.
    One Hot Encoded columns (0/1) are excluded from scaling
    as they are already in the correct range.
    """

    print("\n=== SCALE NUMERIC FEATURES ===")

    # Automatically identify numeric columns
    # Exclude label and One Hot Encoded columns
    # One Hot Encoded columns only contain 0 or 1
    # so we exclude any column where max value is 1 
    # AND min value is 0 AND only has 2 unique values
    exclude = ["label"]
    
    numeric_cols = []
    for col in train_df.select_dtypes(include=np.number).columns:
        if col in exclude:
            continue
        # Skip One Hot Encoded columns (only contain 0 and 1)
        if train_df[col].nunique() == 2 and train_df[col].max() == 1 and train_df[col].min() == 0:
            continue
        numeric_cols.append(col)

    print(f"  Numeric features found : {len(numeric_cols)}")
    print(f"  Features to scale      : {numeric_cols}")

    # Fit scaler on training set ONLY
    scaler = MinMaxScaler()
    scaler.fit(train_df[numeric_cols])

    # Apply to both training and test sets
    train_df[numeric_cols] = scaler.transform(train_df[numeric_cols])
    test_df[numeric_cols]  = scaler.transform(test_df[numeric_cols])

    # Verify scaling worked
    print(f"\n  Sample feature ranges after scaling:")
    for col in numeric_cols[:4]:
        min_val = train_df[col].min()
        max_val = train_df[col].max()
        print(f"    {col:<35} min: {min_val:.4f}  max: {max_val:.4f}")

    print(f"\n  Training set shape : {train_df.shape}")
    print(f"  Test set shape     : {test_df.shape}")

    return train_df, test_df, scaler


def split_features_and_label(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Separate features (X) from the label (y) for both
    training and test sets.

    X = all columns except label (input features)
    y = label column only (what we want to predict)
    """

    print("\n=== SPLIT FEATURES AND LABEL ===")

    # Separate features from label
    X_train = train_df.drop(columns=["label"])
    y_train = train_df["label"]

    X_test  = test_df.drop(columns=["label"])
    y_test  = test_df["label"]

    print(f"  X_train shape : {X_train.shape}  (features only)")
    print(f"  y_train shape : {y_train.shape}  (labels only)")
    print(f"  X_test shape  : {X_test.shape}")
    print(f"  y_test shape  : {y_test.shape}")

    print(f"\n  y_train distribution:")
    for label, count in y_train.value_counts().items():
        meaning = "normal" if label == 0 else "attack"
        print(f"    {meaning} ({label}) : {count:,}  ({count/len(y_train)*100:.1f}%)")

    print(f"\n  y_test distribution:")
    for label, count in y_test.value_counts().items():
        meaning = "normal" if label == 0 else "attack"
        print(f"    {meaning} ({label}) : {count:,}  ({count/len(y_test)*100:.1f}%)")

    return X_train, y_train, X_test, y_test


# Placeholder for the preprocessing steps to be implemented
def main() -> None:

    base_dir = Path(__file__).resolve().parent.parent

    # Step 1 — Load data
    train_df, test_df = load_data()

    print("unique protocol types in the training set: " + str(train_df["protocol_type"].unique()))
    print("unique services in the training set: " + str(train_df["service"].unique()))
    print("unique flags in the training set: " + str(train_df["flag"].unique()))

    # Step - features to drop
    train_df = drop_features(train_df, "train") 
    test_df  = drop_features(test_df,  "test")

    # Step 3 — convert multi-class label to binary
    train_df = create_binary_label(train_df, "train")
    test_df  = create_binary_label(test_df,  "test")

    # Step 4 — Inspect categorical features before encoding
    inspect_categorical_features(train_df, "train")
    inspect_categorical_features(test_df,  "test")

    # Step 5 — Encode categorical features
    train_df_encoded, test_df_encoded = encode_categorical_features(train_df, test_df)

    # Step 6 — Scale numeric features
    train_df_scaled, test_df_scaled, scaler = scale_numeric_features(
        train_df_encoded, test_df_encoded
    )

    # Step 7 — Split into X and y
    X_train, y_train, X_test, y_test = split_features_and_label(
        train_df_scaled, test_df_scaled
    )

    # Step 8 — Save clean dataset
    save_clean_dataset(X_train, y_train, X_test, y_test, base_dir)

   

    #print(f"Original Training set shape : {train_df.shape[0]:,} rows × {train_df.shape[1]} columns")
    #print(f"Original Test set     : {test_df.shape[0]:,} rows × {test_df.shape[1]} columns")

    #stastics of the dataset after scaling  
    #print("\n=== SCALED TRAINING SET STATS ===")
    #print(train_df_scaled.describe().T)

    #print("\n=== SCALED TEST SET STATS ===")
    #print(test_df_scaled.describe().T)


    #preview dataset
    #preview_dataset(train_df_scaled, "train_scaled_preview.csv", base_dir)
    #preview_dataset(test_df_scaled,  "test_scaled_preview.csv",  base_dir)


    #print("training dataset set columns: " + str(train_df.columns))
    #print("testing dataset set columns: " + str(test_df.columns))
   

if __name__ == "__main__":
    main()
