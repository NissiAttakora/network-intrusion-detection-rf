import pandas as pd
from pathlib import Path

# Column names for NSL-KDD (41 features + label + difficulty_level)
COLUMNS = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
    "wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root","num_file_creations",
    "num_shells","num_access_files","num_outbound_cmds","is_host_login",
    "is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
    "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
    "srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
    "dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label",              # attack type
    "difficulty_level"    # difficulty score (not used for training)
]

def get_dataset_paths() -> tuple[Path, Path]:
    """
    Build and return the full paths to the NSL-KDD training and test files.
    Returns:
        (train_path, test_path): Paths to KDDTrain+.txt and KDDTest+.txt.
    """
    base_dir = Path(__file__).resolve().parent.parent
    dataset_dir = base_dir / "dataset"
    train_path = dataset_dir / "KDDTrain+.txt"
    test_path = dataset_dir / "KDDTest+.txt"
    return train_path, test_path

def save_processed_datasets(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dataset_dir: Path
) -> None:
    """
    Save processed NSL-KDD datasets with column headers for reuse.
    """
    processed_dir = dataset_dir / "processed"
    processed_dir.mkdir(exist_ok=True)

    train_df.to_csv(processed_dir / "train_with_headers.csv", index=False)
    test_df.to_csv(processed_dir / "test_with_headers.csv", index=False)

def load_nsl_kdd(save_processed: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the NSL-KDD training and test datasets with correct column names.
    saves processed CSV files with headers.

    Returns:
        train_df
        test_df
    """
    train_path, test_path = get_dataset_paths()
    dataset_dir = train_path.parent

    # Load without headers, then assign the expected schema
    train_df = pd.read_csv(train_path, header=None)
    test_df = pd.read_csv(test_path, header=None)

    # Validate column counts
    if train_df.shape[1] != len(COLUMNS):
        raise ValueError(
            f"Expected {len(COLUMNS)} columns, but got {train_df.shape[1]} "
            f"for training data."
        )

    if test_df.shape[1] != len(COLUMNS):
        raise ValueError(
            f"Expected {len(COLUMNS)} columns, but got {test_df.shape[1]} "
            f"for test data."
        )

    # Assign column names
    train_df.columns = COLUMNS
    test_df.columns = COLUMNS

    # Save processed versions with headers
    if save_processed:
        save_processed_datasets(train_df, test_df, dataset_dir)

    return train_df, test_df

def main() -> None:
    """
    Diagnostic entry point to verify NSL-KDD dataset loading and processing.
    """
    print("Loading NSL-KDD training and testing datasets...\n")
    train_df, test_df = load_nsl_kdd()

    print("✅ Loaded and saved successfully!")
    print("Training data shape:", train_df.shape)
    print("Testing data shape:", test_df.shape)

    print("\n🔹 First 5 rows of training data:")
    print(train_df.head())

    print("\n🔹 Label distribution (Top 10):")
    print(train_df["label"].value_counts().head(10))

    print("\n🔹 Column names:")
    print(train_df.columns.tolist())

if __name__ == "__main__":
    main()
