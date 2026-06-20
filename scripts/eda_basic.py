import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')  # saves charts as files instead of opening a window
import matplotlib.pyplot as plt
import seaborn as sns


def explore_dataset(train_df, test_df):
    print("\n=== COLUMNS ===")
    print(train_df.columns.tolist())

    print("\n=== DATA TYPES (train) ===")
    print(train_df.dtypes)

    print("\n=== MISSING VALUES (train) ===")
    print(train_df.isna().sum())
    print("Total missing values in train:", train_df.isna().sum().sum())
    print("Total missing values in test:", test_df.isna().sum().sum())

    print("\n=== NUMERIC SUMMARY (train) ===")
    print(train_df.describe().T)

    print("\n=== DATA STRUCTURE ===")
    train_df.info()

    print("\n=== DATA TYPES ===")
    print(train_df.dtypes)

    print("\n=== LABEL DISTRIBUTION (train) ===")
    print(train_df["label"].value_counts())

    print("\n=== PROTOCOL TYPE DISTRIBUTION (train) ===")
    print(train_df["protocol_type"].value_counts())

    print("\n=== SERVICE DISTRIBUTION (train) ===")
    print(train_df["service"].value_counts())

    print("\n=== FLAG DISTRIBUTION (train) ===")
    print(train_df["flag"].value_counts())

    print("\n=== DUPLICATE ROWS ===")
    dup_count = train_df.duplicated().sum()
    print(f"Duplicate rows in training set: {dup_count}")
    print(f"That is {dup_count / len(train_df) * 100:.2f}% of the data")

def print_dataset_shape(train_df, test_df):
    print("=== DATA SHAPES ===")
    print("Training set shape:", train_df.shape)
    print("Test set shape:", test_df.shape)

def show_binary_label_distribution(train_df):
    print("\n=== BINARY LABEL DISTRIBUTION ===")

    # Group everything as simply 'normal' or 'attack'
    train_df["binary_label"] = train_df["label"].apply(
        lambda x: "normal" if x == "normal" else "attack"
    )

    print(train_df["binary_label"].value_counts())

    print("\nAs percentages:")
    print(
        train_df["binary_label"]
        .value_counts(normalize=True)
        .mul(100)
        .round(1)
        .astype(str) + "%"
    )

def show_attack_category_breakdown(train_df):
    print("\n=== ATTACK CATEGORY BREAKDOWN ===")

    ATTACK_CATEGORIES = {
        "DoS": ["back","land","neptune","pod","smurf","teardrop",
                "apache2","udpstorm","processtable","mailbomb"],
        "Probe": ["ipsweep","nmap","portsweep","satan","mscan","saint"],
        "R2L": ["ftp_write","guess_passwd","imap","multihop","phf","spy",
                "warezclient","warezmaster","sendmail","named",
                "snmpattack","snmpguess","xlock","xsnoop","worm"],
        "U2R": ["buffer_overflow","loadmodule","perl","rootkit",
                "httptunnel","ps","sqlattack","xterm"],
    }

    def get_category(label):
        for cat, attacks in ATTACK_CATEGORIES.items():
            if label in attacks:
                return cat
        return "normal" if label == "normal" else "Other"

    train_df["attack_category"] = train_df["label"].apply(get_category)

    print(train_df["attack_category"].value_counts())

def numeric_feature_stats(train_df):
    print("\n=== NUMERIC FEATURE STATS ===")
    numeric_cols = train_df.select_dtypes(include=np.number).columns.tolist()
    print(f"Number of numeric features: {len(numeric_cols)}")

    stats = train_df[numeric_cols].describe().T
    stats["skewness"] = train_df[numeric_cols].skew()
    stats["zeros_%"]  = ((train_df[numeric_cols] == 0).sum() / len(train_df) * 100).round(1)

    # Show only the most useful columns
    print(stats[["mean", "std", "min", "max", "skewness", "zeros_%"]].to_string())

def feature_to_drop(train_df) :
    print("\n=== FEATURES TO DROP — CONSTANT OR NEAR CONSTANT ===")

    numeric_cols = train_df.select_dtypes(include=np.number).columns.tolist()

    drop_list = []
    investigate_list = []

    for col in numeric_cols:
        zeros_pct = (train_df[col] == 0).sum() / len(train_df) * 100
        n_unique = train_df[col].nunique()

        if n_unique == 1:
            drop_list.append((col, zeros_pct, "CONSTANT — never changes"))
        elif zeros_pct >= 99:
            drop_list.append((col, zeros_pct, "99%+ zeros — carries almost no information"))
        elif zeros_pct >= 95:
            investigate_list.append((col, zeros_pct, "95-99% zeros — investigate"))

    print(f"\nFeatures to DROP ({len(drop_list)} found):")
    for col, zpct, reason in drop_list:
        print(f"  {col:<30} zeros: {zpct:.1f}%  →  {reason}")

    print(f"\nFeatures to INVESTIGATE ({len(investigate_list)} found):")
    for col, zpct, reason in investigate_list:
        print(f"  {col:<30} zeros: {zpct:.1f}%  →  {reason}")

def variable_correlation(train_df):
    print("\n=== STEP 6 — CORRELATION ANALYSIS ===")
    # Exclude non-original or non-network features
    exclude = ["label", "difficulty_level", "binary_label", "attack_category"]
    numeric_cols = train_df.select_dtypes(include=np.number).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in exclude]

    print(f"Running correlation on {len(numeric_cols)} features\n")

    # Build the correlation matrix
    corr_matrix = train_df[numeric_cols].corr()

    # Find all pairs above 0.90 threshold
    threshold = 0.90
    corr_pairs = []

    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            val = corr_matrix.iloc[i, j]
            if abs(val) >= threshold:
                corr_pairs.append((numeric_cols[i], numeric_cols[j], round(val, 4)))

    # Sort by highest correlation first
    corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    print(f"Pairs with correlation >= {threshold}: {len(corr_pairs)}\n")
    for a, b, val in corr_pairs:
        print(f"  {a:<35} ↔  {b:<35}  {val}")

def unique_values_label (train_df):
    print("\n=== UNIQUE VALUES IN 'label' COLUMN ===")
    unique_labels = train_df["label"].unique()
    print(unique_labels)

def features_correlation_with_label(train_df):
    

    show_binary_label_distribution(train_df)

    print("\n=== FEATURE VS LABEL CORRELATION ===")
    # Convert binary label to numeric: normal = 0, attack = 1
    train_df["label_numeric"] = train_df["binary_label"].apply(
        lambda x: 1 if x == "attack" else 0
    )

    # Exclude non-network columns
    exclude = ["label", "difficulty_level", "binary_label", 
            "attack_category", "label_numeric"]
    numeric_cols = train_df.select_dtypes(include=np.number).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in exclude]

    # Calculate correlation of each feature against the binary label
    correlations = train_df[numeric_cols].corrwith(train_df["label_numeric"])

    # Sort by absolute value — strongest relationship first
    correlations = correlations.reindex(
        correlations.abs().sort_values(ascending=False).index
    )

    print(f"\n  {'Feature':<35} {'Correlation':>12}  Strength")
    print("  " + "-" * 70)

    for feat, val in correlations.items():
        if abs(val) >= 0.3:
            strength = "STRONG"
        elif abs(val) >= 0.1:
            strength = "MODERATE"
        else:
            strength = "WEAK"
        direction = "increases with attacks" if val > 0 else "decreases with attacks"
        print(f"  {feat:<35} {val:>12.4f}  {strength} — {direction}")

def attack_category_distribution_chart(train_df, charts_dir):
    show_attack_category_breakdown(train_df)

    cat_counts = train_df["attack_category"].value_counts()

    plt.figure(figsize=(8, 5))
    colors = ["#2ecc71", "#e74c3c", "#3498db", "#f39c12", "#9b59b6"]
    cat_counts.plot(kind="bar", color=colors[:len(cat_counts)], edgecolor="black")
    plt.title("Attack Category Distribution", fontsize=14, fontweight="bold")
    plt.xlabel("Category")
    plt.ylabel("Number of Records")
    plt.xticks(rotation=0)
    for i, v in enumerate(cat_counts):
        plt.text(i, v + 200, f"{v:,}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(charts_dir / "01_attack_category_distribution.png", dpi=150)
    plt.clf()
    plt.close()
    print("  Chart 1 saved — attack category distribution")

def Correlation_heatmap_chart(train_df, charts_dir):
    exclude = ["label", "difficulty_level", "binary_label",
           "attack_category", "label_numeric"]
    
    numeric_cols = train_df.select_dtypes(include=np.number).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in exclude]

    corr_matrix = train_df[numeric_cols].corr()

    plt.figure(figsize=(20, 16))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=False,
        cmap="coolwarm",
        center=0,
        linewidths=0.3,
        vmin=-1, vmax=1
    )
    plt.title("Feature Correlation Heatmap", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(charts_dir / "02_correlation_heatmap.png", dpi=150)
    plt.clf()
    plt.close()
    print("  Chart 2 saved — correlation heatmap")

def Correlation_Strongfeatures_with_label_chart(train_df, charts_dir):

    show_binary_label_distribution(train_df)

    # Updated to include ALL strong features (absolute correlation >= 0.30)
    key_features = [
        "same_srv_rate",            # -0.75 strongest negative
        "dst_host_srv_count",       # -0.72
        "dst_host_same_srv_rate",   # -0.69
        "logged_in",                # -0.69
        "dst_host_srv_serror_rate", # +0.65
        "dst_host_serror_rate",     # +0.65
        "serror_rate",              # +0.65
        "srv_serror_rate",          # +0.65
        "count",                    # +0.57
    ]

    # 3 rows x 3 columns to fit 9 features
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    axes = axes.flatten()

    for i, feat in enumerate(key_features):
        ax = axes[i]
        cap = train_df[feat].quantile(0.99)
        normal = train_df.loc[train_df["binary_label"] == "normal", feat].clip(upper=cap)
        attack = train_df.loc[train_df["binary_label"] == "attack", feat].clip(upper=cap)

        ax.boxplot(
            [normal, attack],
            labels=["Normal", "Attack"],
            patch_artist=True,
            boxprops=dict(facecolor="lightblue"),
            medianprops=dict(color="red", linewidth=2)
        )
        ax.set_title(f"{feat}", fontsize=10, fontweight="bold")
        ax.set_ylabel("Value")

    plt.suptitle("Strong Features — Normal vs Attack Traffic",
                fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(charts_dir / "03_key_features_boxplots.png", dpi=150)
    plt.clf()
    plt.close()
    print("  Chart 3 saved — key features boxplots")
    print(f"\n  All charts saved to: {charts_dir}")



def Correlation_allFeatures_with_label_chart(train_df, charts_dir):

    features_correlation_with_label(train_df)

    # Define numeric_cols inside this function
    exclude = ["label", "difficulty_level", "binary_label",
               "attack_category", "label_numeric"]
    numeric_cols = train_df.select_dtypes(include=np.number).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in exclude]

    correlations = train_df[numeric_cols].corrwith(train_df["label_numeric"])
    correlations = correlations.reindex(
        correlations.abs().sort_values(ascending=False).index
    )
    # Remove nan (num_outbound_cmds)
    correlations = correlations.dropna()

    colors = []
    for val in correlations.values:
        if abs(val) >= 0.3:
            colors.append("#e74c3c")   # red = strong
        elif abs(val) >= 0.1:
            colors.append("#f39c12")   # orange = moderate
        else:
            colors.append("#95a5a6")   # grey = weak

    plt.figure(figsize=(14, 10))
    plt.barh(correlations.index, correlations.values, color=colors, edgecolor="black")
    plt.axvline(x=0.3,  color="red",    linestyle="--", linewidth=1, label="Strong (0.30)")
    plt.axvline(x=-0.3, color="red",    linestyle="--", linewidth=1)
    plt.axvline(x=0.1,  color="orange", linestyle="--", linewidth=1, label="Moderate (0.10)")
    plt.axvline(x=-0.1, color="orange", linestyle="--", linewidth=1)
    plt.axvline(x=0,    color="black",  linestyle="-",  linewidth=0.8)
    plt.xlabel("Correlation with Label (attack=1, normal=0)")
    plt.title("All Features — Correlation with Binary Label",
              fontsize=14, fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(charts_dir / "04_feature_label_correlation.png", dpi=150)
    plt.clf()
    plt.close()
    print("  Chart saved — all features correlation with label")

#function to save a sample of the dataset as Excel for viewing as a formatted table
def preview_dataset(df: pd.DataFrame, filename: str, base_dir: Path,
                    sample: int = 1000) -> None:
    """
    Save a sample of the dataset as Excel for viewing as a formatted table.
    Always overwrites existing files.
    """
    preview_dir = base_dir / "dataset" / "processed"
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

def main() -> None:
    # Path to processed datasets

    base_dir = Path(__file__).resolve().parent.parent
    
    processed_dir = base_dir / "dataset" / "processed"

    train_path = processed_dir / "train_with_headers.csv"
    test_path = processed_dir / "test_with_headers.csv"
    charts_dir = base_dir / "dataset" / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)


    preview_dataset(pd.read_csv(train_path), "train_KDD.csv", base_dir)
    preview_dataset(pd.read_csv(test_path), "test_KDD.csv", base_dir)



    # Load processed datasets
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print("this is the training dataset shape: " + str(train_df.shape))
    print("this is the testing dataset shape: " + str(test_df.shape))
    #print(test_df["logged_in"].value_counts(normalize=True))

    #function to explore the dataset
    #explore_dataset(train_df, test_df)
    
    #function to print basic info about the dataset
    print_dataset_shape(train_df, test_df)

    #function to show binary label distribution 
    show_binary_label_distribution(train_df)

    #function to show attack category breakdown
    show_attack_category_breakdown(train_df)
     
    #function to show numeric feature stats
    numeric_feature_stats(train_df)

    #function to show features to drop
    #feature_to_drop(train_df)

    #function to show variable correlation
    #variable_correlation(train_df)

    #function to show unique values in label column
    #unique_values_label (train_df)

    #function to show features correlation with label
    #features_correlation_with_label(train_df)

    #function to show attack category distribution plot
    attack_category_distribution_chart(train_df, charts_dir)

    #function to show correlation heatmap chart between features
    Correlation_heatmap_chart(train_df, charts_dir)
    
    #function to show correlation between strong features and label
    Correlation_Strongfeatures_with_label_chart(train_df, charts_dir)

    #function to show correlation between all features and label
    Correlation_allFeatures_with_label_chart(train_df, charts_dir)
   
    
   
if __name__ == "__main__":
    main()
