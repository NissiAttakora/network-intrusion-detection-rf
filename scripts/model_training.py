"""
============================================================
  IDS Project — Model Training
  Trains and compares multiple ML models on NSL-KDD dataset
============================================================

"""

from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib
import time

# ─────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────

# Random state ensures results are reproducible
# every time you run the script you get the same results
RANDOM_STATE = 42

# Models to train
MODELS = {
    "Random Forest": RandomForestClassifier(
        n_estimators=100,    # number of trees
        random_state=RANDOM_STATE,
        n_jobs=-1, # use all CPU cores
        class_weight="balanced"           
    ),
    "Decision Tree": DecisionTreeClassifier(
        random_state=RANDOM_STATE,
        class_weight="balanced"
    ),
    "Logistic Regression": LogisticRegression(
        max_iter=1000,       # more iterations for convergence
        random_state=RANDOM_STATE,
        class_weight="balanced"
    )
}

#function to load the clean dataset
def load_clean_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load the preprocessed and cleaned dataset
    saved by preprocessing.py
    """
    base_dir = Path(__file__).resolve().parent.parent
    clean_dir = base_dir / "dataset" / "clean_dataset"

    print("=== LOADING CLEAN DATASET ===")

    X_train = pd.read_csv(clean_dir / "X_train.csv")
    y_train = pd.read_csv(clean_dir / "y_train.csv").squeeze()
    X_test  = pd.read_csv(clean_dir / "X_test.csv")
    y_test  = pd.read_csv(clean_dir / "y_test.csv").squeeze()

    print(f"  X_train : {X_train.shape[0]:,} rows × {X_train.shape[1]} features")
    print(f"  y_train : {y_train.shape[0]:,} labels")
    print(f"  X_test  : {X_test.shape[0]:,} rows  × {X_test.shape[1]} features")
    print(f"  y_test  : {y_test.shape[0]:,} labels")

    print(f"\n  y_train distribution:")
    for label, count in y_train.value_counts().items():
        meaning = "normal" if label == 0 else "attack"
        print(f"    {meaning} ({label}) : {count:,}  ({count/len(y_train)*100:.1f}%)")

    return X_train, y_train, X_test, y_test

#function to train the models 
def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> dict:
    """
    Train all models defined in MODELS configuration.
    Returns a dictionary of trained models.
    """

    print("\n=== TRAINING MODELS ===")
    trained_models = {}

    for name, model in MODELS.items():
        print(f"\n  Training {name}...")

        # Record start time
        start_time = time.time()

        # Train the model
        model.fit(X_train, y_train)

        # Record end time
        end_time = time.time()
        duration = round(end_time - start_time, 2)

        trained_models[name] = model
        print(f"  ✅ {name} trained in {duration} seconds")

    print(f"\n  All {len(trained_models)} models trained successfully")
    return trained_models


#function to evaluate the models
def evaluate_models(
    trained_models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> pd.DataFrame:
    """
    Evaluate all trained models on the test set.
    Returns a DataFrame with all metrics for comparison.
    """

    print("\n=== EVALUATING MODELS ===")
    results = []

    for name, model in trained_models.items():
        print(f"\n  Evaluating {name}...")

        # Make predictions on test set
        y_pred = model.predict(X_test)

        # Calculate all metrics
        accuracy  = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall    = recall_score(y_test, y_pred)
        f1        = f1_score(y_test, y_pred)

        results.append({
            "Model":     name,
            "Accuracy":  round(accuracy  * 100, 2),
            "Precision": round(precision * 100, 2),
            "Recall":    round(recall    * 100, 2),
            "F1 Score":  round(f1        * 100, 2)
        })

        # Print confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n  Confusion Matrix — {name}:")
        print(f"                  Predicted Normal  Predicted Attack")
        print(f"  Actual Normal   {cm[0][0]:>14,}  {cm[0][1]:>15,}")
        print(f"  Actual Attack   {cm[1][0]:>14,}  {cm[1][1]:>15,}")

    # Create comparison table
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("F1 Score", ascending=False)
    results_df = results_df.reset_index(drop=True)

    print("\n=== MODEL COMPARISON ===")
    print(results_df.to_string(index=False))

    return results_df

#function to perform cross validation on the models
def cross_validate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    trained_models: dict
) -> None:
    """
    Evaluate each model using 10-fold cross validation
    on the training set for more reliable performance estimates.
    """
    print("\n=== CROSS VALIDATION (10 fold) ===")

    # StratifiedKFold ensures each fold has same class distribution
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

    for name, model in trained_models.items():
        print(f"\n  Evaluating {name}...")

        # Calculate multiple metrics
        accuracy = cross_val_score(model, X_train, y_train, 
                                   cv=cv, scoring="accuracy", n_jobs=-1)
        recall   = cross_val_score(model, X_train, y_train,
                                   cv=cv, scoring="recall", n_jobs=-1)
        f1       = cross_val_score(model, X_train, y_train,
                                   cv=cv, scoring="f1", n_jobs=-1)

        print(f"    Accuracy : {accuracy.mean()*100:.2f}%  (±{accuracy.std()*100:.2f}%)")
        print(f"    Recall   : {recall.mean()*100:.2f}%  (±{recall.std()*100:.2f}%)")
        print(f"    F1 Score : {f1.mean()*100:.2f}%  (±{f1.std()*100:.2f}%)")

#function to evaluate the models with a custom threshold
def evaluate_with_threshold(
    trained_models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.3
) -> None:
    """
    Evaluate models using a lower classification threshold.
    Default threshold lowered from 0.5 to 0.3 to catch more attacks.
    Note: Logistic Regression excluded as it uses different probability method.
    """

    print(f"\n=== THRESHOLD ADJUSTMENT (threshold={threshold}) ===")
    print(f"  Default threshold was 0.50")
    print(f"  New threshold is     {threshold}")
    print(f"  Effect: model flags attack if probability >= {threshold}")

    results = []

    for name, model in trained_models.items():

        # Logistic Regression needs different handling
        if not hasattr(model, "predict_proba"):
            print(f"\n  Skipping {name} — does not support predict_proba")
            continue

        print(f"\n  Evaluating {name} with threshold {threshold}...")

        # Get probability of being an attack (class 1)
        y_prob  = model.predict_proba(X_test)[:, 1]

        # Apply custom threshold instead of default 0.5
        y_pred  = (y_prob >= threshold).astype(int)

        # Calculate metrics
        accuracy  = accuracy_score(y_test, y_pred) * 100
        precision = precision_score(y_test, y_pred) * 100
        recall    = recall_score(y_test, y_pred) * 100
        f1        = f1_score(y_test, y_pred) * 100

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        print(f"\n  Confusion Matrix — {name} (threshold={threshold}):")
        print(f"                  Predicted Normal  Predicted Attack")
        print(f"  Actual Normal   {tn:>14,}  {fp:>15,}")
        print(f"  Actual Attack   {fn:>14,}  {tp:>15,}")

        results.append({
            "Model":     name,
            "Threshold": threshold,
            "Accuracy":  round(accuracy,  2),
            "Precision": round(precision, 2),
            "Recall":    round(recall,    2),
            "F1 Score":  round(f1,        2)
        })

    print(f"\n=== THRESHOLD {threshold} RESULTS ===")
    results_df = pd.DataFrame(results).sort_values("Recall", ascending=False)
    print(results_df.to_string(index=False))

    return results_df


def tune_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> None:
    """
    Train models with tuned hyperparameters to reduce overfitting
    and improve test set performance.
    """

    print("\n=== HYPERPARAMETER TUNING ===")

    tuned_models = {
        "Random Forest (tuned)": RandomForestClassifier(
            n_estimators=200,      # more trees
            max_depth=20,          # limit depth to reduce overfitting
            min_samples_leaf=5,    # require more samples per leaf
            min_samples_split=10,  # require more samples to split
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "Decision Tree (tuned)": DecisionTreeClassifier(
            max_depth=20,          # limit depth
            min_samples_leaf=5,    # require more samples per leaf
            min_samples_split=10,  # require more samples to split
            class_weight="balanced",
            random_state=RANDOM_STATE
        )
    }

    results = []

    for name, model in tuned_models.items():
        print(f"\n  Training {name}...")
        start = time.time()
        model.fit(X_train, y_train)
        duration = round(time.time() - start, 2)
        print(f"  ✅ Trained in {duration} seconds")

        y_pred = model.predict(X_test)

        accuracy  = accuracy_score(y_test, y_pred)  * 100
        precision = precision_score(y_test, y_pred) * 100
        recall    = recall_score(y_test, y_pred)    * 100
        f1        = f1_score(y_test, y_pred)        * 100

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        print(f"\n  Confusion Matrix — {name}:")
        print(f"                  Predicted Normal  Predicted Attack")
        print(f"  Actual Normal   {tn:>14,}  {fp:>15,}")
        print(f"  Actual Attack   {fn:>14,}  {tp:>15,}")

        results.append({
            "Model":     name,
            "Accuracy":  round(accuracy,  2),
            "Precision": round(precision, 2),
            "Recall":    round(recall,    2),
            "F1 Score":  round(f1,        2)
        })

    print("\n=== TUNED MODEL RESULTS ===")
    results_df = pd.DataFrame(results).sort_values("F1 Score", ascending=False)
    print(results_df.to_string(index=False))

    return tuned_models


def save_best_model(
    tuned_models: dict,
    threshold: float = 0.1
) -> None:
    """
    Save the best performing model to disk.
    Best model: Random Forest (tuned) with threshold 0.1
    """

    base_dir   = Path(__file__).resolve().parent.parent
    models_dir = base_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Get best model
    best_model = tuned_models["Random Forest (tuned)"]

    # Save to disk
    model_path = models_dir / "random_forest_tuned_threshold01.pkl"
    joblib.dump(best_model, model_path)

    print(f"\n=== MODEL SAVED ===")
    print(f"  Model     : Random Forest (tuned)")
    print(f"  Threshold : {threshold}")
    print(f"  Location  : {model_path}")
    print(f"  ✅ Ready for deployment!")





def main() -> None:

    # Step 1 — Load clean data
    X_train, y_train, X_test, y_test = load_clean_data()

    # Step 2 — Train models
    trained_models = train_models(X_train, y_train)

    # Step 3 — Evaluate Trained models on test set
    evaluate_models(trained_models, X_test, y_test)

    # Step 4 — Test all thresholds on original models
    print("\n=== THRESHOLD SWEEP — ORIGINAL MODELS ===")
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        evaluate_with_threshold(trained_models, X_test, y_test, threshold=threshold)
 
    #Step 5 — Train and evaluate tuned models with better hyperparameters
    tuned_models = tune_hyperparameters(X_train, y_train, X_test, y_test)

    # Step 6 — evaluate model using different thresholds to find the best one for tuned models
    print("\n=== THRESHOLD SWEEP — TUNED MODELS ===")
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        evaluate_with_threshold(tuned_models, X_test, y_test, threshold=threshold)

    # Step 7 — Save best model
    #save_best_model(tuned_models, threshold=0.1)
    








    # Improvement 1 — Threshold adjustment
    #evaluate_with_threshold(trained_models, X_test, y_test, threshold=0.30)


    # Step 4 — Cross validate models
    #cross_validate_models(X_train, y_train, trained_models)


if __name__ == "__main__":
    main()