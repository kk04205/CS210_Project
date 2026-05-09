import os
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sqlalchemy import create_engine, text
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Use environment variable if available, otherwise use local database
DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres@localhost:5432/cs210_diabetes"
)
engine = create_engine(DB_URL)
OUTPUT_DIR = Path("outputs")
FIGURE_DIR = Path("outputs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    query = """
    SELECT
        d.respondent_id,
        l.diabetes_binary,
        d.sex,
        d.age,
        d.education,
        d.income,
        h.highbp,
        h.highchol,
        h.cholcheck,
        h.bmi,
        h.smoker,
        h.stroke,
        h.heartdiseaseorattack,
        h.physactivity,
        h.fruits,
        h.veggies,
        h.hvyalcoholconsump,
        h.anyhealthcare,
        h.nodocbccost,
        h.genhlth,
        h.menthlth,
        h.physhlth,
        h.diffwalk
    FROM demographics d
    JOIN health_indicators h
        ON d.respondent_id = h.respondent_id
    JOIN diabetes_labels l
        ON d.respondent_id = l.respondent_id;
    """
    return pd.read_sql_query(text(query), engine)

def get_model_scores(model, X_test):
    """
    Get scores for ROC-AUC
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        return model.decision_function(X_test)
    else:
        return model.predict(X_test)

def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    print(f"Training {name}...")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_score = get_model_scores(model, X_test)

    return {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_score),
        "model_object": model,
        "y_score": y_score
    }

def plot_model_comparison(results_df):
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    # Reorganize the dataframe so each model is a category on the x-axis
    plot_df = results_df.set_index("model")[metrics]
    plt.figure(figsize=(12, 6))
    plot_df.plot(kind="bar", figsize=(12, 6))
    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.xlabel("Model")
    plt.xticks(rotation=30, ha="right")
    plt.legend(title="Metric")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "model_comparison.png", dpi=300)
    plt.close()

def plot_roc_curves(results, y_test):
    plt.figure(figsize=(9, 7))
    for result in results:
        name = result["model"]
        y_score = result["y_score"]
        fpr, tpr, _ = roc_curve(y_test, y_score)
        auc_score = roc_auc_score(y_test, y_score)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc_score:.3f})")

    # random guess line
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random Guess")
    plt.title("ROC Curve Comparison Across Models")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "roc_curve_comparison.png", dpi=300)
    plt.close()

def main():
    """
    Run the full machine learning workflow
    """
    df = load_data()
    print("Dataset shape:", df.shape)

    # Separate predictors (X) and target label (y)
    X = df.drop(columns=["respondent_id", "diabetes_binary"])
    y = df["diabetes_binary"]

    # stratified split (keep class ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # All models to compare
    models = [
        (
            "Logistic Regression",
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced"
                ))
            ])
        ),
        (
            "Decision Tree",
            DecisionTreeClassifier(
                random_state=42,
                class_weight="balanced"
            )
        ),
        (
            "Random Forest",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1
            )
        ),
        (
            "KNN",
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier())
            ])
        ),
        (
            "Gradient Boosting",
            GradientBoostingClassifier(random_state=42)
        ),
        (
            "SVM",
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", LinearSVC(
                    class_weight="balanced",
                    random_state=42,
                    max_iter=5000
                ))
            ])
        ),
        (
            "Neural Network",
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    max_iter=200,
                    random_state=42
                ))
            ])
        ),
    ]
    full_results = []

    for name, model in models:
        result = evaluate_model(name, model, X_train, X_test, y_train, y_test)
        full_results.append(result)

    results_df = pd.DataFrame([
        {
            "model": r["model"],
            "accuracy": r["accuracy"],
            "precision": r["precision"],
            "recall": r["recall"],
            "f1": r["f1"],
            "roc_auc": r["roc_auc"]
        }
        for r in full_results
    ])

    print("\nModel Results:")
    print(results_df)

    results_df.to_csv(OUTPUT_DIR / "model_results.csv", index=False)

    # Generate visualizations 
    plot_model_comparison(results_df)
    plot_roc_curves(full_results, y_test)

    print("\nSaved results to outputs/model_results.csv")
    print("Saved model comparison figure to outputs/figures/model_comparison.png")
    print("Saved ROC curve figure to outputs/figures/roc_curve_comparison.png")


if __name__ == "__main__":
    main()
