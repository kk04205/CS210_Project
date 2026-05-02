import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://kk@localhost:5432/cs210_diabetes"
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


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }


def plot_model_comparison(results_df):
    plt.figure(figsize=(10, 6))
    plt.bar(results_df["model"], results_df["f1"])
    plt.title("Model Comparison by F1 Score")
    plt.ylabel("F1 Score")
    plt.xlabel("Model")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "model_comparison.png", dpi=300)
    plt.close()


def main():
    df = load_data()

    print("Dataset shape:", df.shape)

    X = df.drop(columns=["respondent_id", "diabetes_binary"])
    y = df["diabetes_binary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

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

    results = []

    for name, model in models:
        print(f"Training {name}...")
        result = evaluate_model(name, model, X_train, X_test, y_train, y_test)
        results.append(result)

    results_df = pd.DataFrame(results)

    print("\nModel Results:")
    print(results_df)

    results_df.to_csv(OUTPUT_DIR / "model_results.csv", index=False)
    plot_model_comparison(results_df)

    print("\nSaved results to outputs/model_results.csv")
    print("Saved model comparison figure to outputs/figures/model_comparison.png")


if __name__ == "__main__":
    main()
