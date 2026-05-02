import pandas as pd
from sqlalchemy import create_engine, text

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt
from pathlib import Path

DB_URL = "postgresql+psycopg2://kk@localhost:5432/cs210_diabetes"
engine = create_engine(DB_URL)

FIGURE_DIR = Path("outputs/figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    query = """
    SELECT *
    FROM demographics d
    JOIN health_indicators h ON d.respondent_id = h.respondent_id
    JOIN diabetes_labels l ON d.respondent_id = l.respondent_id;
    """
    return pd.read_sql_query(text(query), engine)


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }


def main():
    df = load_data()

    df = df.drop(columns=["respondent_id"])

    X = df.drop(columns=["diabetes_binary"])
    y = df["diabetes_binary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = []

    models = [
        ("Logistic", LogisticRegression(max_iter=1000)),
        ("Decision Tree", DecisionTreeClassifier()),
        ("Random Forest", RandomForestClassifier()),
        ("KNN", KNeighborsClassifier()),
        ("Gradient Boosting", GradientBoostingClassifier()),
    ]

    for name, model in models:
        result = evaluate_model(name, model, X_train, X_test, y_train, y_test)
        results.append(result)

    results_df = pd.DataFrame(results)
    print(results_df)

    # ✅ 保存结果
    results_df.to_csv("outputs/model_results.csv", index=False)

    # ✅ 可视化
    plt.figure(figsize=(8, 5))
    plt.bar(results_df["model"], results_df["f1"])
    plt.title("Model Comparison (F1 Score)")
    plt.ylabel("F1 Score")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "model_comparison.png")
    plt.close()


if __name__ == "__main__":
    main()
