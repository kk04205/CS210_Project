import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Update this if your postgres username/database is different
DB_URL = "postgresql+psycopg2://kk@localhost:5432/cs210_diabetes"

engine = create_engine(DB_URL)

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
    ON d.respondent_id = l.respondent_id
"""

df = pd.read_sql(query, engine)

print("Dataset shape:", df.shape)
print(df.head())

X = df.drop(columns=["respondent_id", "diabetes_binary"])
y = df["diabetes_binary"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# Logistic Regression
log_model = LogisticRegression(max_iter=1000, class_weight="balanced")
log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)
y_prob_log = log_model.predict_proba(X_test)[:, 1]

print("\n=== Logistic Regression ===")
print("Accuracy:", round(accuracy_score(y_test, y_pred_log), 4))
print("Precision:", round(precision_score(y_test, y_pred_log), 4))
print("Recall:", round(recall_score(y_test, y_pred_log), 4))
print("F1-score:", round(f1_score(y_test, y_pred_log), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, y_prob_log), 4))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log))
print("Classification Report:\n", classification_report(y_test, y_pred_log))

# Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

print("\n=== Random Forest ===")
print("Accuracy:", round(accuracy_score(y_test, y_pred_rf), 4))
print("Precision:", round(precision_score(y_test, y_pred_rf), 4))
print("Recall:", round(recall_score(y_test, y_pred_rf), 4))
print("F1-score:", round(f1_score(y_test, y_pred_rf), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, y_prob_rf), 4))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))

# Feature importance from Random Forest
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

print("\nTop 10 Important Features:")
print(importances.head(10))

plt.figure(figsize=(10, 6))
importances.head(10).sort_values().plot(kind="barh")
plt.title("Top 10 Feature Importances (Random Forest)")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("outputs/figures/top10_feature_importance.png")
plt.show()
