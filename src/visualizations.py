import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text


# Use an environment variable when available.
# This makes the project more reproducible because users can run the code
# with their own PostgreSQL username/password without editing the source file.
DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://kk@localhost:5432/cs210_diabetes"
)

engine = create_engine(DB_URL)

# All generated figures will be saved into this folder.
# The directory is created automatically so the script will not fail
# if outputs/figures does not already exist.
FIGURE_DIR = Path("outputs/figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset_from_database():
    """
    Load the normalized project dataset from PostgreSQL.

    The purpose of this query is to reconstruct one analysis-ready table
    from the normalized database schema. This shows that the machine learning
    and visualization steps are based on the relational database, not directly
    on the raw CSV file.
    """
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

    df = pd.read_sql_query(text(query), engine)
    return df


def plot_class_distribution(df):
    """
    Plot the distribution of the target variable.

    This plot is important because diabetes classification is affected by
    class imbalance. If most respondents are non-diabetic, accuracy alone may
    be misleading. This supports our decision to report precision, recall,
    F1-score, and ROC-AUC in addition to accuracy.
    """
    label_counts = df["diabetes_binary"].value_counts().sort_index()
    label_names = ["No Diabetes", "Diabetes"]

    plt.figure(figsize=(7, 5))
    bars = plt.bar(label_names, label_counts.values)

    for bar, count in zip(bars, label_counts.values):
        percentage = count / len(df) * 100
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count:,}\n({percentage:.2f}%)",
            ha="center",
            va="bottom"
        )

    plt.title("Class Distribution of Diabetes Status")
    plt.ylabel("Number of Respondents")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "class_distribution.png", dpi=300)
    plt.close()


def plot_bmi_by_diabetes_status(df):
    """
    Compare BMI distributions between non-diabetic and diabetic respondents.

    BMI is expected to be an important health indicator for diabetes risk.
    This plot visually compares the BMI distribution across the two target
    classes and helps explain why BMI may become an important predictor in
    the machine learning models.
    """
    no_diabetes_bmi = df.loc[df["diabetes_binary"] == 0, "bmi"]
    diabetes_bmi = df.loc[df["diabetes_binary"] == 1, "bmi"]

    plt.figure(figsize=(7, 5))
    plt.boxplot(
        [no_diabetes_bmi, diabetes_bmi],
        labels=["No Diabetes", "Diabetes"],
        showfliers=False
    )

    plt.title("BMI Distribution by Diabetes Status")
    plt.ylabel("BMI")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "bmi_by_diabetes_status.png", dpi=300)
    plt.close()


def plot_diabetes_rate_by_age(df):
    """
    Plot diabetes prevalence by age category.

    The BRFSS dataset encodes age as ordered categories. This plot shows
    whether diabetes prevalence increases as age category increases, which
    provides an interpretable public-health insight before model training.
    """
    age_rate = (
        df.groupby("age")["diabetes_binary"]
        .mean()
        .mul(100)
        .reset_index(name="diabetes_rate_percent")
    )

    plt.figure(figsize=(9, 5))
    plt.bar(age_rate["age"].astype(str), age_rate["diabetes_rate_percent"])

    plt.title("Diabetes Rate by Age Category")
    plt.xlabel("Age Category")
    plt.ylabel("Diabetes Rate (%)")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "diabetes_rate_by_age.png", dpi=300)
    plt.close()


def plot_correlation_heatmap(df):
    """
    Plot a correlation heatmap for all numeric variables.

    This visualization helps identify relationships among health indicators,
    demographics, and the diabetes label. It also helps detect variables that
    may be strongly related to the target or strongly related to each other.
    """
    numeric_df = df.drop(columns=["respondent_id"])
    corr = numeric_df.corr(numeric_only=True)

    plt.figure(figsize=(14, 10))
    plt.imshow(corr)
    plt.colorbar(label="Correlation")

    plt.xticks(
        ticks=np.arange(len(corr.columns)),
        labels=corr.columns,
        rotation=90,
        fontsize=8
    )
    plt.yticks(
        ticks=np.arange(len(corr.columns)),
        labels=corr.columns,
        fontsize=8
    )

    plt.title("Correlation Heatmap of Health Indicators")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "correlation_heatmap.png", dpi=300)
    plt.close()


def plot_diabetes_rate_by_key_binary_features(df):
    """
    Plot diabetes rates for selected binary health indicators.

    This figure connects model predictors with interpretable health patterns.
    For example, it helps compare diabetes prevalence among respondents with
    and without high blood pressure, high cholesterol, smoking history, and
    physical activity.
    """
    binary_features = [
        "highbp",
        "highchol",
        "smoker",
        "physactivity",
        "heartdiseaseorattack",
        "diffwalk"
    ]

    rates_no = []
    rates_yes = []

    for feature in binary_features:
        grouped_rate = df.groupby(feature)["diabetes_binary"].mean().mul(100)
        rates_no.append(grouped_rate.get(0, 0))
        rates_yes.append(grouped_rate.get(1, 0))

    x = np.arange(len(binary_features))
    width = 0.35

    plt.figure(figsize=(11, 6))
    plt.bar(x - width / 2, rates_no, width, label="Feature = 0")
    plt.bar(x + width / 2, rates_yes, width, label="Feature = 1")

    plt.title("Diabetes Rate by Key Binary Health Indicators")
    plt.xlabel("Health Indicator")
    plt.ylabel("Diabetes Rate (%)")
    plt.xticks(x, binary_features, rotation=30, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "diabetes_rate_by_key_binary_features.png", dpi=300)
    plt.close()


def main():
    df = load_dataset_from_database()

    print("Loaded dataset shape:", df.shape)

    plot_class_distribution(df)
    plot_bmi_by_diabetes_status(df)
    plot_diabetes_rate_by_age(df)
    plot_correlation_heatmap(df)
    plot_diabetes_rate_by_key_binary_features(df)

    print("EDA visualizations saved to outputs/figures/")


if __name__ == "__main__":
    main()
  
