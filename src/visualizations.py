import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sqlalchemy import create_engine, text

DB_URL = os.getenv(
    "DATABASE_URL",
    postgresql+psycopg2://kk@localhost:5432/cs210_diabetes
engine = create_engine(DB_URL)
FIGURE_DIR = Path("outputs/figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

def load_dataset():
    """
     Load data from PostgreSQL for visualization
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
        h.bmi,
        h.smoker,
        h.genhlth,
        h.menthlth,
        h.physhlth
    FROM demographics d
    JOIN health_indicators h
        ON d.respondent_id = h.respondent_id
    JOIN diabetes_labels l
        ON d.respondent_id = l.respondent_id;
    """
    return pd.read_sql_query(text(query), engine)

# diabetes rate by BMI group
def plot_bmi_category(df):
    bins = [0, 18.5, 25, 30, 100]
    labels = ["Underweight", "Normal", "Overweight", "Obese"]
    df["bmi_cat"] = pd.cut(df["bmi"], bins=bins, labels=labels)
    rate = df.groupby("bmi_cat")["diabetes_binary"].mean() * 100
    plt.figure()
    rate.plot(kind="bar")
    plt.title("Diabetes Rate by BMI Category")
    plt.ylabel("Rate (%)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "bmi_category.png")
    plt.close()

# by general health
def plot_genhlth(df):
    rate = df.groupby("genhlth")["diabetes_binary"].mean() * 100
    plt.figure()
    rate.plot(kind="bar")
    plt.title("Diabetes Rate by General Health")
    plt.ylabel("Rate (%)")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "genhlth.png")
    plt.close()

# highbp vs diabetes
def plot_highbp(df):
    rate = df.groupby("highbp")["diabetes_binary"].mean() * 100
    plt.figure()
    rate.plot(kind="bar")
    plt.title("Diabetes Rate by High Blood Pressure")
    plt.ylabel("Rate (%)")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "highbp.png")
    plt.close()
    
# by income level
def plot_income(df):
    rate = df.groupby("income")["diabetes_binary"].mean() * 100
    plt.figure()
    rate.plot(kind="bar")
    plt.title("Diabetes Rate by Income")
    plt.ylabel("Rate (%)")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "income.png")
    plt.close()

# heatmap
def plot_corr(df):
    corr = df.drop(columns=["respondent_id"]).corr()
    plt.figure(figsize=(10, 8))
    plt.imshow(corr)
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "correlation.png")
    plt.close()

def main():
    df = load_dataset()
    plot_bmi_category(df)
    plot_genhlth(df)
    plot_highbp(df)
    plot_income(df)
    plot_corr(df)

    print("Visualizations saved!")


if __name__ == "__main__":
    main()
