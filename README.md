# CS210 Diabetes Prediction Project

## Overview

This project analyzes diabetes risk factors using the BRFSS 2015 Diabetes Health Indicators Dataset. We built a full data pipeline that includes PostgreSQL database design, ETL processing, SQL analytics, and machine learning models to predict diabetes status.

The project demonstrates how structured health survey data can be transformed into actionable insights through data engineering and predictive analytics.

---

## Objectives

- Build a relational PostgreSQL database from raw survey data
- Normalize the dataset into multiple linked tables
- Perform SQL-based health analytics
- Train machine learning models to predict diabetes
- Identify key risk factors associated with diabetes

---

## Dataset

Source: Kaggle / BRFSS 2015 Diabetes Health Indicators Dataset

Records: 253,680 respondents

Target Variable:

- `diabetes_binary`
    - 0 = No diabetes
    - 1 = Diabetes

Features include:

- BMI
- Age
- Income
- High Blood Pressure
- High Cholesterol
- Smoking
- Physical Activity
- General Health
- Mental Health
- Education

---

## Tech Stack

- Python
- PostgreSQL
- Pandas
- SQLAlchemy
- Scikit-learn
- Matplotlib

---

## Database Design

The raw dataset was normalized into four relational tables:

- `respondents`
- `demographics`
- `health_indicators`
- `diabetes_labels`

This design reduces redundancy and improves query efficiency.

---

## ETL Pipeline

1. Import raw CSV into staging table
2. Clean and validate records
3. Split data into normalized tables
4. Load into PostgreSQL relational schema

---

## SQL Analysis

Example findings:

### Diabetes Prevalence

- 13.94% diabetic
- 86.06% non-diabetic

### Average BMI

- Non-diabetic: 27.81
- Diabetic: 31.94

### Age Trend

Diabetes prevalence increases significantly with age.

---

## Machine Learning Models

Two supervised learning models were trained:

### Logistic Regression

- Accuracy: 72.8%
- Recall: 75.9%
- ROC-AUC: 0.818

Best for screening use cases.

### Random Forest

- Accuracy: 85.7%
- Precision: 45.8%

Best for overall classification performance.

---

## Key Predictors of Diabetes

Feature importance from Random Forest:

1. BMI
2. Age
3. General Health
4. Income
5. High Blood Pressure

---

## Project Structure

```text
CS210_Project/
│── data/
│── sql/
│   ├── schema.sql
│   ├── load_staging.sql
│   └── analysis_queries.sql
│── src/
│   ├── etl.py
│   └── model.py
│── outputs/
│   └── figures/
│── README.md
