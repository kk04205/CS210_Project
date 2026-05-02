from sqlalchemy import create_engine, text

# Database connection string.
# This connects Python to the local PostgreSQL database used in the project.
DB_URL = "postgresql+psycopg2://kk@localhost:5432/cs210_diabetes"

# Create a SQLAlchemy engine so we can execute SQL statements from Python.
engine = create_engine(DB_URL)

# Each SQL statement below represents one step in the ETL pipeline.
# The raw survey data is first assumed to already exist in staging_diabetes_raw.
# Then we normalize the data into separate relational tables.
etl_steps = [
    """
    INSERT INTO respondents (source_row_id)
    SELECT ROW_NUMBER() OVER ()
    FROM staging_diabetes_raw;
    """,
    """
    INSERT INTO demographics (
        respondent_id, sex, age, education, income
    )
    SELECT
        respondent_id,
        s.sex::numeric::smallint,
        s.age::numeric::smallint,
        s.education::numeric::smallint,
        s.income::numeric::smallint
    FROM respondents r
    JOIN (
        SELECT ROW_NUMBER() OVER () AS rn, *
        FROM staging_diabetes_raw
    ) s
    ON r.source_row_id = s.rn;
    """,
    """
    INSERT INTO health_indicators (
        respondent_id, highbp, highchol, cholcheck, bmi, smoker,
        stroke, heartdiseaseorattack, physactivity, fruits, veggies,
        hvyalcoholconsump, anyhealthcare, nodocbccost, genhlth,
        menthlth, physhlth, diffwalk
    )
    SELECT
        respondent_id,
        s.highbp::numeric::smallint,
        s.highchol::numeric::smallint,
        s.cholcheck::numeric::smallint,
        s.bmi::numeric,
        s.smoker::numeric::smallint,
        s.stroke::numeric::smallint,
        s.heartdiseaseorattack::numeric::smallint,
        s.physactivity::numeric::smallint,
        s.fruits::numeric::smallint,
        s.veggies::numeric::smallint,
        s.hvyalcoholconsump::numeric::smallint,
        s.anyhealthcare::numeric::smallint,
        s.nodocbccost::numeric::smallint,
        s.genhlth::numeric::smallint,
        s.menthlth::numeric::smallint,
        s.physhlth::numeric::smallint,
        s.diffwalk::numeric::smallint
    FROM respondents r
    JOIN (
        SELECT ROW_NUMBER() OVER () AS rn, *
        FROM staging_diabetes_raw
    ) s
    ON r.source_row_id = s.rn;
    """,
    """
    INSERT INTO diabetes_labels (
        respondent_id, diabetes_binary
    )
    SELECT
        respondent_id,
        s.diabetes_binary::numeric::smallint
    FROM respondents r
    JOIN (
        SELECT ROW_NUMBER() OVER () AS rn, *
        FROM staging_diabetes_raw
    ) s
    ON r.source_row_id = s.rn;
    """
]

# Run all ETL steps inside a single transaction.
# engine.begin() ensures that if one step fails, the transaction can be rolled back.
with engine.begin() as conn:
    for step in etl_steps:
        conn.execute(text(step))

print("ETL completed successfully.")
