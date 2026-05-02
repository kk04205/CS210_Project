-- Drop tables in reverse dependency order so the schema can be rebuilt cleanly.
-- CASCADE ensures dependent objects are also removed.
DROP TABLE IF EXISTS diabetes_labels CASCADE;
DROP TABLE IF EXISTS health_indicators CASCADE;
DROP TABLE IF EXISTS demographics CASCADE;
DROP TABLE IF EXISTS respondents CASCADE;
DROP TABLE IF EXISTS staging_diabetes_raw CASCADE;

-- Staging table:
-- This raw table stores all imported CSV values as TEXT first.
-- We keep the staging layer simple so the raw data can be loaded
-- without worrying about type conversion during the COPY step.
CREATE TABLE staging_diabetes_raw (
    diabetes_binary TEXT,
    highbp TEXT,
    highchol TEXT,
    cholcheck TEXT,
    bmi TEXT,
    smoker TEXT,
    stroke TEXT,
    heartdiseaseorattack TEXT,
    physactivity TEXT,
    fruits TEXT,
    veggies TEXT,
    hvyalcoholconsump TEXT,
    anyhealthcare TEXT,
    nodocbccost TEXT,
    genhlth TEXT,
    menthlth TEXT,
    physhlth TEXT,
    diffwalk TEXT,
    sex TEXT,
    age TEXT,
    education TEXT,
    income TEXT
);

-- Master respondent table:
-- Each raw record gets a unique respondent_id used as the primary key
-- across the normalized schema.
-- source_row_id preserves the original row order from the staging table
-- so we can map raw rows to normalized tables during ETL.
CREATE TABLE respondents (
    respondent_id SERIAL PRIMARY KEY,
    source_row_id INT UNIQUE NOT NULL
);

-- Demographics table:
-- Stores demographic attributes separately to support normalization.
-- respondent_id is both the primary key and foreign key to respondents.
CREATE TABLE demographics (
    respondent_id INT PRIMARY KEY,
    sex SMALLINT,
    age SMALLINT,
    education SMALLINT,
    income SMALLINT,
    FOREIGN KEY (respondent_id) REFERENCES respondents(respondent_id)
);

-- Health indicators table:
-- Stores health- and behavior-related predictor variables.
-- Keeping these columns in a separate table makes the schema cleaner
-- and easier to query for feature analysis and model building.
CREATE TABLE health_indicators (
    respondent_id INT PRIMARY KEY,
    highbp SMALLINT,
    highchol SMALLINT,
    cholcheck SMALLINT,
    bmi NUMERIC(5,2),
    smoker SMALLINT,
    stroke SMALLINT,
    heartdiseaseorattack SMALLINT,
    physactivity SMALLINT,
    fruits SMALLINT,
    veggies SMALLINT,
    hvyalcoholconsump SMALLINT,
    anyhealthcare SMALLINT,
    nodocbccost SMALLINT,
    genhlth SMALLINT,
    menthlth SMALLINT,
    physhlth SMALLINT,
    diffwalk SMALLINT,
    FOREIGN KEY (respondent_id) REFERENCES respondents(respondent_id)
);

-- Label table:
-- Stores the target variable for supervised learning separately
-- from the predictor variables.
CREATE TABLE diabetes_labels (
    respondent_id INT PRIMARY KEY,
    diabetes_binary SMALLINT,
    FOREIGN KEY (respondent_id) REFERENCES respondents(respondent_id)
);
