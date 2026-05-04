DROP TABLE IF EXISTS diabetes_labels CASCADE;
DROP TABLE IF EXISTS health_indicators CASCADE;
DROP TABLE IF EXISTS demographics CASCADE;
DROP TABLE IF EXISTS respondents CASCADE;
DROP TABLE IF EXISTS staging_diabetes_raw CASCADE;

-- Staging table
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

-- Master respondent table
CREATE TABLE respondents (
    respondent_id SERIAL PRIMARY KEY,
    source_row_id INT UNIQUE NOT NULL
);

-- Demographics table
CREATE TABLE demographics (
    respondent_id INT PRIMARY KEY,
    sex SMALLINT,
    age SMALLINT,
    education SMALLINT,
    income SMALLINT,
    FOREIGN KEY (respondent_id) REFERENCES respondents(respondent_id)
);

-- Health indicators table
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

-- Label table
CREATE TABLE diabetes_labels (
    respondent_id INT PRIMARY KEY,
    diabetes_binary SMALLINT,
    FOREIGN KEY (respondent_id) REFERENCES respondents(respondent_id)
);
