-- Load the raw CSV file into the staging table 
\copy staging_diabetes_raw
FROM 'diabetes_binary_health_indicators_BRFSS2015.csv'
DELIMITER ','
CSV HEADER;
