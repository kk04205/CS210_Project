COPY staging_diabetes_raw
FROM '/Users/kk/Desktop/archive/diabetes_binary_health_indicators_BRFSS2015.csv'
DELIMITER ','
CSV HEADER;
