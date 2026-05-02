-- Load the raw CSV file into the staging table.
-- The staging table stores all values as TEXT, so COPY can import the file
-- directly without type conversions at this stage.
--
-- Important:
-- This file path is local to the current machine. If another team member runs it,
-- they must replace the path with their own CSV file location.
COPY staging_diabetes_raw
FROM '/Users/kk/Desktop/archive/diabetes_binary_health_indicators_BRFSS2015.csv'
DELIMITER ','
CSV HEADER;
