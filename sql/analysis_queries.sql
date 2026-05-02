-- 1. Diabetes prevalence
-- Count how many respondents belong to each diabetes class.
-- diabetes_binary = 0 means no diabetes
-- diabetes_binary = 1 means diabetes / prediabetes
SELECT diabetes_binary, COUNT(*) AS total
FROM diabetes_labels
GROUP BY diabetes_binary
ORDER BY diabetes_binary;

-- 2. Average BMI by diabetes status
-- Compare the mean BMI for each diabetes group.
-- This helps show whether respondents with diabetes tend to have higher BMI.
SELECT d.diabetes_binary,
       ROUND(AVG(h.bmi), 2) AS avg_bmi
FROM diabetes_labels d
JOIN health_indicators h
ON d.respondent_id = h.respondent_id
GROUP BY d.diabetes_binary
ORDER BY d.diabetes_binary;

-- 3. Diabetes rate by age group
-- Calculate the percentage of diabetic respondents in each age category.
-- Since diabetes_binary is coded as 0/1, AVG(diabetes_binary) gives the rate.
SELECT g.age,
       ROUND(AVG(d.diabetes_binary) * 100, 2) AS diabetes_rate_percent
FROM demographics g
JOIN diabetes_labels d
ON g.respondent_id = d.respondent_id
GROUP BY g.age
ORDER BY g.age;
