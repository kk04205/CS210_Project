-- 1. Diabetes prevalence
SELECT diabetes_binary, COUNT(*) AS total
FROM diabetes_labels
GROUP BY diabetes_binary
ORDER BY diabetes_binary;

-- 2. Average BMI by diabetes status
SELECT d.diabetes_binary,
       ROUND(AVG(h.bmi), 2) AS avg_bmi
FROM diabetes_labels d
JOIN health_indicators h
ON d.respondent_id = h.respondent_id
GROUP BY d.diabetes_binary
ORDER BY d.diabetes_binary;

-- 3. Diabetes rate by age group
SELECT g.age,
       ROUND(AVG(d.diabetes_binary) * 100, 2) AS diabetes_rate_percent
FROM demographics g
JOIN diabetes_labels d
ON g.respondent_id = d.respondent_id
GROUP BY g.age
ORDER BY g.age;
