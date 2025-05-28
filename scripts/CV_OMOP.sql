with PATS as (
    SELECT 
    p.person_id,
    p.birth_datetime AS birth_date,
    CASE 
        WHEN p.gender_concept_id = 8507 THEN 'Male'
        WHEN p.gender_concept_id = 8532 THEN 'Female'
        ELSE 'Other'
    END AS sex,
    po.procedure_date AS transplant_date,
    c.concept_name AS procedure_name
FROM 
    person p
INNER JOIN 
    procedure_occurrence po ON p.person_id = po.person_id
INNER JOIN 
    concept c ON po.procedure_concept_id = c.concept_id
WHERE 
    c.concept_name LIKE '%liver transplant%' -- Or use specific liver transplant concept IDs
ORDER BY 
    p.person_id, po.procedure_date;
), 

SMOKING_CTE AS (
    SELECT 
    o.person_id,
    o.observation_date,
    c.concept_name AS smoking_status
FROM 
    observation o
INNER JOIN 
    concept c ON o.observation_concept_id = c.concept_id
INNER JOIN
    PATS p ON o.person_id = p.person_id
WHERE 
    o.observation_concept_id IN (
        -- Replace with the actual smoking-related concept IDs from your vocabulary
        '40764333', -- Current smoker
        '40768543', -- Former smoker
        '40770157'  -- Never smoker
    )
ORDER BY 
    o.person_id, o.observation_date;
), 

INDICATION_CTE (
    SELECT 
    co.person_id,
    co.condition_start_date AS diagnosis_date,
    icd.concept_name AS diagnosis,
    icd.concept_code AS icd10_code
FROM 
    condition_occurrence co
INNER JOIN 
    concept icd ON co.condition_concept_id = icd.concept_id
INNER JOIN
    PATS p ON co.person_id = p.person_id
WHERE 
    icd.concept_code IN ('K75.81%','K76.0%','K70%','C22%','B15%', 'B16%', 'B17%', 'B18%', 'B19%','K72.0%','K75.4%', 'K74.3%', 'K83.01%','Z94.4%', 'T86.4%')
    AND icd.vocabulary_id = 'ICD10CM' -- Ensures the codes come from the ICD-10 vocabulary
ORDER BY 
    co.person_id, co.condition_start_date;
), 

DHD_CTE ( --DIABETES/HYPERTENSION/DYSLIPIDEMIA CTE
    SELECT 
    co.person_id,
    co.condition_start_date AS diagnosis_date,
    icd.concept_name AS diagnosis,
    icd.concept_code AS icd10_code
FROM 
    condition_occurrence co
INNER JOIN 
    concept icd ON co.condition_concept_id = icd.concept_id
INNER JOIN
    PATS p ON co.person_id = p.person_id 
WHERE 
    icd.concept_code IN ('E10%', 'E11%', 'I10%', 'I11%', 'I12%', 'I13%', 'I15%', 'E78.0%', 'E78.1%', 'E78.2%', 'E78.5%')
    AND icd.vocabulary_id = 'ICD10CM' -- Ensures the codes come from the ICD-10 vocabulary
ORDER BY 
    co.person_id, co.condition_start_date;
), 

CV_EVENTS_CTE ( 
    SELECT 
    co.person_id,
    co.condition_start_date AS diagnosis_date,
    icd.concept_name AS diagnosis,
    icd.concept_code AS icd10_code
FROM 
    condition_occurrence co
INNER JOIN 
    concept icd ON co.condition_concept_id = icd.concept_id
INNER JOIN
    PATS p ON co.person_id = p.person_id 
WHERE 
    icd.concept_code IN ('I48.0%', 'I48.1%', 'I48.2%', 'I48.3%', 'I48.4%', 'I48.9%', 'I47%', 'I49%', 'I21%', 'I22%', 'I25.2%', 'I50%', 'I46%', 'G45.3%', 'G45.9%', 'I25.1%','I25.4%', 'I25.5%', 'I25.6%','I25.7%','I25.8%','I25.9%', 'I63%', 'I34%', 'I35%', 'I36%', 'I37%', 'I65%', 'I66%', 'I67.0%', 'I69.3%', 'I42%')
    AND icd.vocabulary_id = 'ICD10CM' -- Ensures the codes come from the ICD-10 vocabulary
ORDER BY 
    co.person_id, co.condition_start_date;
), 

LABS_CTE (
    SELECT
    m.person_id,
    m.measurement_date,
    icd.concept_name AS test_name,
    icd.concept_code AS test_code,
    m.value_as_number AS test_value,
    unit_concept.concept_name AS test_unit
FROM measurement m
INNER JOIN
    concept icd ON m.measurement_concept_id = icd.concept_id
INNER JOIN
    PATS p ON m.person_id = p.person_id
INNER JOIN
    concept unit_concept ON m.unit_concept_id = unit_concept.concept_id
WHERE
    icd.concept_code IN (
        '3013671',  -- BMI
        '3036278',  -- HDL Cholesterol
        '3036280',  -- LDL Cholesterol
        '3036282',  -- Total Cholesterol
        '3023103',  -- Alanine Aminotransferase
        '3023102',  -- Aspartate Aminotransferase
        '3027114',  -- Alkaline Phosphatase
        '3023420',  -- Creatinine
        '3039012',  -- Tacrolimus trough level
        '3039014',  -- Cyclosporine trough level
    )
    AND icd.vocabulary_id = 'LOINC'  -- Ensures the tests are from the LOINC vocabulary
ORDER BY 
    p.person_id, m.measurement_date;
),

MEDS_CTE (
SELECT 
    de.person_id,
    c.concept_name AS medication_name,
    c.concept_code AS medication_code,
    de.drug_exposure_start_date AS start_date,
    de.drug_exposure_end_date AS end_date,
    de.quantity AS dosage
FROM 
    drug_exposure de
JOIN 
    concept c ON de.drug_concept_id = c.concept_id
INNER JOIN
    PATS p ON de.person_id = p.person_id
WHERE 
    de.person_id IN (<list_of_person_ids>) -- Replace with your list of person_ids
    AND c.concept_name IN (
    '%ASPIRIN%','%ASA%','%CLOPIDOGREL%','%PLAVIX%','%PRASUGREL%','%TICAGRELOR%','%DIPYRIDAMOLE%',
    '%AMLODIPINE%','%CADUET%','%LECARNIDIPINE%','%DILTIAZEM%','%VERAPAMIL%','%DOXAZOSIN%',
    '%TERAZOSIN%','%PRAZOSIN%','%ATENOLOL%','%BISOPROLOL%','%METOPROLOL%','%NADOLOL%',
	'%NEBIVOLOL%','%PROPANOLOL%','%RAMIPRIL%','%PERINDOPRIL%','%CAPTOPRIL%','%LISINOPRIL%',
	'%ENALAPRIL%','%CANDESARTAN%','%IRBESARTAN%','%LOSARTAN%','%TELMISARTAN%','%HYDROCHLORTHIAZIDE%',
	'%INDAPAMIDE%',
    '%STATIN%','%ATORVASTATIN%','%SIMVASTATIN%','%ZOCOR%','%ROSUVASTATIN%','%CRESTOR%',
    '%FLUVASTATIN%','%EZETIMIBE%','%BEZAFIBRATE%','%FENOFIBRATE%','%LIPITOR%',
    '%LESCOL%','%LOVASTATIN%','%MEVACOR%','%ALTOPREV%','%LIVALO%','%ZYPITAMAG%',
    '%PITAVASTATIN%','%PRAVACHOL%','%PRAVASTATIN%','%EZALLOR%','%VYTORIN%'
    );
)
    
-- QUERIES (GENERATE ALL THE SEVEN TABLES BELOW)
select * from PATS;
-- select * from SMOKING_CTE;
-- select * from INDICATION_CTE;
-- select * from DHD_CTE;
-- select * from CV_EVENTS_CTE;
-- select * from LABS_CTE;
-- select * from MEDS_CTE;

