# lt_cvd

### Contents
- [Getting Started](#getting-started)
- [Running the Code](#running-the-code)
- [Required Format of Subjects File](#required-format-of-subjects-file)
  - [Further explanations of features](#further-explanations-of-features)
  - [Associating Features with a Prediction Time](#associating-features-with-a-prediction-time)
- [Outputs](#outputs)


### Getting started
Clone the repo and set up your python environment. The [scikit-survival](https://scikit-survival.readthedocs.io/en/stable/install.html), [lifelines](https://lifelines.readthedocs.io/en/latest/), and [sklearn](https://scikit-learn.org/stable/install.html) packages are required.

### Running the code
Run the `run_model.py` script that is in the `lt_cvd` directory:
```
usage: python run_model.py --model_dir <model directory> --subjects_file <subjects csv> [--outdir <output directory>]

                --model_dir MODEL_DIR (the directory with the rsf.pkl and norm.pkl files - they are in lt_cvd in the repo)
                --subjects_file SUBJECTS_FILE (csv file with your subjects, their features, and their labels)
                [--outdir OUTDIR] (output directory)
```

## Required format of subjects file
The subjects file is a csv with the patient information in the following columns
|Column name| Content | Notes |
| --------- | ------- | ----- |
|ID |identifier of patient |use randomized, de-identified IDs |
|AGE_AT_TX |age at transplant in years | |
|CURR_AGE|age at prediction time in years |current/age at the time of labs being used |
|YRS_SINCE_TRANS|# of years since transplant | |
| SEX|0 for male 1 for female | |
| SMOKER|smoking status |binary, 1 for current OR ex-smoker |
| DM|diabetes status |binary. ICD10: E10, E11 |
| HTN|hypertension status |binary. ICD10: I10, I11, I12, I13, I15  |
| LIP| dyslipidemia status|binary. ICD10: E78.0,E78.1,E78.2,E78.5 |
| CV_HISTORY| whether had any prior CV events | relative to prediction time, at YRS_SINCE_TRANS years post-transplant |
| ANTI_PLATELET|anti-platelet medication status |at prediction time |
| ANTI_HTN|anti-hypertensive medication status |at prediction time |
| STATIN|statin medication status |at prediction time |
| BMI| bmi| |
| CANCER|HCC (liver cancer) was indication for transplant |ICD10: C22  Note: multiple indications are possible|
| METAB|MASH was indication for transplant | ICD10: K76.0, K75.81 |
| ALD|alcoholic liver disease was indication for transplant | ICD10: K70  |
| HEP|Hepatitis B or C was indication for transplant | ICD10: B15-B19 |
| FULM| fulminant hepatic failure (acute liver failure) was indication for transplant| ICD10: K72.0 |
| IMMUNE| Autoimmune liver disease was indication for transplant | ICD10: K75.4, K74.3, K83.01 Includes PSC, PBC|
| RE_TX| Re-transplantation| (Possibly- history of liver transplant failure: T86.4, history of liver transplant: Z94.4)  |
| CYCLOSPORINE_TROUGH_LEVEL| medication blood level| ng/mL|
| TACROLIMUS_TROUGH_LEVEL| medication blood level | ng/mL NOTE: should be on cyclosporine OR tacrolimus, mutually exclusive|
| ALP|alkaline phosphatase |IU/L |
| ALT| alanine aminotransferase|IU/L |
| AST| aspartate aminotransferase|IU/L |
| SERUM_CREATININE| creatinine level|umol/L |
| EVENT| whether the subject ever had an event after prediction time | 1 for event, 0 for censored |
| MONTHS_TO_EVENT| time in months from prediction time until event or censoring | |

### Further explanations of features
- **HTN**: a patient is considered hypertensive if they have a diagnosis of hypertension OR are on anti-hypertensive medication
- **LIP**: a patient is considered dislipidemic if they have a diagnosis of dyslipidemia, OR are on a statin, OR have ever had a lipid lab with LDL > 4.1 or Total cholesterol > 5.2 or triglycerides > 2.3
- **CV_HISTORY/EVENT**: we consider the following to be major adverse cardiovascular events: acute myocardial infarction (MI), non-MI ischemic heart disease, heart failure, arrhythmias, valvulopathy, cardiac and carotid revascularization procedures, cardiogenic shock and ischemic stroke and death from any of these. <br>
These are captured in part by ICD10 codes: I48.0, I48.1, I48.2, I48.3, I48.4, I48.9, I47, I49, I21, I22, I25.2, I46, I25.1, I25.4, I25.6, I25.7, I25.8, I25.9, G45.3, G45.9, I65, I66, I67.0 I63, I69.3, I34, I35, I36, I37, I25.5, I50, I42 <br>
Note this list may not be exhaustive. In particular, _it does not include procedures_: PCI, stenting, bypass, ablation, valve replacement, pacemaker and defibrillator insertion, which are all indicative of a CV event.
- **ANTI_PLATELET**: keywords: aspirin, asa, clopidogrel, plavix, prasugrel, ticagrelor, dipyridamole. Note - may not be an exhaustive list.
- **ANTI_HTN**: keywords: Amlodipine, caduet, Lecarnidipine, Diltiazem, Verapamil, Doxazosin, Terazosin, Prazosin, Atenolol, bisoprolol, metoprolol, nadolol, nebivolol, propanolol, Ramipril, Perindopril, Captopril, Lisinopril, Enalapril, Candesartan, irbesartan, losartan, telmisartan, Hydrochlorthiazide, indapamide
- **STATIN**: keywords: Atorvastatin, caduet, simvastatin, zocor, rosuvastatin, crestor, fluvastatin, Ezetimibe, Bezafibrate, fenofibrate, lipitor, lescol, lovastatin, mevacor, altoprev, livalo, zypitamag, pitavastatin, pravachol, pravastatin, ezallor, vytorin. Note: medication lists may not be exhaustive.
- **CYCLOSPORINE_TROUGH_LEVEL**: if the patient is on tacrolimus leave this empty (or 0)
- **MONTHS_TO_EVENT**: This should be the time in months from the prediction timepoint until the next CV event if there is one, or until censoring time for that patient. Patients can be censored at death by non-CV cause or at end of follow-up time which could be when the patient is lost to follow-up, leaves the network, or the last date in the data pull.
  
### Associating features with a prediction time
The model predicts risk of an event within 10 years. The timepoint from which that prediction is made is the time from which the feature values are collected, and specified by YRS_SINCE_TRANS. Labs (BMI, ALP, AST, ALT, SERUM_CREATININE) should only be considered from >3 months post-transplant, and YEARS_SINCE_TRANS should be > 1. The most recent lab value prior to YRS_SINCE_TRANS and after the 3 month recovery window should be used.

Health and medication statuses should be from the prediction timepoint (ie a subsequent diabetes diagnoses that occurs after YRS_SINCE_TRANS should not be considered as a positive), and include diagnoses and prescriptions from pre-transplant as well.
Note that positive health and medication statuses should be assumed to persist - a diagnosis of hypertension or requiring statins is assumed not to reverse over time. (An exception is post-transplant diabetes, which is often transient, but usually coded differently).

CV_HISTORY includes any of the CV events that occurred prior to YRS_SINCE_TRANS, including prior to transplant. EVENT is 1 if there is a CV event after YRS_SINCE_TRANS, and MONTHS_TO_EVENT is until the _next_ event.

NOTE: There may be data to define multiple such time points per patient. For validation, pick only one per patient. Select timepoints with more recent lab measurements and lower missingness of features. Picking earlier timepoints will also allow for longer follow up times, and more observed events.



### Outputs
The code will print out the C-index and Brier score and save 3 files in the output directory: `predictions.csv` with the predicted 10 year risk, `metrics.txt` with the C-index and Brier score metrics, and a calibration plot in `calibration.png`
