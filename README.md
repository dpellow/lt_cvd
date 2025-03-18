# lt_cvd

### Contents
- [Getting Started](#getting-started)
- [Running the Code](#running-the-code)
- [Required Format of Subjects File](#required-format-of-subjects-file)
  - [Further explanations of features](#further-explanations-of-features)
  - [Associating Features with a Prediction Time](#associating-features-with-a-prediction-time)


### Getting started
Clone the repo and set up your python environment. The [scikit-survival](https://scikit-survival.readthedocs.io/en/stable/install.html), [shap](https://shap.readthedocs.io/en/latest/#install), and [matplotlib](https://matplotlib.org/stable/install/index.html) packages are required.

### Running the code
Run the `run_model.py` script that is in the `lt_cvd` directory:
```
usage: python run_model.py  --model_dir MODEL_DIR (the directory with the rsf.pkl and norm.pkl files - they are in lt_cvd in the repo)
                            --subjects_file SUBJECTS_FILE (csv file with your subjects, their features, and their labels)
                            [--outdir OUTDIR] (output directory)
```

## Required format of subjects file
The subjects file is a csv with the patient information in the following columns
|Column name| Content | Notes |
| --------- | ------- | ----- |
|ID |identifier of patient |patients should not be identifiable |
|AGE_AT_TX |age at transplant in years | |
|CURR_AGE|age at prediction time in years |current or age at the time of labs being used |
|YRS_SINCE_TRANS|# of years since transplant | |
| SEX|0 for male 1 for female | |
| SMOKER|smoking status |binary, 1 for current OR ex-smoker |
| DM|diabetes status |binary |
| HTN|hypertension status |binary |
| LIP| dyslipidemia status|binary |
| CV_HISTORY| whether had any prior CV events | relative to prediction time, at YRS_SINCE_TRANS years post-transplant |
| ANTI_PLATELET|anti-platelet medication status |at prediction time |
| ANTI_HTN|anti-hypertensive medication status |at prediction time |
| STATIN|statin medication status |at prediction time |
| BMI| | |
| CANCER|HCC (liver cancer) was indication for transplant | Note: multiple indications are possible|
| METAB|MASH was indication for transplant | |
| ALD|alcoholic liver disease was indication for transplant | |
| HEP|Hepatitis B or C was indication for transplant | |
| FULM| fulminant hepatic failure (acute liver failure) was indication for transplant| |
| IMMUNE| Autoimmune liver disease was indication for transplant | Includes PSC, PBC|
| RE_TX| Re-transplantation| |
| CYCLOSPORINE_TROUGH_LEVEL| medication blood level| ng/mL|
| TACROLIMUS_TROUGH_LEVEL| medication blood level | ng/mL NOTE: should be on cyclosporine OR tacrolimus, mutually exclusive|
| ALP|alkaline phosphatase |IU/L |
| ALT| alanine aminotransferase|IU/L |
| AST| aspartate aminotransferase|IU/L |
| SERUM_CREATININE| creatinine level|umol/L |
| EVENT| whether the subject ever had an event after prediction time | 1 for event, 0 for censored |
| MONTHS_TO_EVENT| time in months from prediction time until event or censoring | |

### Further explanations of features
### Associating features with a prediction time
