# preprocess data pulled by OMOP script

import argparse
import os
import json

import pandas as pd
import numpy as np
import project_lists


def process_pats(pats):
    ''' PATS table processing.
        columns:
            - person_id
            - birth_date -> convert to datetime
            - transplant_date -> convert to datetime. Take the last transplant per patient
            - sex -> convert to binary: M=0, F=1. Remove any other values
            - procedure_name -> check that it's in the list and drop columns that aren't      
        new colums:
            - age_at_trans -> calculate from birth_date and transplant_date
    '''
    print("Processing patient cohort")
    
    pats['birth_date'] = pd.to_datetime(pats['birth_date'], format='mixed')
    pats['transplant_date'] = pd.to_datetime(pats['transplant_date'], format='mixed')
    
    # keep only the last transplant per patient
    pats = pats.sort_values('transplant_date').groupby('person_id').tail(1)
    
    pats['sex'] = pats['sex'].apply(lambda x: 0 if x == 'Male' else 1 if x == 'Female' else np.nan)
        
    # drop pats with sex=nan and print warning
    missing_sex = pats.loc[pats['sex'].isna(), 'person_id'].values.tolist()
    if(len(missing_sex) > 0):
        print(f"Dropping {len(missing_sex)} patients with missing sex values:")
        print(missing_sex)
    pats = pats[pats['sex'].notna()]
    
    # drop pats with procedure_name not in list and print warning
    nontx_pats = pats.loc[~(pats['procedure_name'].str.contains('liver transplant|liver allotransplant', regex=True, case=False)), 'person_id'].values.tolist()
    if len(nontx_pats) > 0:
        print(f"Dropping {len(nontx_pats)} patients with non-transplant procedure names:")
        print(nontx_pats)
    pats = pats[pats['procedure_name'].str.contains('liver transplant|liver allotransplant', regex=True, case=False)]
    pats = pats.drop(columns=['procedure_name'])
    
    pats['age_at_tx'] = ((pats['transplant_date'] - pats['birth_date']).dt.days / 365.25).round(2)
    pats = pats.drop(columns=['birth_date'])
        
    # drop pats < 18 at tx
    if len(pats[pats['age_at_tx'] < 18]) > 0:
        print(f"Dropping {len(pats[pats['age_at_tx'] < 18])} patients < 18 at transplant:")
    pats = pats[pats['age_at_tx'] >= 18]
    
    print("Cohort size:", len(pats))
    
    return pats


def process_deaths(cohort, deaths):
    ''' DEATHS_CTE table processing.
        columns:
            - person_id
            - death_date -> convert to datetime
        new colums:
            - CENSOR_DATE -> date of death or study end date for those with no death date
            
    '''
    
    print("Processing death and censoring dates...")
    deaths['death_date'] = pd.to_datetime(deaths['death_date'], format='mixed')
    # merge with cohort on person_id
    cohort = pd.merge(cohort, deaths[['person_id', 'death_date']], on='person_id', how='left')
    # set the censor date to the death date or study end date
    cohort['CENSOR_DATE'] = cohort['death_date'].fillna(pd.to_datetime(project_lists.STUDY_CUTOFF_DATE))
    # drop the death date column
    cohort = cohort.drop(columns=['death_date'])
    
    # drop patients censored less than 1.25 years post-transplant
    censored = cohort[cohort['CENSOR_DATE'] < (cohort['transplant_date'] + pd.DateOffset(years=1, months=3))]
    if len(censored) > 0:
        print(f"Dropping {len(censored)} patients with censor date < 1.25 years post-transplant:")
        print(censored['person_id'].values.tolist())

    cohort = cohort[cohort['CENSOR_DATE'] >= (cohort['transplant_date'] + pd.DateOffset(years=1, months=3))]

    print("Cohort size:", len(cohort))

    return cohort

def process_smoke(cohort, smoke):
    ''' SMOKING_CTE table processing.
        columns:
            - person_id
            - smoking_code
            - smoking_status -> codes for current, former, never.
            - observation_date
        new columns:
            - SMOKER -> binary: 0 if never, 1 else.
    '''
    
    # smoke['code'] = smoke['smoking_status'].apply(lambda x: project_lists.SMOKER_INV[x] if x in project_lists.SMOKER_INV else np.nan)
    # smoker_ids = smoke.loc[(smoke['code'] in project_lists.SMOKER_CONCEPT_CODES), 'person_id'].values.tolist()
    
    smoker_ids = smoke.loc[smoke['smoking_code'].isin(project_lists.SMOKER_CONCEPT_CODES), 'person_id'].values.tolist()
    cohort['SMOKER'] = cohort['person_id'].apply(lambda x: 1 if x in smoker_ids else 0)
    
    return cohort
    
    
def process_inds(cohort, inds, pats):
    ''' INDICATION_CTE table processing.
        columns:
            - person_id
            - diagnosis_date -> convert to date, keep only the one at the final transplant
            - diagnosis
            - icd10_code -> map from codes to indication columns
            
        new columns:
            - METAB -> metabolic syndrome as indication
            - ALD -> alcoholic liver disease as indication
            - CANCER -> HCC as indication
            - HEP -> hepatitis as indication
            - FULM -> fulminant liver failure as indication
            - IMMUNE -> autoimmune liver disease as indication
            - RE_TX -> re-transplant as indication
    '''
    inds['diagnosis_date'] = pd.to_datetime(inds['diagnosis_date'], format='mixed')
    
    # the code can be a prefix of the full code, so we need to check for that
    inds['METAB'] = inds['icd10_code'].apply(lambda x: 1 if any([x.startswith(c) for c in project_lists.METAB_CODES]) else 0)    
    inds['ALD'] = inds['icd10_code'].apply(lambda x: 1 if any([x.startswith(c) for c in project_lists.ALD_CODES]) else 0)
    inds['CANCER'] = inds['icd10_code'].apply(lambda x: 1 if any([x.startswith(c) for c in project_lists.CANCER_CODES]) else 0)
    inds['HEP'] = inds['icd10_code'].apply(lambda x: 1 if any([x.startswith(c) for c in project_lists.HEP_CODES]) else 0)
    inds['FULM'] = inds['icd10_code'].apply(lambda x: 1 if any([x.startswith(c) for c in project_lists.FULM_CODES]) else 0)
    inds['IMMUNE'] = inds['icd10_code'].apply(lambda x: 1 if any([x.startswith(c) for c in project_lists.IMMUNE_CODES]) else 0)
    ## do RE_TX separately using the pats table !
    ## inds['RE_TX'] = inds['icd10_code'].apply(lambda x: 1 if any([x.startswith(c) for c in project_lists.RE_TX_CODES]) else 0)

    # merge all the rows of each patient into a single row
    inds = inds.groupby('person_id').agg({'METAB':'max', 'ALD':'max', 'CANCER':'max', 'HEP':'max', 'FULM':'max', 'IMMUNE':'max', 'diagnosis_date':'min'}).reset_index()

    
    ## C-S cohort has unexpectedly very high number of FULM - this is unlikely,
    ## probably these were coded with K72.0 due to some difference in coding practice
    ## Exclude any of these that are also one of the other conditions
    inds['FULM'] = (inds['FULM'] & ~(inds['METAB'] | inds['ALD'] | inds['CANCER'] | inds['HEP'] | inds['IMMUNE'])).astype(int)
        
    
    cohort = pd.merge(cohort, inds, on='person_id', how='left')
    # TODO: Figure out best way to do this date filtering.
    # pre_tx_timedelta = (cohort['transplant_date'] - cohort['diagnosis_date']).dt.days
    # # cohort = cohort.loc[((cohort['RE_TX']==1)&(pre_tx_timedelta>1))|
    # #                     ((pre_tx_timedelta <= 365)&(pre_tx_timedelta >= 0))]
    # cohort = cohort.loc[(pre_tx_timedelta >= 0)]
        
    cohort = cohort.drop(columns = ['diagnosis_date'])
    
    pats['transplant_date'] = pd.to_datetime(pats['transplant_date'], format='mixed')

    pats = pats.sort_values(['person_id','transplant_date'])
    def had_prior_tx(subdf):
        last_date = subdf['transplant_date'].dt.normalize().iloc[-1]
        earlier_dates = subdf['transplant_date'].dt.normalize() < last_date  # strictly earlier
        return int(earlier_dates.any())

    indicator = (
        pats.groupby('person_id')
        .apply(had_prior_tx)
        .rename('RE_TX')
    )
    
    cohort = cohort.merge(indicator, left_on='person_id', how='left', right_index=True)
    
    return cohort


# Helper function to apply conditions across time
def mark_condition(cohort, dhd, condition_name, code_list):
    end_date = pd.to_datetime(project_lists.STUDY_CUTOFF_DATE)
    max_years = int((end_date - cohort['transplant_date'].min()).days / 365.25)
    condition_df = dhd[dhd['icd10_code'].str.startswith(tuple(code_list))]
    merged = condition_df.merge(cohort[['person_id', 'transplant_date']], on='person_id', how='left')
    merged['years_since_tx'] = ((merged['diagnosis_date'] - merged['transplant_date']).dt.days / 365.25)

    for i in range(1, max_years + 1):
        hits = merged.loc[merged['years_since_tx'] < (i+0.25), 'person_id'].unique()
        cohort[f'{condition_name}_{i}'] = cohort['person_id'].isin(hits).astype(int)
    return cohort


def process_dhd(cohort, dhd):
    ''' DHD_CTE table processing.
        columns:
            - person_id
            - diagnosis_date -> convert to date, keep only the one at the final transplant
            - diagnosis
            - icd10_code -> map from codes to indication columns
            
        new columns:
            - DIABETES_<yr> -> diabetes as diagnosis
            - HYPERTENSION_<yr> -> hypertension as diagnosis
            - DYSLIPIDEMIA_<yr> -> dyslipidemia as diagnosis
            
            
        NOTE: we construct the columns for each year from the transplant date.
        NOTE: Years are offset by 3 months, basically we start counting from 3 months post-tx.         
    '''
    
    # for each patient create columns DM_1, DM_2,... HTN_1... LIP_1... 
    # up to now from 3 months after the transplant
    dhd['diagnosis_date'] = pd.to_datetime(dhd['diagnosis_date'], format='mixed')

    # Apply for each condition
    cohort = mark_condition(cohort, dhd, 'DM', project_lists.DM_CODES)
    cohort = mark_condition(cohort, dhd, 'HTN', project_lists.HTN_CODES)
    cohort = mark_condition(cohort, dhd, 'LIP', project_lists.LIP_CODES)

    return cohort
    
def match_chronic(df,codes):    
    is_match = df['icd10_code'].str.startswith(tuple(codes))
    # Get first match per patient
    first_match = df[is_match].groupby('person_id', as_index=False).first()
    # Get all non-matching rows
    non_match = df[~is_match]
    # Combine them
    result = pd.concat([non_match, first_match], ignore_index=True).sort_values(['person_id', 'diagnosis_date'])
    return result

def group_events(df, gap=30):
    df = df.sort_values(by = ['person_id','diagnosis_date']).reset_index(drop=True)
    df_to_collapse = df.copy()
    # compute gap
    df_to_collapse["prev_date"] = df_to_collapse.groupby(["person_id","icd10_code"])["diagnosis_date"].shift()
    df_to_collapse["days_since_prev"] = (df_to_collapse["diagnosis_date"] - df_to_collapse["prev_date"]).dt.days

    # new cluster whenever first event or gap exceeded
    df_to_collapse["new_cluster"] = (df_to_collapse["days_since_prev"].isna()) | (df_to_collapse["days_since_prev"] > gap)

    # cluster id
    df_to_collapse["cluster_id"] = df_to_collapse.groupby(["person_id","icd10_code"])["new_cluster"].cumsum()

    # now: keep *first event in each cluster* only
    collapsed = df_to_collapse.groupby(["person_id","icd10_code","cluster_id"]).first().reset_index()
    result = collapsed.sort_values(["person_id","diagnosis_date"]).reset_index(drop=True)
    
    return result


def process_events(cohort, events):
    ''' CV_EVENTS_CTE table processing.
        columns:
            - person_id
            - diagnosis_date
            - diagnosis 
            - icd10_code 
            
        new columns:
            - CV_HISTORY_<yr> -> past cardiovascular event as diagnosis
            - MONTHS_TO_EVENT_<yr> -> months to event for each year
    '''
    events['diagnosis_date'] = pd.to_datetime(events['diagnosis_date'], format='mixed')
    
    # drop anything here that is not coded as a CV event
    events = events[events['icd10_code'].str.startswith(tuple(project_lists.CV_CODES))]
    events = events[events['person_id'].isin(cohort['person_id'])]
    
    # try to group repeated event codes into a single event - window = 7 days
    events = group_events(events, project_lists.CV_EVENT_GAP_DAYS)
    
    cohort = mark_condition(cohort, events, 'CV_HISTORY', project_lists.CV_CODES)
    
    # now we need to calculate the time to next event for each year
    merged = events.merge(cohort[['person_id', 'transplant_date']], on='person_id', how='left')

    # Get max duration (e.g., 10 years follow-up)
    max_years = int((pd.to_datetime(project_lists.STUDY_CUTOFF_DATE) - cohort['transplant_date'].min()).days / 365.25)

    # Sort for quick lookup
    merged = merged.sort_values(by=['person_id', 'diagnosis_date'])
    
    
    # chronic events are counted as events only the first time    
    merged = match_chronic(merged,project_lists.CAD_CHRONIC_CODES) 
    merged = match_chronic(merged,project_lists.ARYTHMIA_CHRONIC_CODES)
    merged = match_chronic(merged,project_lists.VALV_CHRONIC_CODES)
    merged = match_chronic(merged,project_lists.HEART_FAIL_CHRONIC_CODES)
    merged = match_chronic(merged,project_lists.CEREBRO_CHRONIC_CODES)
    merged = match_chronic(merged,project_lists.ACS_CHRONIC_CODES)


    # Loop through each follow-up year
    for i in range(1, max_years + 1):
        colname = f'MONTHS_TO_EVENT_{i}'
        cohort[colname] = np.nan

        for idx, row in cohort.iterrows():
            pid = row['person_id']
            anchor_date = row['transplant_date'] + pd.DateOffset(years=i,months=3)

            # Get all  diagnoses for this patient after the anchor date
            future_df = merged[(merged['person_id'] == pid) & (merged['diagnosis_date'] > anchor_date)]

            if not future_df.empty:
                next_event_date = future_df['diagnosis_date'].iloc[0]
                time_to_event = (next_event_date - anchor_date).days / 30.4
                cohort.at[idx, colname] = time_to_event
        cohort[colname] = cohort[colname].round()
                
    return cohort, merged


def process_labs(cohort, labs):
    ''' LABS_CTE table processing.
        columns:
            - person_id
            - measurement_date
            - test_name 
            - test_code
            - test_value
            - test_unit
            
        new columns:
            - <lab>_<yr> -> lab value for each lab for each year
        NOTE: forward fill labs
        NOTE: labs we use: ALT, ALP, AST, BMI, CREATININE, CYCLO, TAC
    '''
    # filter out any labs from before 3 months post-tx
    labs['measurement_date'] = pd.to_datetime(labs['measurement_date'], format='mixed')
    lab_df = labs.merge(cohort[['person_id', 'transplant_date']], on='person_id', how='left')
    labs = lab_df[lab_df['measurement_date'] >= (lab_df['transplant_date'] + pd.DateOffset(months=3))].copy()
    labs = labs.sort_values(['person_id', 'measurement_date'])
    lab_cols = ['ALT', 'ALP', 'AST', 'BMI', 'CREATININE', 'CYCLO', 'TAC']
    
    max_years = int((pd.to_datetime(project_lists.STUDY_CUTOFF_DATE) - cohort['transplant_date'].min()).days / 365.25)
    new_cols_df = pd.DataFrame(np.nan,index=cohort.index, columns=[f'{lab}_{i}' for lab in lab_cols for i in range(1, max_years + 1)])
    cohort = pd.concat([cohort, new_cols_df], axis=1)
    for lab in lab_cols:
        lab_subset = labs[labs['test_code'].isin(project_lists.LABS_DICT[lab])].copy()
        for i in range(1, max_years + 1):
            col_name = f'{lab}_{i}'

            # Define the window of interest per patient
            for idx, row in cohort.iterrows():
                pid = row['person_id']
                anchor_date = row['transplant_date'] + pd.DateOffset(years=i, months=3)

                # Filter lab values before (or at) anchor date
                labs_for_patient = lab_subset[lab_subset['person_id'] == pid]
                labs_before = labs_for_patient[labs_for_patient['measurement_date'] <= anchor_date]

                if not labs_before.empty:
                    most_recent = labs_before.sort_values('measurement_date', ascending=False).iloc[0]['test_value']
                    cohort.at[idx, col_name] = most_recent
                    
    # adjust tac and cyclo:
    # if *any* tac values for a patient across all years, set all their cyclo to 0.
    # if no tac values for a patient, and any cyclo values, set all tac to 0.
    
    tac_cols = [f'TAC_{i}' for i in range(1, max_years + 1)]
    cyclo_cols = [f'CYCLO_{i}' for i in range(1, max_years + 1)]
    for idx, row in cohort.iterrows():
        pid = row['person_id']
        tac_values = [row[col] for col in tac_cols]
        cyclo_values = [row[col] for col in cyclo_cols]
        
        if any(tac_values):
            # set all cyclo values to 0
            for col in cyclo_cols:
                cohort.at[idx, col] = 0
        elif not any(tac_values) and any(cyclo_values):
            # set all tac values to 0
            for col in tac_cols:
                cohort.at[idx, col] = 0
    
    return cohort


def process_meds(cohort, meds):
    ''' MEDS_CTE table processing.
        columns:
            - person_id
            - start_date
            - end_date
            - medication_name
            - medication_code
            - dosage
            
        new columns:
            - ANTI_HTN_<yr>
            - ANTI_PLATELET_<yr>
            - STATIN_<yr>
        NOTE: forward fill meds
    '''
    meds['start_date'] = pd.to_datetime(meds['start_date'], format='mixed')
    meds = meds.sort_values(['person_id', 'start_date'])
    
    med_cols = ['ANTI_HTN', 'ANTI_PLATELET', 'STATIN']
    
    max_years = int((pd.to_datetime(project_lists.STUDY_CUTOFF_DATE) - cohort['transplant_date'].min()).days / 365.25)

    for med in med_cols:
        med_subset = meds[meds['medication_code'].isin(project_lists.MEDS_DICT[med])].copy()
        # keep only the first medication per patient
        med_subset = med_subset.groupby('person_id').first().reset_index()
        cohort = cohort.merge(med_subset[['person_id', 'start_date']], on='person_id', how='left')
        for i in range(1, max_years + 1):
            col_name = f'{med}_{i}'
            cohort[col_name] = 0
            cohort[col_name] = (cohort['start_date'] <= (cohort['transplant_date'] + pd.DateOffset(years=i, months=3))).astype(int)
            # Drop the start_date column
        cohort = cohort.drop(columns=['start_date'])
        
    return cohort

def add_dhd(cohort, labs):
    ''' Additional updates to the diseases:
        - Anyone on ANTI_HTN meds has HTN, anyone on STATIN has LIP
        - Any LDL > 4.1 or tryglycerides > 2.3 or Total cholesterol > 5.2 inidicative of LIP 
    '''
    
    max_yrs = int((pd.to_datetime(project_lists.STUDY_CUTOFF_DATE) - cohort['transplant_date'].min()).days / 365.25)
    for i in range(1, max_yrs + 1):
        cohort[f'HTN_{i}'] |= cohort[f'ANTI_HTN_{i}']
        cohort[f'LIP_{i}'] |= cohort[f'STATIN_{i}']
        
    labs_subset = labs[labs['test_name'].isin([x for k in project_lists.LIP_LAB_IDS.keys() for x in project_lists.LIP_LAB_IDS[k]])].copy()
    
    # TODO: finish this with triglycerides and total cholesterol
    labs_subset = labs_subset[(labs_subset['test_name'].isin(project_lists.LIP_LAB_IDS['LDL'])&\
                                labs_subset['test_value'] > 4.1)]
    labs_subset = labs_subset.sort_values(['person_id', 'measurement_date'])
    # take the first irregular lab for each patient
    labs_subset = labs_subset.groupby('person_id').first().reset_index()
    # merge with cohort
    cohort = cohort.merge(labs_subset[['person_id', 'measurement_date']], on='person_id', how='left')
    # for each year, if the lab is before the anchor date, set LIP_<yr> to 1
    for i in range(1, max_yrs + 1):
        cohort[f'LIP_{i}'] |= (cohort['measurement_date'] <= (cohort['transplant_date'] + pd.DateOffset(years=i,months=3))).astype(int)
    cohort.drop(columns=['measurement_date'], inplace=True)
    
    return cohort
    
def get_cohort_info(cohort, processed_events, outdir):
    ''' Dump demographic info and stats on the cohort to a json file.
    '''
    
    demo_dict = {}
    demo_dict['Females'] = cohort['sex'].sum()
    demo_dict['Males'] = len(cohort) - demo_dict['Females']
    demo_dict['Age'] = {'Median': cohort['age_at_tx'].median(), 'Lower': cohort['age_at_tx'].quantile(0.25),
                        'Upper': cohort['age_at_tx'].quantile(0.75)}
    demo_dict['Current, ex-smokers'] = cohort['SMOKER'].sum()
    demo_dict['Indications'] = {x:cohort[x].sum() for x in ['METAB', 'ALD', 'CANCER', 'HEP', 'FULM', 'IMMUNE', 'RE_TX']}
        
    bin_varying = ['DM', 'HTN', 'LIP', 'CV_HISTORY', 'ANTI_HTN', 'ANTI_PLATELET', 'STATIN']
    max_years = int((pd.to_datetime(project_lists.STUDY_CUTOFF_DATE) - cohort['transplant_date'].min()).days / 365.25)
    for v in bin_varying:
        demo_dict[v] = {'First':cohort[f'{v}_1'].sum(), 'Last':cohort[f'{v}_{max_years}'].sum()}
    
    labs = ['ALT', 'ALP', 'AST', 'BMI', 'CREATININE', 'CYCLO', 'TAC']
    for l in labs:
        demo_dict[l] = {}
        demo_dict[l]['Missing, final'] = cohort[f'{l}_{max_years}'].isna().sum()
        demo_dict[l]['Median, final'] = cohort[f'{l}_{max_years}'].median()
        demo_dict[l]['Lower, final'] = cohort[f'{l}_{max_years}'].quantile(0.25)        
        demo_dict[l]['Upper, final'] = cohort[f'{l}_{max_years}'].quantile(0.75)
    # for tac and cyclosporine median, upper, lower should be of the non-zero. Missing should include zeros:
    for l in ['TAC', 'CYCLO']:
        demo_dict[l]['Median, final'] = cohort[f'{l}_{max_years}'][cohort[f'{l}_{max_years}'] > 0].median()
        demo_dict[l]['Lower, final'] = cohort[f'{l}_{max_years}'][cohort[f'{l}_{max_years}'] > 0].quantile(0.25)        
        demo_dict[l]['Upper, final'] = cohort[f'{l}_{max_years}'][cohort[f'{l}_{max_years}'] > 0].quantile(0.75)
        demo_dict[l]['Missing, final'] = cohort[f'{l}_{max_years}'].isna().sum() + cohort[f'{l}_{max_years}'][cohort[f'{l}_{max_years}'] == 0].sum()
    
    # number of first events for a patient, median, upper, and lower MONTHS_TO_EVENT - for first event
    # number of rows with a non-null value in one of the MONTHS_TO_EVENT columns
    first_event_times = cohort['MONTHS_TO_EVENT_1'].dropna()
    median = first_event_times.median()
    lower_q = first_event_times.quantile(0.25)
    upper_q = first_event_times.quantile(0.75)
    demo_dict['CV_EVENTS'] = {}
    demo_dict['CV_EVENTS']['First'] = {'N':len(first_event_times),
                                     'Median':median,
                                     'Lower':lower_q, 'Upper':upper_q}
    processed_events = processed_events[processed_events['diagnosis_date'] >= (processed_events['transplant_date'] + pd.DateOffset(months=15))]
    
    # times between events for the same patient
    times_df = processed_events[['person_id','diagnosis_date','transplant_date']].copy()
    times_df = times_df.drop_duplicates(subset=['person_id','diagnosis_date'])
    times_df = times_df.sort_values(by=['person_id','diagnosis_date']).reset_index(drop=True)
    times_df["months_since_prev"] = times_df.groupby("person_id")["diagnosis_date"].diff().dt.days / 30.4
    times_df["months_since_prev"] = times_df["months_since_prev"].fillna((((times_df["diagnosis_date"] - (times_df["transplant_date"])).dt.days) / 30.4)-15)
    median = times_df['months_since_prev'].median()
    lower_q = times_df['months_since_prev'].quantile(0.25)
    upper_q = times_df['months_since_prev'].quantile(0.75)
    
    demo_dict['CV_EVENTS']['Total'] = {'N':len(times_df),
                                       'Median':median,
                                       'Lower':lower_q, 'Upper':upper_q}   
    demo_dict['CV_EVENTS']['Total']['Arrhythmia'] = processed_events['icd10_code'].str.startswith(tuple(ARYTHMIA_CODES)).sum().sum()
    demo_dict['CV_EVENTS']['Total']['Valvular'] = processed_events['icd10_code'].str.startswith(tuple(VALV_CODES)).sum().sum()
    demo_dict['CV_EVENTS']['Total']['ACS'] = processed_events['icd10_code'].str.startswith(tuple(ACS_CODES)).sum().sum()
    demo_dict['CV_EVENTS']['Total']['CAD'] = processed_events['icd10_code'].str.startswith(tuple(CAD_CODES)).sum().sum()
    demo_dict['CV_EVENTS']['Total']['Cerebrovascular'] = processed_events['icd10_code'].str.startswith(tuple(CEREBRO_CODES)).sum().sum()
    demo_dict['CV_EVENTS']['Total']['Heart failure'] = processed_events['icd10_code'].str.startswith(tuple(HF_CODES)).sum().sum()
    
    print(demo_dict)
    
    def convert_to_native(obj):
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(v) for v in obj]
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif pd.isna(obj):
            return None
        return obj

    demo_dict_clean = convert_to_native(demo_dict)
    
    # save the cohort info to a json file
    with open(os.path.join(outdir, 'cohort_info.json'), 'w') as f:
        json.dump(demo_dict_clean, f, indent=4)        


def get_prediction_cohort(cohort):
    '''
        For each patient, take the follow up year with the least missingness.
        If there is a tie, take the earlier one.
        Set up the censoring dates
    '''
    pred_cohort = cohort[['person_id', 'transplant_date', 'CENSOR_DATE', 'sex', 'age_at_tx', 'SMOKER', 'METAB', 'ALD', \
                          'CANCER', 'HEP', 'FULM', 'IMMUNE', 'RE_TX']].copy()
    
    max_year = int((pd.to_datetime(project_lists.STUDY_CUTOFF_DATE) - cohort['transplant_date'].min()).days / 365.25)
    years = list(range(1, max_year + 1))
    
    years_to_censor = (cohort['CENSOR_DATE'] - cohort['transplant_date']).dt.days / 365.25
    
    missingness = pd.DataFrame(0,index=cohort.index, columns=years)
    lab_cols = ['ALT', 'ALP', 'AST', 'BMI', 'CREATININE', 'CYCLO', 'TAC']
    for col in lab_cols:
        for year in years:
            col_name = f'{col}_{year}'
            # fill any value past years_to_censor with nan
            cohort.loc[years_to_censor<year+0.25 ,col_name] = np.nan
            missingness[year] += (cohort[col_name].isnull()).astype(int)
    best_year = missingness.idxmin(axis=1)
    
    selected_values = []

    varying_cols = lab_cols + ['DM', 'HTN', 'LIP', 'CV_HISTORY', 'ANTI_HTN', 'ANTI_PLATELET', 'STATIN', 'MONTHS_TO_EVENT']
    for idx, year in best_year.items():
        patient_values = []
        for c in varying_cols:
            col_name = f'{c}_{year}'
            patient_values.append(cohort.at[idx, col_name])
        selected_values.append(patient_values)
        
    pred_cohort[varying_cols] = pd.DataFrame(selected_values, columns=varying_cols)
    pred_cohort['YRS_SINCE_TRANS'] = best_year + 0.25
    pred_cohort['CURR_AGE'] = pred_cohort['age_at_tx']+pred_cohort['YRS_SINCE_TRANS']
    
    pred_cohort['EVENT'] = pred_cohort['MONTHS_TO_EVENT'].notnull().astype(int)
    # fill null MONTHS_TO_EVENT with end point - anchor date (date of transplant + years since transplant)
    pred_cohort['anchor_dates'] = pred_cohort['transplant_date'] + pd.to_timedelta(
                                        (pred_cohort['YRS_SINCE_TRANS'] * 365.25).round().astype(int), unit="D") #pred_cohort.apply(lambda row: row['transplant_date'] + pd.DateOffset(days = int(row['YRS_SINCE_TRANS']*365.25)), axis=1)
    pred_cohort['MONTHS_TO_EVENT'] = (pred_cohort['MONTHS_TO_EVENT'].fillna((pred_cohort['CENSOR_DATE'] - pred_cohort['anchor_dates']).dt.days / 30.4)).round()
    pred_cohort.drop(columns = ['transplant_date','CENSOR_DATE','anchor_dates'], inplace=True)
    
    pred_cohort.rename(columns={'age_at_tx':'AGE_AT_TX','person_id':'ID','sex':'SEX', 'CYCLO':'CYCLOSPORINE_TROUGH_LEVEL',
                                'TAC':"TACROLIMUS_TROUGH_LEVEL",'CREATININE' : "SERUM_CREATININE"}, inplace=True)
    
    return pred_cohort
    
    

def main(pats_path, smoke_path, inds_path, dhd_path, events_path, labs_path, meds_path, deaths_path, outdir):
    
    pats = pd.read_csv(pats_path)
    smoke = pd.read_csv(smoke_path)
    inds = pd.read_csv(inds_path)
    dhd = pd.read_csv(dhd_path)
    events = pd.read_csv(events_path)
    labs = pd.read_csv(labs_path)
    meds = pd.read_csv(meds_path)
    deaths = pd.read_csv(deaths_path)
    
    # process the patients
    print("Processing patients...")
    cohort = process_pats(pats)
    
    # add in death dates as censoring dates
    print("Adding death dates...")
    cohort = process_deaths(cohort, deaths)
           
    # process the smoking history
    print("Processing smoking history...")
    cohort = process_smoke(cohort, smoke)
    
    # process the indications
    print("Processing indications...")
    cohort = process_inds(cohort, inds, pats)
    
    # process the diabetes, hypertension, dyslipidemia health statuses
    print("Processing diabetes, hypertension, dyslipidemia...")
    cohort = process_dhd(cohort, dhd)
    
    print("Processing cardiovascular events...")
    cohort, processed_events = process_events(cohort, events)
    
    print("Processing labs...")
    cohort = process_labs(cohort, labs)
    
    print("Processing medications...")
    cohort = process_meds(cohort, meds)
    
    # add to diabetes etc based on the meds and labs
    print("Updating diseases based on labs and meds...")
    cohort = add_dhd(cohort, labs)
    
    # stats on the cohort
    get_cohort_info(cohort, processed_events, outdir)
    
    # save the cohort
    print("Saving preprocessed cohort...")
    cohort.to_csv(os.path.join(outdir, 'preprocessed_cohort_WIDE.csv'), index=False)
    
    # pick a date for each patient from which to predict.
    # Use the minimum date with the maximum features.
    print("Getting final cohort for predictions...")
    prediction_cohort = get_prediction_cohort(cohort)
    print("Saving prediction cohort csv to pass to model...")
    prediction_cohort.to_csv(os.path.join(outdir, 'prediction_cohort.csv'), index=False)
    


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Command line args")

    # Add a command-line argument
    parser.add_argument('--pats', type=str, required=True, help='Path to the PATS table csv file')
    parser.add_argument('--smoke', type=str, required=True, help='Path to the SMOKING_CTE table csv file')
    parser.add_argument('--inds', type=str, required=True, help='Path to the INDICATION_CTE table csv file')
    parser.add_argument('--dhd', type=str, required=True, help='Path to the DHD_CTE table csv file')
    parser.add_argument('--events', type=str, required=True, help='Path to the CV_EVENTS_CTE table csv file')
    parser.add_argument('--labs', type=str, required=True, help='Path to the LABS_CTE table csv file')
    parser.add_argument('--meds', type=str, required=True, help='Path to the MEDS_CTE table csv file')
    parser.add_argument('--deaths', type=str, required=True, help='Path to the DEATHS_CTE table csv file')
    parser.add_argument('--outdir', type=str, required=False, help='Output directory')
    
    args = parser.parse_args()
    
    main(args.pats, args.smoke, args.inds, args.dhd, args.events, args.labs, args.meds, args.deaths, args.outdir)
