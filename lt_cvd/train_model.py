import os, argparse, pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
import project_lists

from run_model import run_predictions, run_evaluations, save_results

from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv


def train_model(train):
    
    rsf = RandomSurvivalForest(max_depth = 9, n_estimators = 500)
    train['MONTHS_TO_EVENT'] = train['MONTHS_TO_EVENT'].round()
    X = train[[c for c in train.columns if c not in ['MONTHS_TO_EVENT','EVENT', 'ID']]]
    y = Surv.from_dataframe('EVENT', 'MONTHS_TO_EVENT', train)
       
    rsf.fit(X,y)
    
    return rsf
    

def process_df(df):
    data=[]
    static_cols = ['person_id', 'transplant_date', 'CENSOR_DATE', 'sex', 'age_at_tx', 'SMOKER', 'METAB', 'ALD', \
                          'CANCER', 'HEP', 'FULM', 'IMMUNE', 'RE_TX']
    var_col_names = ['ALT', 'ALP', 'AST', 'BMI', 'CREATININE', 'CYCLO', 'TAC', 'DM', 'HTN', 'LIP', 'CV_HISTORY', 'ANTI_HTN', 'ANTI_PLATELET', 'STATIN', 'MONTHS_TO_EVENT']
    new_cols = ['YRS_SINCE_TRANS',"CURR_AGE"]
    max_year = int((pd.to_datetime(project_lists.STUDY_CUTOFF_DATE) - df['transplant_date'].min()).days / 365.25)
    years = list(range(1, max_year + 1))
    df['transplant_date'] = pd.to_datetime(df['transplant_date'])
    df['CENSOR_DATE'] = pd.to_datetime(df['CENSOR_DATE'])
    
    
    df['YRS_TO_CENSOR'] = ((df['CENSOR_DATE'] - df['transplant_date']).dt.days / 365.25)
    for idx, row in df.iterrows():
        static_data = row[static_cols].tolist()
        censor_time = row['YRS_TO_CENSOR']
        for y in years:
            if censor_time < y + 0.25:
                break
            var_cols = [f'{col}_{y}' for col in var_col_names]
            var_data = row[var_cols].tolist()
            new_data = [y+0.25, row['age_at_tx'] + y + 0.25]
            row_data = static_data + var_data + new_data
            data.append(row_data)
    df_long = pd.DataFrame(data, columns = static_cols + var_col_names + new_cols)
    
    
    # add EVENT column using censoring - as was done for pred cohort
    df_long['EVENT'] = df_long['MONTHS_TO_EVENT'].notnull().astype(int)
    # fill null MONTHS_TO_EVENT with end point - anchor date (date of transplant + years since transplant)
    df_long['anchor_dates'] = df_long['transplant_date'] + pd.to_timedelta(
                                        (df_long['YRS_SINCE_TRANS'] * 365.25).round().astype(int), unit="D") 
    df_long['MONTHS_TO_EVENT'] = (df_long['MONTHS_TO_EVENT'].fillna((df_long['CENSOR_DATE'] - df_long['anchor_dates']).dt.days / 30.4)).round()
    
    df_long = df_long.drop(columns = ['transplant_date','CENSOR_DATE','anchor_dates',"YRS_TO_CENSOR"])
    
    df_long = df_long.rename(columns={'age_at_tx':'AGE_AT_TX','person_id':'ID','sex':'SEX', 'CYCLO':'CYCLOSPORINE_TROUGH_LEVEL',
                                'TAC':"TACROLIMUS_TROUGH_LEVEL",'CREATININE' : "SERUM_CREATININE"})
    
    
    col_order = ['ID', 'AGE_AT_TX', 'CURR_AGE', 'YRS_SINCE_TRANS', 'SEX', 'SMOKER', 'DM', 'HTN', 'LIP',
                 'CV_HISTORY', 'ANTI_PLATELET', 'ANTI_HTN', 'STATIN', 'BMI', 'CANCER',  'METAB',
                 'ALD', 'HEP', 'FULM', 'IMMUNE', 'RE_TX', "CYCLOSPORINE_TROUGH_LEVEL",
                 "TACROLIMUS_TROUGH_LEVEL", "ALP", "ALT", "AST", "SERUM_CREATININE",
                 'MONTHS_TO_EVENT', 'EVENT']           
    df_long = df_long[col_order]
    
    df_long = df_long[df_long['AGE_AT_TX'] >= 18]
    df_long = df_long[df_long['YRS_SINCE_TRANS'] >= 1]
    
    return df_long

def get_normalizer(df,n_bins=50):
    norm_cols = ["AGE_AT_TX",  "ALP", "ALT", "AST", "BMI", "CURR_AGE", "CYCLOSPORINE_TROUGH_LEVEL",   "SERUM_CREATININE",  "TACROLIMUS_TROUGH_LEVEL", "YRS_SINCE_TRANS"]
    norm_cols = [c for c in norm_cols if c in df.columns]
    normalizer = {}
    for col in norm_cols:
        binner = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
        binner.fit(df[col].dropna().values.reshape(-1,1))
        normalizer[col] = binner
    return normalizer

def normalize_df(df, normalizer,n_bins=50):
    med_bin = (n_bins-1) // 2
    df_scaled = df.copy()
    norm_cols = list(normalizer.keys())
    for col in norm_cols:
        transformed_values = np.full(df[col].shape, np.nan)
        non_nan_mask = ~df[col].isna()  # Mask for non-NaN values
        transformed_values[non_nan_mask] = normalizer[col].transform(df.loc[non_nan_mask, [col]])[:, 0]
        df_scaled[col] = transformed_values
        df_scaled[col] = df_scaled[col].fillna(med_bin)
    return df_scaled


def run_test(test_df, train_df, rsf, outdir):
    
    # select a random time for each patient
    test_df_single_times = (train_df.groupby("ID", group_keys=False).sample(n=1, random_state=54213))    
    preds = run_predictions(test_df,rsf)
        
    c_ind, brier, cd_auc, wm_sae, binned_results   = run_evaluations(preds, test_df_single_times, train_df)
    
    save_results(preds, c_ind, brier, cd_auc, wm_sae, binned_results, outdir)   

def main(subjects_file, outdir):
    # load the cohort
    df = pd.read_csv(subjects_file)
    
    # process df into correct format
    if not os.path.exists(os.path.join(outdir,'preprocessed_cohort_LONG.csv')):
        df = process_df(df)
        df.to_csv(os.path.join(outdir,'preprocessed_cohort_LONG.csv'), index=False)
    else:
        df = pd.read_csv(os.path.join(outdir,'preprocessed_cohort_LONG.csv'))
    
    # split into train and test sets - hold out all rows for test patients
    ids = pd.Series(df['ID'].unique())
    ids = ids.sample(frac=1, random_state=54213).reset_index(drop=True)  # shuffle ids
    TRAIN_FRAC = 0.8
    train_df = df[df['ID'].isin(ids[:int(TRAIN_FRAC*len(ids))])]
    test_df = df[df['ID'].isin(ids[int(TRAIN_FRAC*len(ids)):])]
    
    # normalize / bin train and then test
    norm = get_normalizer(train_df)
    train_df = normalize_df(train_df, norm)
    test_df = normalize_df(test_df, norm)
    
    # train model
    rsf = train_model(train_df)
    
    # save the model
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    with open(os.path.join(outdir,'rsf.pkl'),'wb') as f:
        pickle.dump(rsf,f)
    with open(os.path.join(outdir,'norm.pkl'),'wb') as f:
        pickle.dump(norm,f)
        
    # run evalution on the test set
    run_test(test_df, train_df, rsf,outdir)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Command line args")

    # Add a command-line argument
    parser.add_argument('--subjects_file', type=str, required=True, help='cohort_WIDE file path')
    parser.add_argument('--outdir', type=str, required=False, help='Output directory')
    
    args = parser.parse_args()
    
    main(args.subjects_file, args.outdir)