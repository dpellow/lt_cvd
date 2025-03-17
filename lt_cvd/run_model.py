# Load and run a model on subjects in an input csv file
import argparse
import os
import pickle

import pandas as pd
import numpy as np

from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv 
from sksurv.metrics import concordance_index_censored, brier_score

from matplotlib import pyplot as plt

import warnings

# Suppress specific UserWarning from sklearn.utils.validation
warnings.filterwarnings("ignore", message="X has feature names, but KBinsDiscretizer was fitted without feature names", module="sklearn.utils.validation")


def load_model(model_path):
    print("Loading model")
    rsffile = os.path.join(model_path, 'rsf.pkl')
    with open(rsffile, 'rb') as f:
        model = pickle.load(f)
    
    bins_file = os.path.join(model_path, 'norm.pkl')
    with open(bins_file, 'rb') as f:
        bins = pickle.load(f)
    
    # needed to compute brier score
    training_distr = pd.read_csv(os.path.join(model_path, 'training_distr.csv'), index_col=0)
        
    return model, bins, training_distr

def preprocess(df):
    # add any columns that are missing (they will be imputed as missing values - warn)
    cols = ['ID', 'AGE_AT_TX', 'CURR_AGE', 'YRS_SINCE_TRANS',
            'SEX', 'SMOKER', 'DM', 'HTN', 'LIP', 'CV_HISTORY', 'ANTI_PLATELET',
            'ANTI_HTN', 'STATIN', 'BMI', 'CANCER', 'METAB', 'ALD',
            'HEP', 'FULM', 'IMMUNE', 'RE_TX', "CYCLOSPORINE_TROUGH_LEVEL",
            "TACROLIMUS_TROUGH_LEVEL", "ALP", "ALT", "AST", "SERUM_CREATININE",
            "EVENT", "MONTHS_TO_EVENT"]
    imp_cols = ["BMI","CYCLOSPORINE_TROUGH_LEVEL", "TACROLIMUS_TROUGH_LEVEL",
                "ALP", "ALT", "AST", "SERUM_CREATININE"]
    for c in cols:
        if c not in df.columns:
            if c in imp_cols:
                print(f"Warning: {c} not in dataframe - will be imputed as a constant for all patients")
                df[c] = np.nan
            else:
                print(f"Error: Feature {c} required. Update the cohort file to include this feature")
                raise ValueError(f"Column {c} not in dataframe")
    
    # check patient ages and drop those underage at tx
    # drop any less than 1 year post tx
    # print out ids of all dropped patients
    if len(df[df['AGE_AT_TX'] < 18]) > 0:
        print("Patients must be >18 years at transplant to be included. Dropping:")
        print(df[df['AGE_AT_TX'] < 18]['ID'].values.tolist())
    if len(df[df["YRS_SINCE_TRANS"] < 1]) > 0:
        print("Patients must be >1 year post transplant to be included. Dropping:")
        print(df[df["YRS_SINCE_TRANS"] < 1]['ID'].values.tolist())
    df = df[df['AGE_AT_TX'] >= 18]
    df = df[df['YRS_SINCE_TRANS'] >= 1]
    
    # re-order the columns, and keep only the ones that are needed
    df = df[cols]
    
    # preprocess tac/cyclo levels as follows:
    # if tac is not nan or 0, set cyclo to 0
    # if cyclo is not nan or 0, set tac to 0
    # if cyclo is nan set to 0
    df["CYCLOSPORINE_TROUGH_LEVEL"] = df["CYCLOSPORINE_TROUGH_LEVEL"].fillna(0)
    df.loc[((df["TACROLIMUS_TROUGH_LEVEL"].notna()) & (df["TACROLIMUS_TROUGH_LEVEL"]>0)),
                    "CYCLOSPORINE_TROUGH_LEVEL"] = 0
    df.loc[((df["CYCLOSPORINE_TROUGH_LEVEL"].notna()) & (df["CYCLOSPORINE_TROUGH_LEVEL"]>0)),
                    "TACROLIMUS_TROUGH_LEVEL"] = 0
    
    # constant imputation of nan to -1 in the impute columns
    
    df[imp_cols] = df[imp_cols].fillna(-1)
    
    # if any nans in the other columns - drop the row and warn
    if df[[c for c in cols if c in cols and c not in imp_cols]].isna().any().any():
        print("Dropping the following patients that have missing values in required columns:")
        print(df[df[[c for c in cols if c in cols and c not in imp_cols]].isna().any(axis=1)]['ID'].values.tolist())
    df = df.dropna(subset=[c for c in cols if c not in imp_cols])
    
    return df


def run_binning(df, bins):
    norm_cols = ["AGE_AT_TX","ALP", "ALT", "AST", "BMI", "CURR_AGE", "CYCLOSPORINE_TROUGH_LEVEL", 
                 "SERUM_CREATININE", "TACROLIMUS_TROUGH_LEVEL", "YRS_SINCE_TRANS"]
    for col in norm_cols:
        transformed_values = np.full(df[col].shape, -1) # nan value is -1
        non_nan_mask = ~(df[col]==-1)  # Mask for non-NaN values
        transformed_values[non_nan_mask] = bins[col].transform(df.loc[non_nan_mask, [col]])[:, 0]
        df[col] = transformed_values
    return df



def get_subjects(cohort_path, bins):
    print("Loading cohort")
    df = pd.read_csv(cohort_path)
    
    # preprocess the dataframe
    df = preprocess(df)
    
    df = run_binning(df,bins)

    return df


def run_predictions(df, rsf):
    print("Running predictions")
    chfs = rsf.predict_survival_function(df.drop(columns=['ID','EVENT','MONTHS_TO_EVENT']))
    ten_yr_risk = 1 - np.array([f([120]) for f in chfs])
    preds = pd.DataFrame(ten_yr_risk, columns=['10 year risk'],index=df['ID'])
    return preds


def run_evaluations(preds, df, training_brier_distr):
    print("Evaluating predictions")
    yt = Surv.from_arrays(df['EVENT'], df['MONTHS_TO_EVENT'])
    y = Surv.from_arrays(training_brier_distr['EVENT'], training_brier_distr['MONTHS_TO_EVENT'])
    c_ind = concordance_index_censored(df['EVENT'].astype(bool), df['MONTHS_TO_EVENT'], preds['10 year risk'])[0]
    _, brier = brier_score(y, yt, 1-preds['10 year risk'], 120)
    print(f"Concordance index: {c_ind.round(3)}")
    print(f"Brier score: {brier.round(3)}")
    return c_ind.round(4), brier.round(4)


def save_results(preds, c_ind, brier, outdir):
    print("Saving results")
    preds.to_csv(os.path.join(outdir, 'predictions.csv'))
    with open(os.path.join(outdir, 'metrics.txt'), 'w') as f:
        f.write(f"Concordance index: {c_ind}\n")
        f.write(f"Brier score: {brier}\n")
    print("Results saved")


def plot_calibration(preds, df, outdir):
    fig, ax = plt.subplots()
    print("Plotting calibration")
    
    # Merge pred into df on ID
    df = df.merge(preds, left_on='ID', right_index=True)
    df = df.rename(columns = {'10 year risk':'predicted_event_prob'})
    df = df.dropna(subset=['predicted_event_prob'])
        
    df['bin'] = pd.cut(preds['10 year risk'].values, np.arange(0,1.01,0.05), labels=False)
    # print(pd.qcut(predictions_at_t0, 100, labels=False))
    # df['bin'] = pd.qcut(predictions_at_t0, 100, labels=False)

    # Calculate observed event probabilities within each bin
    bin_data = df.groupby('bin').apply(lambda x: pd.Series({
        'predicted_prob': x['predicted_event_prob'].mean(),
        'observed_prob': np.mean((x['EVENT']) & (x['MONTHS_TO_EVENT'] <= 120)),
        'num_pred' : len(x),
        'num_obs' : sum((x['EVENT']) & (x['MONTHS_TO_EVENT'] <= 120))                                               
    })).reset_index()
    
    # Normalize the sizes for better visualization
    sizes = bin_data['num_pred']
    min_size = 20
    max_size = 300
    sizes = min_size + (max_size - min_size) * (sizes - sizes.min()) / (sizes.max() - sizes.min())
       
    # Plot the calibration plot
    scatter = ax.scatter(bin_data['predicted_prob'], bin_data['observed_prob'], s= sizes, alpha=0.7, label='Observed')
    corr = bin_data['predicted_prob'].corr(bin_data['observed_prob'],method='pearson')
    print(f"Correlation of predicted and actual {corr}")
    
    # add the correlation at the bottom right, round to 2 decimals
    ax.text(0.95, 0.05, f'Correlation: {corr:.2f}', ha='right', va='bottom', transform=ax.transAxes)
    
    ax.plot([0, 1], [0, 1], "k--", label='Ideal')
    ax.set_xlabel(f'Predicted event probability')
    ax.set_ylabel('Observed event probability')
    ax.set_title(f'Calibration Plot')
    
    # Create a legend for the sizes
    handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6)
    size_labels = [f'{int(size)} predictions' for size in np.linspace(bin_data['num_pred'].min(), bin_data['num_pred'].max(), num=5).astype(int)]
    size_handles = [plt.Line2D([], [], marker='o', linestyle='', alpha=0.6, markersize=np.sqrt(min_size + (max_size - min_size) * (size - bin_data['num_pred'].min()) / (bin_data['num_pred'].max() - bin_data['num_pred'].min())), color='b') for size in np.linspace(bin_data['num_pred'].min(), bin_data['num_pred'].max(), num=5)]

    # Add the size legend to the plot
    legend2 = ax.legend(size_handles, size_labels, title="Number of Predictions", loc="upper right", frameon=True)
    ax.add_artist(legend2)
    
    plt.savefig(os.path.join(outdir,'calibration.png'))


def main(model_path, cohort_path, outdir):
    
    # load the model (and binning info)
    rsf, bins, training_distr = load_model(model_path)
    
    df = get_subjects(cohort_path,bins)
    
    preds = run_predictions(df,rsf)
    
    c_ind, brier = run_evaluations(preds, df, training_distr)
    
    save_results(preds, c_ind, brier, outdir)
    
    plot_calibration(preds, df, outdir)
    
    return preds, c_ind, brier
    



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Command line args")

    # Add a command-line argument
    parser.add_argument('--model_dir', type=str, required=True, help='Path to the model directory')
    parser.add_argument('--subjects_file', type=str, required=True, help='csv file with the subjects to predict')
    parser.add_argument('--outdir', type=str, required=False, help='Output directory')
    
    args = parser.parse_args()
    
    main(args.model_dir, args.subjects_file, args.outdir)