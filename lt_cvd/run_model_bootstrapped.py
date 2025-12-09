# Load and run a model on subjects in an input csv file
import argparse
import os
import pickle

import pandas as pd
import numpy as np

from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv 
from sksurv.metrics import concordance_index_censored, brier_score, cumulative_dynamic_auc

from matplotlib import pyplot as plt

from lifelines import KaplanMeierFitter

import shap

from train_model import process_df, normalize_df, run_test_bootstrap


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
    
    explainer_file = os.path.join(model_path, 'explainer.pkl') 
    with open(explainer_file, "rb") as f:
        explainer = pickle.load(f)
    
    # needed to compute brier score
    training_distr = pd.read_csv(os.path.join(model_path, 'training_distr.csv'), index_col=0)
        
    return model, bins, training_distr, explainer



def run_predictions(df, rsf):
    print("Running predictions")
    chfs = rsf.predict_survival_function(df.drop(columns=['ID','EVENT','MONTHS_TO_EVENT']))
    ten_yr_risk = 1 - np.array([f([120]) for f in chfs])
    preds = pd.DataFrame(ten_yr_risk, columns=['10 year risk'],index=df['ID'])
    return preds

def make_structured_array(event, time):
    return np.array([(bool(e), t) for e, t in zip(event, time)],
                    dtype=[('event', 'bool'), ('time', 'f8')])
    
def compute_binwise_km_calibration(data, preds, t_eval=120):
    bins = [0.0, 0.075, 0.20, 1.0]
    labels = ['<7.5%', '7.5-20%', '>20%']
    data = data.copy()
    # data['predicted_event_prob'] = preds["10 year risk"]
    data = pd.merge(data, preds, on="ID", how="left")  ## updated 2025-09-12
    data = data.rename(columns={ 
        "10 year risk": "predicted_event_prob"
    })
    # data['bin'] = pd.cut(preds["10 year risk"], bins=bins, labels=labels, include_lowest=True)
    bin_data = pd.cut(preds.squeeze(), bins=bins, labels=labels, include_lowest=True) ## updated 2025-09-12
    data = pd.merge(data, bin_data, on="ID", how="left")  ## updated 2025-09-12
    data = data.rename(columns={
        "10 year risk": "bin"
    })
    results = []
    for label in labels:
        bin_df = data[data['bin'] == label]
        if len(bin_df) == 0:
            results.append({'bin': label,
                            'n': 0,
                            'mean_pred': np.nan,
                            'km_event_rate': np.nan,
                            'abs_error': np.nan})
            continue
        surv_data = make_structured_array(bin_df['EVENT'], bin_df['MONTHS_TO_EVENT'])
        
        kmf = KaplanMeierFitter()
        kmf.fit(surv_data['time'], event_observed=surv_data['event'])
        km_surv = kmf.survival_function_at_times(t_eval).values[0]
        km_event_rate = 1 - km_surv
        
        pred_mean = bin_df['predicted_event_prob'].mean()
        abs_error = abs(pred_mean - km_event_rate)

        results.append({
            'bin': label,
            'n': len(bin_df),
            'mean_pred': pred_mean,
            'km_event_rate': km_event_rate,
            'abs_error': abs_error
        })
    results = pd.DataFrame(results)
    weights = results['n'] / results['n'].sum()
    weighted_avg = (results['abs_error'] * weights).sum()
    print(results)
    return weighted_avg, results


def run_evaluations(preds, df, training_brier_distr):
    print("Evaluating predictions")
    yt = Surv.from_arrays(df['EVENT'], df['MONTHS_TO_EVENT'])
    y = Surv.from_arrays(training_brier_distr['EVENT'], training_brier_distr['MONTHS_TO_EVENT'])
    
    max_month = min(120,min(int(df['MONTHS_TO_EVENT'].max()),int(training_brier_distr['MONTHS_TO_EVENT'].max()))-1)
    print(max_month)
    c_ind = concordance_index_censored(df['EVENT'].astype(bool), df['MONTHS_TO_EVENT'], preds['10 year risk'])[0]
    _, brier = brier_score(y, yt, 1-preds['10 year risk'], max_month)
    auc, _ = cumulative_dynamic_auc(y, yt, preds['10 year risk'], [max_month])
    avg_calib, binned_results = compute_binwise_km_calibration(df, preds['10 year risk'], t_eval=120)
    print(f"Concordance index: {c_ind.round(3)}")
    print(f"Brier score: {brier.round(3)}")
    print(f"CD-AUC: {auc[0].round(3)}")
    print("Average absolute calibration error (binwise KM): ", avg_calib.round(4))
    return c_ind.round(4), brier.round(4), auc[0].round(4), avg_calib.round(4), binned_results


def save_results(preds, c_ind, brier, cd_auc, wm_sae, binned_results, outdir):
    print("Saving results")
    preds.to_csv(os.path.join(outdir, 'predictions.csv'))
    binned_results.to_csv(os.path.join(outdir, 'binned_calibration.csv'))
    with open(os.path.join(outdir, 'metrics.txt'), 'w') as f:
        f.write(f"Concordance index: {c_ind}\n")
        f.write(f"Brier score: {brier}\n")
        f.write(f"CD-AUC: {cd_auc}\n")
        f.write(f"wm-SAE: {wm_sae}\n")
    print("Results saved")



class RSFPredictWrapper:
    """Callable wrapper around an RSF model to make it picklable."""
    def __init__(self, rsf, t_eval):
        self.rsf = rsf
        self.t_eval = t_eval

    def __call__(self, X):
        surv_fns = self.rsf.predict_survival_function(X)
        risk_scores = np.array([1 - fn(self.t_eval[0]) for fn in surv_fns])
        return risk_scores

def run_shap(df, explainer, outdir):
    X_test = df.drop(columns=['ID','EVENT','MONTHS_TO_EVENT'])
    print("Running SHAP analysis")
    shap_values = explainer(X_test)
    
    shap.summary_plot(shap_values, X_test, feature_names=X_test.columns, show=False,
                      max_display=15, plot_size=[12,10])
    plt.tight_layout()
    plt.xticks(fontsize=14)
    plt.savefig(os.path.join(outdir, 'shap_summary_plot.png'),dpi=400)
    plt.close()
    
    return shap_values



def main(model_path, cohort_path, outdir):
    os.makedirs(outdir, exist_ok=True)
    # load the model (and binning info)
    rsf, bins, training_distr, explainer = load_model(model_path)
    
    df = pd.read_csv(cohort_file)
    df = process_df(df)
    norm_cols = ["AGE_AT_TX",  "ALP", "ALT", "AST", "BMI", "CURR_AGE", "CYCLOSPORINE_TROUGH_LEVEL",   "SERUM_CREATININE",  "TACROLIMUS_TROUGH_LEVEL", "YRS_SINCE_TRANS"]
    df = normalize_df(df, bins)

    run_test_bootstrap(df,training_distr, rsf, outdir)
    
    shap = run_shap(df,explainer, outdir)
    

    



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Command line args")

    # Add a command-line argument
    parser.add_argument('--model_dir', type=str, required=True, help='Path to the model directory')
    parser.add_argument('--subjects_file', type=str, required=True, help='csv file with the subjects to predict')
    parser.add_argument('--outdir', type=str, required=False, help='Output directory')
    
    args = parser.parse_args()
    
    main(args.model_dir, args.subjects_file, args.outdir)