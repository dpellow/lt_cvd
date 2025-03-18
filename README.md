# lt_cvd

### Getting started
Clone the repo and set up your python environment. The [scikit-survival](https://scikit-survival.readthedocs.io/en/stable/install.html), [shap](https://shap.readthedocs.io/en/latest/#install), and [matplotlib](https://matplotlib.org/stable/install/index.html) packages are required.

### Running the code
Run the `run_model.py` script that is in the `lt_cvd` directory:
```
usage: python run_model.py  --model_dir MODEL_DIR (the directory with the rsf.pkl and norm.pkl files - they are in lt_cvd in the repo)
                            --subjects_file SUBJECTS_FILE (csv file with your subjects, their features, and their labels)
                            [--outdir OUTDIR] (output directory)
```
