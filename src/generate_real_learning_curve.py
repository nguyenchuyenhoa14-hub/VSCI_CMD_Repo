import numpy as np, time, os, sys
from scipy.optimize import minimize_scalar
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, KFold, cross_val_predict, GridSearchCV, learning_curve
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.kernel_ridge import KernelRidge
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin
from run_final_dual_pipeline import run_unified_pipeline
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 8
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False


import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Helvetica Neue', 'DejaVu Sans']
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def main():
    DIR = r'C:\Users\Nguyen\Documents\JHL_SVD\Paper_nearfield\CPC_submission\data'
    LUT_PATH = os.path.join(DIR, 'tsv_physics_lut_multimode.npz')
    CACHE_TR = os.path.join(DIR, 'fdtd_cache_highres_training_1629s.npz')
    
    lut = np.load(LUT_PATH)
    tr = np.load(CACHE_TR)
    y_tr = tr['labels']
    
    # We patch run_unified_pipeline to return the features by copying the code we need or just running it and catching?
    # Wait, run_unified_pipeline doesn't return Xt_b_final.
    pass

if __name__ == '__main__':
    main()
