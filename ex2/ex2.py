import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize
from pandas_profiling import ProfileReport
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
import random
import time
import os
import sys
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

sys.path.insert(0, "/home/oem/PycharmProjects/LielTools")
#sys.path.insert(0, "C:\\Users\\Eilay Koren\\PycharmProjects\\lab_projects\\LielTools_4")
import PlotTools
import DataTools
import FileTools
import importlib

import types
from deepchecks.tabular.suites import full_suite
from deepchecks.tabular import Dataset
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error as MSE  # numerical prediction
from rdkit import Chem

# ---------------------------------""" paths """-------------------------- <editor-fold>

PROJECT_DIR = os.path.dirname(os.path.abspath("ex2.py"))
PROJECT_DIR = os.path.join(PROJECT_DIR,"ex2")
if os.path.exists(PROJECT_DIR):
    print('found all paths  :) ')
else:
    print('cant find a path  :( ')
    exit()

# ------------------------------------------------------------------------ </editor-fold>


# ---------------------------------""" functions """-------------------------- <editor-fold>

def select_featchers_by_lasso(X_train, y_train):
    """
    selected featchers by lasso
    """
    # Instantiate a lasso regressor: lasso
    lasso = Lasso(alpha=0.4)

    # Fit the regressor to the data
    lasso.fit(X_train, y_train)

    # Compute and print the coefficients
    lasso_coef = pd.DataFrame(lasso.coef_, columns=["coef"])
    # Plot the coefficients
    lasso_coef = lasso_coef.loc[lasso_coef["coef"] != 0]
    print(lasso_coef)

    plt.plot(lasso_coef.index, lasso_coef["coef"])
    plt.xticks(lasso_coef.index, lasso_coef.index, rotation=60)
    plt.title("selected featchers by lasso")
    plt.margins(0.02)
    plt.show()

    lasso_coef.index = list(X_train.iloc[:,lasso_coef.index])
    return lasso_coef

# ------------------------------------------------------------------------ </editor-fold>




# --------------------------------""" global variables """---------------- <editor-fold>
SEED = 11
starttime = time.time()
random.seed(SEED)
# ------------------------------------------------------------------------ </editor-fold>

# --------------------------------""" load data """----------------------- <editor-fold>
data = pd.read_csv(os.path.join(PROJECT_DIR, "X_y_train.csv"), index_col=0)
#data2 = pd.read_csv(os.path.join(PROJECT_DIR, "tp53_features_train_smiles.csv"), index_col=0)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0,stratify=y)
# ------------------------------------------------------------------------ </editor-fold>

# --------------------------------""" see the data """----------------------- <editor-fold>
describtion = data.describe().T
info = data.info()

# ------------------------------------------------------------------------ </editor-fold>

select_featchers_by_lasso_df = select_featchers_by_lasso(X_train,y_train)


#https://askubuntu.com/questions/1298198/ubuntu-20-04-doesnt-wake-up-after-suspend
