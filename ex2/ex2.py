import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process.kernels import RBF, DotProduct
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import random
import time
import os
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPClassifier
sys.path.insert(0, "/home/oem/PycharmProjects/LielTools")
from sklearn.model_selection import RepeatedKFold


the_parameters_to_choose_for_each_model = {}
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
    lasso_coef_orig = lasso_coef.copy()
    # Plot the coefficients
    lasso_coef = lasso_coef.loc[lasso_coef["coef"] != 0]

    lasso_coef.index = list(X_train.iloc[:,lasso_coef.index])
    plt.plot(lasso_coef.index, lasso_coef["coef"])
    plt.xticks(lasso_coef.index, lasso_coef.index, rotation=60)
    plt.title("selected featchers by lasso")
    plt.margins(0.02)
    plt.show()

    return lasso_coef,lasso_coef_orig


# ------------------------------------------------------------------------ </editor-fold>




# --------------------------------""" global variables """---------------- <editor-fold>
SEED = 11
starttime = time.time()
random.seed(SEED)
# ------------------------------------------------------------------------ </editor-fold>

# --------------------------------""" load data """----------------------- <editor-fold>
data = pd.read_csv(os.path.join(PROJECT_DIR, "X_y_train.csv"))
#data2 = pd.read_csv(os.path.join(PROJECT_DIR, "tp53_features_train_smiles.csv"), index_col=0)
scaler = StandardScaler()
data_orig = data.copy()
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X = pd.DataFrame(scaler.fit_transform(X),columns=X.columns)
the_test = pd.read_csv(os.path.join(PROJECT_DIR, "X_test.csv"), index_col=0)
objects = []
if os.path.exists(os.path.join(PROJECT_DIR, "the_parameters_to_choose_for_each_model.pkl")):
    with (open(os.path.join(PROJECT_DIR, "the_parameters_to_choose_for_each_model.pkl"), "rb")) as openfile:
        try:
            the_parameters_to_choose_for_each_model = pickle.load(openfile)
        except EOFError:
            print()


# ------------------------------------------------------------------------ </editor-fold>

# --------------------------------""" see the data """----------------------- <editor-fold>
describtion = data.describe().T
info = data.info()

number_of_unique_per_fitcher = data.apply(lambda col: col.nunique(),axis=0)
data = data.drop(number_of_unique_per_fitcher.loc[number_of_unique_per_fitcher<2].index,axis=1)

if os.path.exists(os.path.join(PROJECT_DIR,"got_to_know_data","corr_matrix_pearson.csv")):
    pearson_corr = pd.read_csv(os.path.join(PROJECT_DIR,"got_to_know_data","corr_matrix_pearson.csv"),index_col=0)
else:
    pearson_corr = data.corr(method="pearson")
    pearson_corr.to_csv(os.path.join(PROJECT_DIR, "got_to_know_data", "corr_matrix_pearson.csv"))

if os.path.exists(os.path.join(PROJECT_DIR,"got_to_know_data","corr_matrix_spearman.csv")):
    spearman_corr = pd.read_csv(os.path.join(PROJECT_DIR,"got_to_know_data","corr_matrix_spearman.csv"),index_col=0)
else:
    spearman_corr = data.corr(method="spearman")
    spearman_corr.to_csv(os.path.join(PROJECT_DIR, "got_to_know_data", "corr_matrix_spearman.csv"))

spearman_corr_nul = spearman_corr.iloc[:-1,:-1].applymap(lambda x: x if x>0.75 or x<-0.75 else np.nan)
for x in range(len(spearman_corr_nul.index)):
    spearman_corr_nul.iloc[x,x]=np.nan
spearman_corr_nul_sum = spearman_corr_nul.notnull().sum()
fetcher_high_corlate = spearman_corr_nul.loc[spearman_corr_nul_sum>0,spearman_corr_nul_sum>0]

correlated_features=set()
for i in range(len(fetcher_high_corlate .columns)):
    for j in range(i):
        if abs(fetcher_high_corlate.iloc[i, j]) > 0.75:
            colname = fetcher_high_corlate.columns[i]
            correlated_features.add(colname)
X = X.drop(correlated_features,axis=1)
the_test = the_test.drop(correlated_features,axis=1)

# ------------------------------------------------------------------------ </editor-fold>
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0,stratify=y)
cv = RepeatedKFold(n_splits=5, n_repeats=2,
                   random_state=1)  # Repeated k-fold cross-validation has the benefit of improving the estimate of the mean model performance at the cost of fitting and evaluating many more models.


def add_hyper_tuning(the_parameters_to_choose_for_each_model,model,param_grid,cv):
    model_name=str(type(knn)).rstrip("'>").split(".")[-1]
    try:
        the_parameters_to_choose_for_each_model[model_name]
    except KeyError:
        grid_cv = GridSearchCV(model, param_grid, cv=cv,scoring="accuracy")
        grid_cv.fit(X, y)
        prediction = grid_cv.predict(the_test)
        prediction_df = pd.DataFrame(prediction,columns=["y_pred"],index=the_test.index)
        prediction_df.name = "ID"
        prediction_df.to_csv(os.path.join(PROJECT_DIR,model_name+".csv"))
        inner_parametes_dict = {"Parameters":grid_cv.best_params_,
                                "best_cv_score":grid_cv.best_score_,
                                "prediction":prediction,
                                "prediction_df":prediction_df}
        the_parameters_to_choose_for_each_model[model_name] = inner_parametes_dict
    return the_parameters_to_choose_for_each_model

knn = KNeighborsClassifier()
n_space = list(range(1, 31))
param_grid = {'n_neighbors': n_space}
the_parameters_to_choose_for_each_model = add_hyper_tuning(
    the_parameters_to_choose_for_each_model=the_parameters_to_choose_for_each_model,
    model=knn,
    param_grid=param_grid,
    cv=cv)

logreg = LogisticRegression()
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space, 'penalty': ['l1', 'l2']}
the_parameters_to_choose_for_each_model = add_hyper_tuning(
    the_parameters_to_choose_for_each_model=the_parameters_to_choose_for_each_model,
    model=logreg,
    param_grid=param_grid,
    cv=cv)

tree = DecisionTreeClassifier()
param_grid = {"max_depth": range[1, 20],
              "min_samples_leaf": range[1, 20],
              "criterion": ["gini", "entropy"]}
the_parameters_to_choose_for_each_model = add_hyper_tuning(
    the_parameters_to_choose_for_each_model=the_parameters_to_choose_for_each_model,
    model=tree,
    param_grid=param_grid,
    cv=cv)


clf = SVC()
param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid',"linear","precomputed"]}
the_parameters_to_choose_for_each_model = add_hyper_tuning(
    the_parameters_to_choose_for_each_model=the_parameters_to_choose_for_each_model,
    model=clf,
    param_grid=param_grid,
    cv=cv)

trees = RandomForestClassifier()
param_grid ={'classifier__n_estimators': list(range(10, 101, 10)),
            'classifier__max_features': list(range(6, 32, 5))}
the_parameters_to_choose_for_each_model = add_hyper_tuning(
    the_parameters_to_choose_for_each_model=the_parameters_to_choose_for_each_model,
    model=trees,
    param_grid=param_grid,
    cv=cv)

GPC = GaussianProcessClassifier()
param_grid = [{
    "alpha":  [1e-2, 1e-3],
    "kernel": [RBF(l) for l in np.logspace(-1, 1, 2)]
}, {
    "alpha":  [1e-2, 1e-3],
    "kernel": [DotProduct(sigma_0) for sigma_0 in np.logspace(-1, 1, 2)]
}]
the_parameters_to_choose_for_each_model = add_hyper_tuning(
    the_parameters_to_choose_for_each_model=the_parameters_to_choose_for_each_model,
    model=GPC,
    param_grid=param_grid,
    cv=cv)


mlp = MLPClassifier()
param_grid = {'solver': ['lbfgs'], 'max_iter': [1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000 ], 'alpha': 10.0 ** -np.arange(1, 10), 'hidden_layer_sizes':np.arange(10, 15), 'random_state':[0,1,2,3,4,5,6,7,8,9]}
the_parameters_to_choose_for_each_model = add_hyper_tuning(
    the_parameters_to_choose_for_each_model=the_parameters_to_choose_for_each_model,
    model=mlp,
    param_grid=param_grid,
    cv=cv)


qda = QuadraticDiscriminantAnalysis()
param_grid={'reg_param': [0.1, 0.2, 0.3, 0.4, 0.5]}
the_parameters_to_choose_for_each_model = add_hyper_tuning(
    the_parameters_to_choose_for_each_model=the_parameters_to_choose_for_each_model,
    model=qda,
    param_grid=param_grid,
    cv=cv)

nb_classifier = GaussianNB()
param_grid = {'var_smoothing': np.logspace(0,-9, num=100)}
the_parameters_to_choose_for_each_model = add_hyper_tuning(
    the_parameters_to_choose_for_each_model=the_parameters_to_choose_for_each_model,
    model=nb_classifier,
    param_grid=param_grid,
    cv=cv)

DTC = DecisionTreeClassifier(**the_parameters_to_choose_for_each_model["DecisionTreeClassifier"])
ABC = AdaBoostClassifier(base_estimator = DTC)
param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"],
              "n_estimators": [1, 2,3,4,5,6,7]
             }
the_parameters_to_choose_for_each_model = add_hyper_tuning(
    the_parameters_to_choose_for_each_model=the_parameters_to_choose_for_each_model,
    model=ABC,
    param_grid=param_grid,
    cv=cv)



# knn_cv = GridSearchCV(knn,param_grid,cv=cv)
# knn_cv.fit(X,y)
# print("Tuned knn_cv Parameters: {}".format(knn_cv.best_params_))
# print("Best knn_cv score is {}".format(knn_cv.best_score_))
# prediction = knn_cv.predict(the_test)


#select_featchers_by_lasso_df = select_featchers_by_lasso(X_train,y_train)

import autosklearn.classification
cls = autosklearn.classification.AutoSklearnClassifier()
cls.fit(X, y)
predictions = cls.predict(the_test)
prediction_df = pd.DataFrame(predictions,columns=["y_pred"],index=the_test.index)
prediction_df.name = "ID"
prediction_df.to_csv(os.path.join(PROJECT_DIR,"autosklearn"+".csv"))
the_parameters_to_choose_for_each_model["autosklearn"]= {"Parameters":np.nan,
                                "best_cv_score":np.nan,
                                "prediction":predictions,
                                "prediction_df":prediction_df}


a_file = open(os.path.join(PROJECT_DIR, "the_parameters_to_choose_for_each_model.pkl"), "wb")
pickle.dump(the_parameters_to_choose_for_each_model, a_file)
a_file.close()
