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
from sklearn.metrics import roc_curve, auc

sys.path.insert(0, "/home/oem/PycharmProjects/LielTools")
from sklearn.model_selection import RepeatedKFold
from catboost import CatBoostClassifier

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

def is_it_a_categorial_col(col):
    for num in col:
        if num - int(num) != 0:
            return False
    return True


def plot_auc(y_test_auc, y_score_auc, y_auc,model_auc):
    n_classes = y_auc.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_auc[:, i], y_score_auc[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.figure()
        lw = 2
        plt.plot(
            fpr[2],
            tpr[2],
            color="darkorange",
            lw=lw,
            label="feture " + i + ' roc: ' + roc_auc[2],
        )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("model: ",model_auc)
    plt.legend(loc="lower right")
    plt.show()


# ------------------------------------------------------------------------ </editor-fold>




# --------------------------------""" global variables """---------------- <editor-fold>
SEED = 11
starttime = time.time()
random.seed(SEED)
# ------------------------------------------------------------------------ </editor-fold>

# --------------------------------""" load data """----------------------- <editor-fold>
data = pd.read_csv(os.path.join(PROJECT_DIR, "X_y_train.csv"))
number_of_unique_per_fitcher = data.apply(lambda col: col.nunique(), axis=0)
data = data.drop(number_of_unique_per_fitcher.loc[number_of_unique_per_fitcher < 2].index, axis=1)

#  cheaking if there is null cells
print(data.isnull().sum().sum())

# looking for str values

dty = data.dtypes

for _ in dty:
    if _ == 'object':
        print(_)

# dealing with the categorical data
likely_cat = {}
for var in data.columns:
    likely_cat[var] = 1. * data[var].nunique() / data[var].count() < 0.05

# standard scaler for all non categorical


cat_list = []
scaler_list = []
for col in data.columns:
    if likely_cat[col]:
        cat_list.append(col)
    else:
        scaler_list.append(col)

scaler = StandardScaler()

##################### for train ##############################
data_scalar = pd.DataFrame(scaler.fit_transform(data[scaler_list]),columns=scaler_list)

# creat dummis for all the categorical values

all_dummy_df = pd.get_dummies(data=data, columns=cat_list[:-1], drop_first=True)
all_dummy_df = data[cat_list[:-1]]

con_df = pd.concat([all_dummy_df, data_scalar], axis=1)
#change all the colomns name to be sort by X1 X2...
con_df.columns = con_df.columns.map(lambda x: int(x[1:]))
con_df = con_df.sort_index(axis=1)
con_df.columns = con_df.columns.map(lambda x: "X"+str(x))

# data2 = pd.read_csv(os.path.join(PROJECT_DIR, "tp53_features_train_smiles.csv"), index_col=0)
# scaler = StandardScaler()
# data_orig = data.copy()
X = con_df
y = data.iloc[:, -1]

select_featchers_by_lasso_df = select_featchers_by_lasso(X,y)
X_by_lasso = X.loc[:,select_featchers_by_lasso_df[0].index]
the_test = pd.read_csv(os.path.join(PROJECT_DIR, "X_test.csv"), index_col=0)
the_test_by_lasso = the_test.loc[:,select_featchers_by_lasso_df[0].index]
objects = []
if os.path.exists(os.path.join(PROJECT_DIR, "the_parameters_to_choose_for_each_model.pkl")):
    with (open(os.path.join(PROJECT_DIR, "the_parameters_to_choose_for_each_model.pkl"), "rb")) as openfile:
        try:
            the_parameters_to_choose_for_each_model = pickle.load(openfile)
        except EOFError:
            print()
##################### for test ##############################
# new_the_test = the_test.iloc[:,1:]
new_the_test = the_test.reset_index(drop=True)
data_scalar_test = pd.DataFrame(scaler.fit_transform(new_the_test[scaler_list]),columns=scaler_list)

# creat dummis for all the categorical values

# all_dummy_df_test = pd.get_dummies(data=new_the_test,columns=cat_list[:-1], drop_first=True)
all_dummy_df_test = new_the_test[cat_list[:-1]]

x_test = pd.concat([all_dummy_df_test, data_scalar_test], axis=1)
x_test.set_index(the_test.index,inplace=True)
#change all the colomns name to be sort by X1 X2...
x_test.columns = x_test.columns.map(lambda x: int(x[1:]))
x_test = x_test.sort_index(axis=1)
x_test.columns = x_test.columns.map(lambda x: "X"+str(x))

print(X.shape)
print(x_test.shape)



# ------------------------------------------------------------------------ </editor-fold>

# --------------------------------""" see the data """----------------------- <editor-fold>
describtion = data.describe().T
info = data.info()


if os.path.exists(os.path.join(PROJECT_DIR, "got_to_know_data", "corr_matrix_pearson.csv")):
    pearson_corr = pd.read_csv(os.path.join(PROJECT_DIR, "got_to_know_data", "corr_matrix_pearson.csv"), index_col=0)
else:
    pearson_corr = data.corr(method="pearson")
    pearson_corr.to_csv(os.path.join(PROJECT_DIR, "got_to_know_data", "corr_matrix_pearson.csv"))

if os.path.exists(os.path.join(PROJECT_DIR,"got_to_know_data","corr_matrix_spearman.csv")):
    spearman_corr = pd.read_csv(os.path.join(PROJECT_DIR,"got_to_know_data","corr_matrix_spearman.csv"),index_col=0)
else:
    spearman_corr = data.corr(method="spearman")
    spearman_corr.to_csv(os.path.join(PROJECT_DIR, "got_to_know_data", "corr_matrix_spearman.csv"))

spearman_corr_nul = spearman_corr.iloc[:-1, :-1].applymap(lambda x: x if x > 0.75 or x < -0.75 else np.nan)
for x in range(len(spearman_corr_nul.index)):
    spearman_corr_nul.iloc[x, x] = np.nan
spearman_corr_nul_sum = spearman_corr_nul.notnull().sum()
fetcher_high_corlate = spearman_corr_nul.loc[spearman_corr_nul_sum > 0, spearman_corr_nul_sum > 0]

correlated_features = set()
for i in range(len(fetcher_high_corlate.columns)):
    for j in range(i):
        if abs(fetcher_high_corlate.iloc[i, j]) > 0.75:
            colname = fetcher_high_corlate.columns[i]
            correlated_features.add(colname)
X = X.drop(correlated_features, axis=1)
the_test = the_test.drop(correlated_features, axis=1)

# ------------------------------------------------------------------------ </editor-fold>
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, stratify=y)
cv = RepeatedKFold(n_splits=5, n_repeats=2,
                   random_state=1)  # Repeated k-fold cross-validation has the benefit of improving the estimate of the mean model performance at the cost of fitting and evaluating many more models.


def add_hyper_tuning(the_parameters_to_choose_for_each_model, model, param_grid, cv):
    model_name = str(type(model)).rstrip("'>").split(".")[-1]
    try:
        the_parameters_to_choose_for_each_model[model_name]
    except KeyError:
        #grid_cv = GridSearchCV(model, param_grid, cv=cv, scoring="accuracy")
        grid_cv = GridSearchCV(model, param_grid, cv=cv,scoring="roc_auc_ovo_weighted")
        grid_cv.fit(X, y)
        prediction = grid_cv.predict(the_test)
        prediction_df = pd.DataFrame(prediction, columns=["y_pred"], index=the_test.index)
        prediction_df.name = "ID"
        prediction_df.to_csv(os.path.join(PROJECT_DIR, model_name + ".csv"))
        print(str(os.path.join(PROJECT_DIR, model_name + ".csv")))
        inner_parametes_dict = {"Parameters": grid_cv.best_params_,
                                "best_cv_score": grid_cv.best_score_,
                                "prediction": prediction,
                                "prediction_df": prediction_df}
        the_parameters_to_choose_for_each_model[model_name] = inner_parametes_dict
    return the_parameters_to_choose_for_each_model

def add_hyper_tuning_lasso_selected(the_parameters_to_choose_for_each_model,model,param_grid,cv):
    model_name=str(type(model)).rstrip("'>").split(".")[-1]
    try:
        the_parameters_to_choose_for_each_model[model_name]
    except KeyError:
        grid_cv = GridSearchCV(model, param_grid, cv=cv,scoring="roc_auc_ovo_weighted")
        grid_cv.fit(X_by_lasso, y)
        prediction = grid_cv.predict(the_test_by_lasso)
        prediction_df = pd.DataFrame(prediction,columns=["y_pred"],index=the_test.index)
        prediction_df.name = "ID"
        prediction_df.to_csv(os.path.join(PROJECT_DIR,model_name+"lasso.csv"))
        print(str(os.path.join(PROJECT_DIR,model_name+"lasso.csv")))
        inner_parametes_dict = {"Parameters":grid_cv.best_params_,
                                "best_cv_score":grid_cv.best_score_,
                                "prediction":prediction,
                                "prediction_df":prediction_df}
        the_parameters_to_choose_for_each_model[model_name] = inner_parametes_dict
    a_file = open(os.path.join(PROJECT_DIR, "the_parameters_to_choose_for_each_model.pkl"), "ab")
    pickle.dump(the_parameters_to_choose_for_each_model, a_file)
    a_file.close()
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
param_grid = {"max_depth": range(1, 20),
              "min_samples_leaf": range(1, 20),
              "criterion": ["gini", "entropy"]}
the_parameters_to_choose_for_each_model = add_hyper_tuning(
    the_parameters_to_choose_for_each_model=the_parameters_to_choose_for_each_model,
    model=tree,
    param_grid=param_grid,
    cv=cv)


clf = SVC()
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001],
              'kernel': ['rbf', 'poly', 'sigmoid', "linear", "precomputed"]}
the_parameters_to_choose_for_each_model = add_hyper_tuning(
    the_parameters_to_choose_for_each_model=the_parameters_to_choose_for_each_model,
    model=clf,
    param_grid=param_grid,
    cv=cv)

trees = RandomForestClassifier()
param_grid = {'n_estimators': list(range(10, 101, 10)),
              'max_features': list(range(6, 32, 5))}
the_parameters_to_choose_for_each_model = add_hyper_tuning(
    the_parameters_to_choose_for_each_model=the_parameters_to_choose_for_each_model,
    model=trees,
    param_grid=param_grid,
    cv=cv)

# GPC = GaussianProcessClassifier()
# param_grid = [{
#     "alpha":  [1e-2, 1e-3],
#     "kernel": [RBF(l) for l in np.logspace(-1, 1, 2)]
# }, {
#     "alpha":  [1e-2, 1e-3],
#     "kernel": [DotProduct(sigma_0) for sigma_0 in np.logspace(-1, 1, 2)]
# }]
# the_parameters_to_choose_for_each_model = add_hyper_tuning(
#     the_parameters_to_choose_for_each_model=the_parameters_to_choose_for_each_model,
#     model=GPC,
#     param_grid=param_grid,
#     cv=cv)


mlp = MLPClassifier()
param_grid = {'solver': ['lbfgs'], 'max_iter': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000],
              'alpha': 10.0 ** -np.arange(1, 10), 'hidden_layer_sizes': np.arange(10, 15),
              'random_state': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}
the_parameters_to_choose_for_each_model = add_hyper_tuning(
    the_parameters_to_choose_for_each_model=the_parameters_to_choose_for_each_model,
    model=mlp,
    param_grid=param_grid,
    cv=cv)


qda = QuadraticDiscriminantAnalysis()
param_grid = {'reg_param': [0.1, 0.2, 0.3, 0.4, 0.5]}
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
ABC = AdaBoostClassifier(base_estimator=DTC)
param_grid = {"base_estimator__criterion": ["gini", "entropy"],
              "base_estimator__splitter": ["best", "random"],
              "n_estimators": [1, 2, 3, 4, 5, 6, 7]
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
prediction_df = pd.DataFrame(predictions, columns=["y_pred"], index=the_test.index)
prediction_df.name = "ID"
prediction_df.to_csv(os.path.join(PROJECT_DIR, "autosklearn" + ".csv"))
the_parameters_to_choose_for_each_model["autosklearn"] = {"Parameters": np.nan,
                                                          "best_cv_score": np.nan,
                                                          "prediction": predictions,
                                                          "prediction_df": prediction_df}

a_file = open(os.path.join(PROJECT_DIR, "the_parameters_to_choose_for_each_model.pkl"), "ab")
pickle.dump(the_parameters_to_choose_for_each_model, a_file)
a_file.close()

# ---------------------------------XGBoost------------------------------- <editor-fold>


from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

model = XGBClassifier()
learning_rate = [0.0001, 0.01, 0.1, 0.2, 0.3]
param_grid = dict(learning_rate=learning_rate)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)

grid_search = GridSearchCV(model, param_grid, scoring='neg_log_loss', n_jobs=1, cv=kfold)
grid_cv = grid_search.fit(X, y)

prediction = grid_cv.predict(x_test)
prediction_df = pd.DataFrame(prediction, columns=["y_pred"], index=x_test.index)
prediction_df.name = "ID"
prediction_df.to_csv(os.path.join(PROJECT_DIR, "xgboost" + ".csv"))

# ---------------------------------catboost------------------------------- <editor-fold>


from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer

# Grid search cv
clf = CatBoostClassifier()

params = {'iterations': [500,1000,1500],
          'depth': [2, 4, 5],
          'loss_function': ['MultiClass','log_loss'],
          'l2_leaf_reg': np.logspace(-20, -19, 3),
          'leaf_estimation_iterations': [10],
          'logging_level': ['Silent'],
          'random_seed': [42]
          }

scorer = make_scorer(accuracy_score)
clf_grid = GridSearchCV(estimator=clf, param_grid=params, scoring=scorer, cv=5)

clf_grid.fit(X, y)
best_param = clf_grid.best_params_
print('grid search best params: ')
print(best_param)
#
# # use_best_model params to prevent model overfitting
model = CatBoostClassifier(iterations=500,
                           loss_function='MultiClass',
                           depth=4,
                           l2_leaf_reg=best_param['l2_leaf_reg'],
                           # eval_metric='Accuracy',
                           leaf_estimation_iterations=10,
                           # use_best_model=True,
                           logging_level='Silent',
                           random_seed=42
                           )


# train the model
# model.fit(X, y, eval_set=(x_validation, y_validation),)
model.fit(X, y)

# make the prediction using the resulting model
preds_class = model.predict(x_test)
# preds_proba = model.predict_proba(x_test)
# print("class = ", preds_class)
# print("proba = ", preds_proba)
cat_pred = preds_class.flatten()

prediction_df = pd.DataFrame(cat_pred, columns=["y_pred"], index=x_test.index)
prediction_df.name = "ID"
prediction_df.to_csv(os.path.join(PROJECT_DIR, "Catboost" + ".csv"))


# plot_auc(y_test_auc, y_score_auc, y_auc=y,model_auc='Catboost')
# ------------------------------------------------------------------------ </editor-fold>
