"""
here i will save all the base lines that are importent hopefully well documented for you future Eilay
"""
from random import randint
import tensorflow as tf
from tensorflow import keras, matmul, Variable, estimator
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_val_score
from tensorflow import feature_column

X = pd.DataFrame()# or X = np.arrey
y = pd.DataFrame()# or X = np.arrey
digits = datasets.load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25, random_state=0,stratify=y)
"""
random_state is SEED
stratify is to split the data to save the distrebution of the y
"""


""" 
classification
"""
# <editor-fold> classification
""" 
    KNN spliter
"""
#<editor-fold>
from sklearn.neighbors import KNeighborsClassifier
# Create a k-NN classifier with 7 neighbors: knn
knn = KNeighborsClassifier(7)
# Fit the classifier to the training data
knn.fit(X_train,y_train)
# Print the accuracy
print(knn.score(X_test, y_test))
# </editor-fold>

""" 
    Building a logistic regression model
"""
#<editor-fold>
from sklearn.linear_model import LogisticRegression
# Create the classifier: logreg
logreg = LogisticRegression()

# Fit the classifier to the training data
logreg.fit(X_train,y_train)

# Predict the labels of the test set: y_pred
y_pred = logreg.predict(X_test)

# Create the hyperparameter grid
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space, 'penalty': ['l1', 'l2']}

# </editor-fold>

""" 
    Building a DecisionTreeClassifier
"""
#<editor-fold> DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
# Instantiate a Decision Tree classifier: tree
tree = DecisionTreeClassifier()
# </editor-fold>




# </editor-fold>

""" 
Regression
"""
# <editor-fold> regration

"""
    LinearRegression
"""
#<editor-fold>
from sklearn.linear_model import LinearRegression
# Create the regressor: reg
reg = LinearRegression()

X_fertility = [[2.73], [6.43], [2.24], [1.4 ], [1.96], [1.41], [1.99], [1.89], [2.38], [1.83], [1.42], [1.82], [2.91], [5.27], [2.51], [3.48], [2.86], [1.9 ], [1.43], [6.04], [6.48], [3.05], [5.17], [1.68], [6.81], [1.89], [2.43], [5.05], [5.1 ], [1.91], [4.91], [1.43], [1.5 ], [1.89], [3.76], [2.73], [2.95], [2.32], [5.31], [5.16], [1.62], [2.74], [1.85], [1.97], [4.28], [5.8 ], [1.79], [1.37], [4.19], [1.46], [4.12], [5.34], [5.25], [2.74], [3.5 ], [3.27], [1.33], [2.12], [2.64], [2.48], [1.88], [2.  ], [2.92], [1.39], [2.39], [1.34], [2.51], [4.76], [1.5 ], [1.57], [3.34], [5.19], [1.42], [1.63], [4.79], [5.78], [2.05], [2.38], [6.82], [1.38], [4.94], [1.58], [2.35], [1.49], [2.37], [2.44], [5.54], [2.05], [2.9 ], [1.77], [2.12], [2.72], [7.59], [6.02], [1.96], [2.89], [3.58], [2.61], [4.07], [3.06]]

# Create the prediction space
prediction_space = np.linspace(min(X_fertility), max(X_fertility)).reshape(-1,1)

# Fit the model to the data
reg.fit(X_fertility,y)

# Compute predictions over the prediction space: y_pred
y_pred = reg.predict(prediction_space)

# Print R^2
print(reg.score(X_fertility, y))

# Plot regression line
plt.plot(prediction_space, y_pred, color='black', linewidth=3)
plt.show()
# </editor-fold>
"""
    Lasso
"""
#<editor-fold>
from sklearn.linear_model import Lasso

# Instantiate a lasso regressor: lasso
lasso = Lasso(alpha=0.4,normalize=True)

# Fit the regressor to the data
lasso.fit(X,y)

# Compute and print the coefficients
lasso_coef = lasso.coef_
print(lasso_coef)
df_columns=['population', 'fertility', 'HIV', 'CO2', 'BMI_male', 'GDP','BMI_female', 'child_mortality']
# Plot the coefficients
plt.plot(range(len(df_columns)), lasso_coef)
plt.xticks(range(len(df_columns)), df_columns, rotation=60)
plt.margins(0.02)
plt.show()

# </editor-fold>
"""
    Ridge
"""
#<editor-fold>

from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# Setup the array of alphas and lists to store scores
alpha_space = np.logspace(-4, 0, 50)
ridge_scores = []
ridge_scores_std = []

# Create a ridge regressor: ridge
ridge = Ridge(normalize=True)

# Compute scores over range of alphas
for alpha in alpha_space:
    # Specify the alpha value to use: ridge.alpha
    ridge.alpha = alpha

    # Perform 10-fold CV: ridge_cv_scores
    ridge_cv_scores = cross_val_score(ridge, X, y, cv=10)

    # Append the mean of ridge_cv_scores to ridge_scores
    ridge_scores.append(np.mean(ridge_cv_scores))

    # Append the std of ridge_cv_scores to ridge_scores_std
    ridge_scores_std.append(np.std(ridge_cv_scores))

# Display the plot
def display_plot(cv_scores, cv_scores_std):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alpha_space, cv_scores)

    std_error = cv_scores_std / np.sqrt(10)

    ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)
    ax.set_ylabel('CV Score +/- Std Error')
    ax.set_xlabel('Alpha')
    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
    ax.set_xlim([alpha_space[0], alpha_space[-1]])
    ax.set_xscale('log')
    plt.show()
display_plot(ridge_scores, ridge_scores_std)
# </editor-fold>

""" 
    Building a ElasticNet: a*L1+b*L2 combination between lasso and ridge
"""
#<editor-fold> ElasticNet
from sklearn.linear_model import ElasticNet
# Create the hyperparameter grid
l1_space = np.linspace(0, 1, 30)
param_grid = {"l1_ratio": l1_space}
elastic_net = ElasticNet()
gm_cv = GridSearchCV(elastic_net, param_grid, cv=5)
gm_cv.fit(X_train,y_train)
y_pred = gm_cv.predict(X_test)
r2 = gm_cv.score(X_test,y_test)
mse = mean_squared_error(y_pred, y_test)
print("Tuned ElasticNet l1 ratio: {}".format(gm_cv.best_params_))
print("Tuned ElasticNet R squared: {}".format(r2))
print("Tuned ElasticNet MSE: {}".format(mse))


# </editor-fold>


# </editor-fold>


# </editor-fold>

"""
validation
"""
#validation <editor-fold>
"""
    cross_val_score
"""
from sklearn.model_selection import cross_val_score
#just to do a score and fit 5 times
cvscores_5 = cross_val_score(reg,X,y,cv=5)
"""
    confusion_matrix, classification_report
"""
from sklearn.metrics import confusion_matrix, classification_report
# Generate the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

"""
    Plotting an ROC curve
"""
# <editor-fold>

# Import necessary modules
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test,y_pred_prob)

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))
# Compute cross-validated AUC scores: cv_auc
cv_auc = cross_val_score(logreg,X,y,cv=5,scoring='roc_auc')

# Print list of AUC scores
print("AUC scores computed using 5-fold cross-validation: {}".format(cv_auc))

# </editor-fold>

"""
    mean_squared_error
"""
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_pred, y_test)

# </editor-fold>


"""
Hyperparameter tuning
"""
#Hyperparameter tuning  <editor-fold>
"""
    Hyperparameter tuning with GridSearchCV
"""
# <editor-fold>

from sklearn.model_selection import GridSearchCV
# Setup the hyperparameter grid
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space}

# Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(logreg,param_grid,cv=5)

# Fit it to the data
logreg_cv.fit(X,y)
# Print the tuned parameters and score
print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_))
print("Best score is {}".format(logreg_cv.best_score_))
# </editor-fold>

"""
    Hyperparameter tuning with RandomizedSearchCV
"""
# <editor-fold>
from sklearn.model_selection import RandomizedSearchCV
# Setup the parameters and distributions to sample from: param_dist
param_dist = {"max_depth": [3, None],
              "max_features": randint(1, 9),
              "min_samples_leaf": randint(1, 9),
              "criterion": ["gini", "entropy"]}
tree_cv = RandomizedSearchCV(tree,param_dist, cv=5)
tree_cv.fit(X,y)
# Print the tuned parameters and score
print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))

# </editor-fold>



# </editor-fold>

"""
preproccecing
"""
#<editor-fold>

"""
    Creating dummy variables
"""
#<editor-fold>Creating dummy variables
# Create dummy variables: df_region
df_region = pd.get_dummies(X)

# Print the columns of df_region
print(df_region.columns)

# Create dummy variables with drop_first=True: df_region better!
df_region = pd.get_dummies(X,drop_first=True)

# Print the new columns of df_region
print(df_region.columns)


#</editor-fold>

"""
    missing data
"""
#<editor-fold>missing data
# Import the Imputer module
from sklearn.impute import SimpleImputer as Imputer
from sklearn.svm import SVC
# Setup the Imputation transformer: imp
imp = Imputer(missing_values="NaN", strategy="most_frequent")#for categorial
imp2 = Imputer(missing_values="NaN", strategy="mean")#for non categorial
# Instantiate the SVC classifier: clf
clf = SVC()
# Setup the pipeline with the required steps: steps
steps = [('imputation', imp),
        ('SVM', clf)]


"""
        using pipline
"""
#<editor-fold>using pipline
from sklearn.pipeline import Pipeline
# Create the pipeline: pipeline
pipeline = Pipeline(steps)
# Fit the pipeline to the train set
pipeline.fit(X_train,y_train)
# Predict the labels of the test set
y_pred = pipeline.predict(X_test)
# Compute metrics
print(classification_report(y_test, y_pred))
#</editor-fold>

#</editor-fold>

"""
    Centering and scaling your data
"""
#<editor-fold>
from sklearn.preprocessing import scale
# Scale the features: X_scaled
X_scaled = scale(X)
from sklearn.preprocessing import StandardScaler

"""
        using pipeline
"""
#<editor-fold>
# Setup the pipeline steps: steps
steps = [('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())]
scaler = StandardScaler()
data = pd.DataFrame(scaler.fit_transform(X),columns=X.columns)

# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Fit the pipeline to the training set: knn_scaled
scaled = pipeline.fit(X_train,y_train)
# Instantiate and fit a k-NN classifier to the unscaled data
unscaled = KNeighborsClassifier().fit(X_train, y_train)
# Compute and print metrics
print('Accuracy with Scaling: {}'.format(scaled.score(X_test,y_test)))
print('Accuracy without Scaling: {}'.format(unscaled.score(X_test,y_test)))

#</editor-fold>

#</editor-fold>


#</editor-fold>


"""
Bringing it all together I: Pipeline for classification
"""
# Setup the pipeline
steps = [('scaler', StandardScaler()),
         ('SVM', SVC())]

pipeline = Pipeline(steps)

# Specify the hyperparameter space
parameters = {'SVM__C':[1, 10, 100],
              'SVM__gamma':[0.1, 0.01]}

# Create train and test sets
X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.2,random_state=21)

# Instantiate the GridSearchCV object: cv
cv = GridSearchCV(pipeline,parameters,cv=5)

# Fit to the training set
cv.fit(X_train,y_train)

# Predict the labels of the test set: y_pred
y_pred = cv.predict(X_test)

# Compute and print metrics
print("Accuracy: {}".format(cv.score(X_test, y_test)))
print(classification_report(y_test, y_pred))
print("Tuned Model Parameters: {}".format(cv.best_params_))

"""
Bringing it all together I: Pipeline for regretion
"""
# Setup the pipeline steps: steps
steps = [('imputation', Imputer(missing_values='NaN', strategy='mean', axis=0)),
         ("scaler",StandardScaler() ),
         ('elasticnet' ,ElasticNet() )]

# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Specify the hyperparameter space
parameters = {'elasticnet__l1_ratio':np.linspace(0,1,30)}

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=42)

# Create the GridSearchCV object: gm_cv
gm_cv = GridSearchCV(pipeline,parameters)

# Fit to the training set
gm_cv.fit(X_train,y_train)

# Compute and print the metrics
r2 = gm_cv.score(X_test, y_test)
print("Tuned ElasticNet Alpha: {}".format(gm_cv.best_params_))
print("Tuned ElasticNet R squared: {}".format(r2))


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


"""
change TEXT TO tfidf
"""
# Create the vectorizer method
tfidf_vec = TfidfVectorizer()
text = np.array(["it was a really hard day","me is a nice person"])
# Transform the text into tf-idf vectors when it is text like description
text_tfidf = tfidf_vec.fit_transform(text)
def return_weights(text_tfidf,tfidf_vec,vector_index):
    vocab = {v:k for k,v in tfidf_vec.vocabulary_.items()}
    zipped_row = dict(zip(text_tfidf[vector_index].indices,text_tfidf[vector_index].data))
    return {vocab[i]:zipped_row[i] for i in tfidf_vec[vector_index].indices}


# Add in the rest of the parameters
def return_weights_topn(tfidf_vec, text_tfidf, vector_index, top_n):
    vocab = {v:k for k,v in tfidf_vec.vocabulary_.items()}
    original_vocab = tfidf_vec.vocabulary_
    zipped = dict(zip(text_tfidf[vector_index].indices, text_tfidf[vector_index].data))

    # Let's transform that zipped dict into a series
    zipped_series = pd.Series({vocab[i]: zipped[i] for i in text_tfidf[vector_index].indices})

    # Let's sort the series to pull out the top n weighted words
    zipped_index = zipped_series.sort_values(ascending=False)[:top_n].index
    return [original_vocab[i] for i in zipped_index]


def words_to_filter(tfidf_vec, text_tfidf, top_n):
    vocab = {v:k for k,v in tfidf_vec.vocabulary_.items()}
    original_vocab = tfidf_vec.vocabulary_
    filter_list = []
    for i in range(0, text_tfidf.shape[0]):
        # Here we'll call the function from the previous exercise, and extend the list we're creating
        filtered = return_weights(vocab, original_vocab, text_tfidf, i, top_n)
        filter_list.extend(filtered)
    # Return the list in a set, so we don't get duplicate word indices
    return set(filter_list)

# By converting filtered_words back to a list, we can use it to filter the columns in the text vector
filtered_text = text_tfidf[:, list(words_to_filter(tfidf_vec, text_tfidf, 3))]
# Split the dataset according to the class distribution of category_desc, using the filtered_text vector
train_X, test_X, train_y, test_y = train_test_split(filtered_text.toarray(), y, stratify=y)
from sklearn.naive_bayes import GaussianNB as nb
# Fit the model to the training data
nb.fit(train_X,train_y)

# Print out the model's accuracy
print(nb.score(test_X,test_y))
"""
Take a minute to look at the correlations. 
Identify a column where the correlation value is greater than 0.75 
at least twice and store it in the to_drop variable.
"""

"""
PCA
"""
from sklearn.decomposition import PCA
pca=PCA()
data_pca=pca.fit_transform(X,y)
print(pca.explained_variance_ratio_)








"""
tensor flow
sigmoid for output of binary
softmax for output lair > 2 classes
relo for all hidden lair

SGD(stochasic gardient decent) optimizer training rate from 0.01-0.5. simple and easy.
RMS(root mean squared) for each featcher differnt learning rate.good for alot of dimentions.allow build momentum. and decay(low value mean prevent accumalating over a long time)
big momentum = less local minimum[0-1]
Adam(adaotivee moment), beta1 small will make to decay faster. better in defult.

random.normal just normal distribution
random.truncated_normal just normal distribution with less קיצוני

"""

opt = keras.optimizers.SGD(learning_rate=0.01)
opt.minimize(lambda: loss_function(x_1), var_list=[x_1])

weight = tf.Varible(tf.random.normal([500,500]))

num_of_output=5
dense1 = tf.keras.layers.Dense(num_of_output,activation ="relu",kernel_initializer="zeros")(input_feather)
dropout = tf.keras.layers.Dropout(0.25)(dense1)#dropout 25% of connection to avoid overfit
dense2 =  tf.keras.layers.Dense(num_of_output,activation ="relu",kernel_initializer="zeros")(dropout)


"""
all together
"""
# Define the layer 1 weights
w1 = Variable(random.normal([23, 7]))
# Initialize the layer 1 bias
b1 = Variable(ones([7]))
# Define the layer 2 weights
w2 = Variable(random.normal([7, 1]))
# Define the layer 2 bias
b2 = Variable(0.0)

# Define the model
def model(w1, b1, w2, b2, features = borrower_features):
	# Apply relu activation functions to layer 1
	layer1 = keras.activations.relu(matmul(features, w1) + b1)
    # Apply dropout rate of 0.25
	dropout = keras.layers.Dropout(0.25)(layer1)
	return keras.activations.sigmoid(matmul(dropout, w2) + b2)

# Define the loss function
def loss_function(w1, b1, w2, b2, features = borrower_features, targets = default):
	predictions = model(w1, b1, w2, b2)
	# Pass targets and predictions to the cross entropy loss
	return keras.losses.binary_crossentropy(targets, predictions)
# Train the model
for j in range(100):
    # Complete the optimizer
	opt.minimize(lambda: loss_function(w1, b1, w2, b2),
                 var_list=[w1, b1, w2, b2])

# Make predictions with model using test features
model_predictions = model(w1, b1, w2, b2, test_features)

# Construct the confusion matrix
confusion_matrix(test_targets, model_predictions)



"""
high level API of tensirFlow
"""
"""
    The sequential model in Keras wich mean add leayer by leayer in herricahl order
"""
# Define a Keras sequential model
model = keras.Sequential()

# Define the first dense layer
model.add(keras.layers.Dense(16, activation='relu', input_shape=(784,)))

# Define the second dense layer
model.add(keras.layers.Dense(8, activation='relu'))

# Define the output layer
model.add(keras.layers.Dense(4, activation='softmax'))

# Print the model architecture
print(model.summary())

# Compile the model
model.compile('adam', loss='categorical_crossentropy')

"""
    Defining a multiple input model wich you want to make tow inputs seperatly and than merge tham in the output
"""
m1_inputs = pd.DataFrame(["a"])
m2_inputs = pd.DataFrame(["a"])
# For model 1, pass the input layer to layer 1 and layer 1 to layer 2
m1_layer1 = keras.layers.Dense(12, activation='sigmoid')(m1_inputs)
m1_layer2 = keras.layers.Dense(4, activation='softmax')(m1_layer1)

# For model 2, pass the input layer to layer 1 and layer 1 to layer 2
m2_layer1 = keras.layers.Dense(12, activation='relu')(m2_inputs)
m2_layer2 = keras.layers.Dense(4, activation='softmax')(m2_layer1)

# Merge model outputs and define a functional model
merged = keras.layers.add([m1_layer2, m2_layer2])
model = keras.Model(inputs=[m1_inputs, m2_inputs], outputs=merged)

# Print a model summary
print(model.summary())

"""
    train and validate
"""

# Define a sequential model
model = keras.Sequential()

# Define a hidden layer
model.add(keras.layers.Dense(16, activation='relu', input_shape=(784,)))

# Define the output layer
model.add(keras.layers.Dense(4, activation='softmax', input_shape=(784,)))


# Compile the model
model.compile('SGD', loss='categorical_crossentropy')

sign_language_features = pd.DataFrame(randint(46,100),shape=(1000,784))#784 featchers and 1000 rows
sign_language_labels = pd.DataFrame(randint(46,100),shape=(1000, 4))#1000 rows and 4 classificaton task

# Complete the fitting operation
model.fit(sign_language_features, sign_language_labels, epochs=5)


# Set the optimizer, loss function, and metrics(for the validation of splits)
model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Add the number of epochs and the validation split
model.fit(sign_language_features, sign_language_labels, epochs=10, validation_split=0.1)

# Evaluate the  model using the train data
small_train = model.evaluate(train_features, train_labels)

# Evaluate the  model using the test data
small_test = model.evaluate(test_features, test_labels)


"""
very high level API name esitmator of tensirFlow
"""

# Define feature columns for bedrooms and bathrooms
housing = pd.DataFrame()
bedrooms = feature_column.numeric_column("bedrooms")
bathrooms = feature_column.numeric_column("bathrooms")

# Define the list of feature columns
feature_list = [bedrooms, bathrooms]

def input_fn():
	# Define the labels
	labels = np.array(housing['price'])
	# Define the features
	features = {'bedrooms':np.array(housing['bedrooms']),
                'bathrooms':np.array(housing['bathrooms'])}
	return features, labels


# Define the model and set the number of steps
model = estimator.DNNRegressor(feature_columns=feature_list, hidden_units=[2,2])
model.train(input_fn, steps=1)
# Define the model and set the number of steps
model = estimator.LinearRegressor(feature_columns=feature_list)
model.train(input_fn, steps=2)
