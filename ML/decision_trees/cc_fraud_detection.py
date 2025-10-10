# Import the libraries we need to use in this lab
from __future__ import print_function
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.svm import LinearSVC

import warnings
warnings.filterwarnings('ignore')

# download the dataset
url= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/creditcard.csv"

# read the input data
raw_data=pd.read_csv(url)
raw_data

# get the set of distinct classes
labels = raw_data.Class.unique()

# get the count of each class
sizes = raw_data.Class.value_counts().values

# plot the class value counts
fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='%1.3f%%')
ax.set_title('Target Variable Value Counts')
plt.show()

# It is also prudent to understand which features affect the model in what way.
# We can visualize the effect of the different features on the model using the code below.
correlation_values = raw_data.corr()['Class'].drop('Class')
correlation_values.plot(kind='barh', figsize=(10, 6))

# standardize features by removing the mean and scaling to unit variance
raw_data.iloc[:, 1:30] = StandardScaler().fit_transform(raw_data.iloc[:, 1:30])
data_matrix = raw_data.values

# X: feature matrix (for this analysis, we exclude the Time variable from the dataset)
X = data_matrix[:, 1:30]

# y: labels vector
y = data_matrix[:, 30]

# data normalization
X = normalize(X, norm="l1")
# Dataset Train/Test Split
# Now that the dataset is ready for building the classification models, you need to first divide the pre-processed
# dataset into a subset to be used for training the model (the train set)
# and a subset to be used for evaluating the quality of the model (the test set).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build a Decision Tree Classifier model with Scikit-Learn
# Compute the sample weights to be used as input to the train routine so that it takes into
# account the class imbalance present in this dataset.
w_train = compute_sample_weight('balanced', y_train)

# for reproducible output across multiple function calls, set random_state to a given integer value
dt = DecisionTreeClassifier(max_depth=4, random_state=35)

dt.fit(X_train, y_train, sample_weight=w_train)


# Unlike Decision Trees, we do not need to initiate a separate sample_weight for SVMs.
# We can simply pass a parameter in the scikit-learn function.
# for reproducible output across multiple function calls, set random_state to a given integer value
svm = LinearSVC(class_weight='balanced', random_state=31, loss="hinge", fit_intercept=False)
svm.fit(X_train, y_train)

# Run the following cell to compute the probabilities of the test samples belonging to the class of fraudulent transactions.
y_pred_dt = dt.predict_proba(X_test)[:,1

# The AUC-ROC score evaluates your model's ability to distinguish positive and negative
# classes considering all possible probability thresholds. T
# he higher its value, the better the model is considered for separating the two classes of values.
roc_auc_dt = roc_auc_score(y_test, y_pred_dt)
print('Decision Tree ROC-AUC score : {0:.3f}'.format(roc_auc_dt))

# compute the probabilities of the test samples belonging to the class of fraudulent transactions.
y_pred_svm = svm.decision_function(X_test)
# evaluate the accuracy of SVM on the test set in terms of the ROC-AUC score.
roc_auc_svm = roc_auc_score(y_test, y_pred_svm)
print("SVM ROC-AUC score: {0:.3f}".format(roc_auc_svm))

# Currently, we have used all 30 features of the dataset for training the models.
# Use the corr() function to find the top 6 features of the dataset to train the models on.
correlation_values = abs(raw_data.corr()['Class']).drop('Class')
correlation_values = correlation_values.sort_values(ascending=False)[:6]
correlation_values

# Using only these 6 features, modify the input variable for training.
# X = data_matrix[:,[3,10,12,14,16,17]]
# Replace the statement defining the variable `X` with the following and run the cell again.
# standardize features by removing the mean and scaling to unit variance
raw_data.iloc[:, 1:30] = StandardScaler().fit_transform(raw_data.iloc[:, 1:30])
data_matrix = raw_data.values

# X: feature matrix (for this analysis, we exclude the Time variable from the dataset)
#X = data_matrix[:, 1:30]
X = data_matrix[:,[3,10,12,14,16,17]]

# y: labels vector
y = data_matrix[:, 30]

# data normalization
X = normalize(X, norm="l1")

# Execute the Decision Tree model for this modified input variable. How does the value of ROC-AUC metric change?
roc_auc_dt = roc_auc_score(y_test, y_pred_dt)
print('Decision Tree ROC-AUC score : {0:.3f}'.format(roc_auc_dt))
