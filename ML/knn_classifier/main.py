import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
%matplotlib inline

# Let's read the data using pandas library and print the first five rows.
df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv')
df.head()

# Let us first look at the class-wise distribution of the data set.
df['custcat'].value_counts()

# We can also visualize the correlation map of the data set to determine how the different features are related to each other.
correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)

# will give us a list of features sorted in the descending order of their absolute correlation values with respect to the target field.
correlation_values = abs(df.corr()['custcat'].drop('custcat')).sort_values(ascending=False)
correlation_values

# Separate the input and target features
# Now, we can separate the data into the input data set and the target data set.
X = df.drop('custcat',axis=1)
y = df['custcat']

# This helps KNN make better decisions based on the actual relationships between features, not just on the magnitude of their values.
X_norm = StandardScaler().fit_transform(X)

# should separate the training and the testing data. You can retain 20% of the data for testing purposes
# and use the rest for training. Assigning a random state ensures reproducibility of the results across multiple executions.
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=4)

# Training
# Initially, you may start by using a small value as the value of k, say k = 4.
k = 3
#Train Model and Predict
knn_classifier = KNeighborsClassifier(n_neighbors=k)
knn_model = knn_classifier.fit(X_train,y_train)

# Predicting
# Once the model is trained, we can now use this model to generate predictions for the test set.
yhat = knn_model.predict(X_test)

# Accuracy evaluation
# In multilabel classification, accuracy classification score is a function that computes subset accuracy.
# This function is equal to the jaccard_score function. Essentially, it calculates how closely the actual labels and
# predicted labels are matched in the test set.
print("Test set Accuracy: ", accuracy_score(y_test, yhat))


# build the model again, but this time with k=6?
k = 6
knn_model_6 = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
yhat6 = knn_model_6.predict(X_test)
print("Test set Accuracy: ", accuracy_score(y_test, yhat6))

# Check the performance of the model for 10 values of k, ranging from 1-9.
# You can evaluate the accuracy along with the standard deviation of the accuracy as well to
# get a holistic picture of the model performance.
Ks = 10
acc = np.zeros((Ks))
std_acc = np.zeros((Ks))
for n in range(1,Ks+1):
    #Train Model and Predict
    knn_model_n = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat = knn_model_n.predict(X_test)
    acc[n-1] = accuracy_score(y_test, yhat)
    std_acc[n-1] = np.std(yhat==y_test)/np.sqrt(yhat.shape[0])


# plot the model accuracy and the standard deviation to identify the model with the most suited value of k.
plt.plot(range(1,Ks+1),acc,'g')
plt.fill_between(range(1,Ks+1),acc - 1 * std_acc,acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy value', 'Standard Deviation'))
plt.ylabel('Model Accuracy')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()

print( "The best accuracy was with", acc.max(), "with k =", acc.argmax()+1)


# Plot the variation of the accuracy score for the training set for 100 value of Ks.
Ks =100
acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
for n in range(1,Ks):
    #Train Model and Predict
    knn_model_n = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat = knn_model_n.predict(X_train)
    acc[n-1] = accuracy_score(y_train, yhat)
    std_acc[n-1] = np.std(yhat==y_train)/np.sqrt(yhat.shape[0])

plt.plot(range(1,Ks),acc,'g')
plt.fill_between(range(1,Ks),acc - 1 * std_acc, acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy value', 'Standard Deviation'))
plt.ylabel('Model Accuracy')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()


