import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import seaborn as sns
from sklearn.metrics import confusion_matrix


# Train a model using a pipeline We'll start with an example of building a pipeline, fitting it to the Iris data,
# and evaluating its accuracy.
# Load the Iris data set
data = load_iris()
X, y = data.data, data.target
labels = data.target_names

# Instantiate a pipeline consisting of StandardScaler, PCA, and KNeighborsClassifier
pipeline = Pipeline([
    ('scaler', StandardScaler()),       # Step 1: Standardize features
    ('pca', PCA(n_components=2),),       # Step 2: Reduce dimensions to 2 using PCA
    ('knn', KNeighborsClassifier(n_neighbors=5,))  # Step 3: K-Nearest Neighbors classifier
])

# Split the data into training and test sets
# Be sure to stratify the target.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# Fit the pipeline on the training set
# The pipeline consists of a sequence of three estimators, and should be viewed as the machine learning model we are training and testing.
# Let's go ahead and fit the model to the training data and evaluate its accuracy.
pipeline.fit(X_train, y_train)

# Measure the pipeline accuracy on the test data
test_score = pipeline.score(X_test, y_test)
print(f"{test_score:.3f}")

# Get the model predictions
y_pred = pipeline.predict(X_test)

# Generate the confusion matrix for the KNN model and plot it
# generate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Create a plot for the confusion matrix
plt.figure()
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d',
            xticklabels=labels, yticklabels=labels)

# Set the title and labels
plt.title('Classification Pipeline Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Show the plot
plt.tight_layout()
plt.show()


# Instantiate the pipeline
# We'll preprocess the data by scaling it and transforming it onto a to-be-determined number of principle components,
# follow that up with a KNN model classifier, and combine these estimators into a pipeline. We'll then optimize the
# pipeline using crossvalidation over a hyperparameter grid search. This will allow us find the best model for the
# set of trial hyperparamters.
# make a pipeline without specifying any parameters yet
pipeline = Pipeline(
                    [('scaler', StandardScaler()),
                     ('pca', PCA()),
                     ('knn', KNeighborsClassifier())
                    ]
                   )

# Define a model parameter grid to search over
# Hyperparameter search grid for numbers of PCA components and KNN neighbors
param_grid = {'pca__n_components': [2, 3],
              'knn__n_neighbors': [3, 5, 7]
             }

# Choose a cross validation method
# To ensure the target is stratified, we can use scikit-learn's StratifiedKFold cross-validation class.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Determine the best parameters
# Pass your pipeline, param_grid, and the StratifiedKFold cross validation method to GridSearchCV
best_model = GridSearchCV(estimator=pipeline,
                          param_grid=param_grid,
                          cv=cv,
                          scoring='accuracy',
                          verbose=2
                         )

# Fit the best GridSearchCV model to the training data
best_model.fit(X_train, y_train)


# Evaluate the accuracy of the best model on the test set
test_score = best_model.score(X_test, y_test)
print(f"{test_score:.3f}")
# Output: 0.933

# We've made a great accuracy improvement from 90% to 93%.
# Display the best parameters
best_model.best_params_
# Output: {'knn__n_neighbors': 3, 'pca__n_components': 3}

# Plot the confusion matrix for the predictions on the test set
y_pred = best_model.predict(X_test)

# Generate the confusion matrix for KNN
conf_matrix = confusion_matrix(y_test, y_pred)

# Create a single plot for the confusion matrix
plt.figure()
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d',
            xticklabels=labels, yticklabels=labels)

# Set the title and labels
plt.title('KNN Classification Testing Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Show the plot
plt.tight_layout()
plt.show()
