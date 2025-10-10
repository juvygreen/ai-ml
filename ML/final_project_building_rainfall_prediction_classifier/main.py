

# About The Dataset
# The original source of the data is Australian Government's Bureau of Meteorology and the latest data can be gathered from
# http://www.bom.gov.au/climate/dwo/.
# The dataset you'll use in this project was downloaded from Kaggle at
# https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package/
# Column definitions were gathered from http://www.bom.gov.au/climate/dwo/IDCJDW0000.shtml


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

# Load the data
# Execute the following cells to load the dataset as a pandas dataframe.
url="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/_0eYOqji3unP1tDNKWZMjg/weatherAUS-2.csv"
df = pd.read_csv(url)
df.head()

df.count()

# Drop all rows with missing values
# To try to keep things simple we'll drop rows with missing values and see what's left
df = df.dropna()
df.info()


# Since we still have 56k observations left after dropping missing values, we may not need to impute any missing values.
# Let's see how we do.
df.columns
# Output:
# Index(['Date', 'Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation',
#       'Sunshine', 'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm',
#       'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
#       'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am',
#       'Temp3pm', 'RainToday', 'RainTomorrow'],
#      dtype='object')

# Consider features that rely on the entire duration of today for their evaluation.
# If we adjust our approach and aim to predict today’s rainfall using historical weather data up to and including yesterday,
# then we can legitimately utilize all of the available features. This shift would be particularly useful for practical applications,
# such as deciding whether you will bike to work today.
# With this new target, we should update the names of the rain columns accordingly to avoid confusion.
df = df.rename(columns={'RainToday': 'RainYesterday',
                        'RainTomorrow': 'RainToday'
                        })

# Because there might still be some slight variations in the weather patterns we'll keep Location as a categorical variable.
df = df[df.Location.isin(['Melbourne','MelbourneAirport','Watsonia',])]
df. info()


# Extracting a seasonality feature
# Now consider the Date column. We expect the weather patterns to be seasonal, having different predictablitiy
# levels in winter and summer for example.
# There may be some variation with Year as well, but we'll leave that out for now. Let's engineer a Season
# feature from Date and drop Date afterward, since it is most likely less informative than season.
# An easy way to do this is to define a function that assigns seasons to given months, then use that
# function to transform the Date column.
# Create a function to map dates to seasons
def date_to_season(date):
    month = date.month
    if (month == 12) or (month == 1) or (month == 2):
        return 'Summer'
    elif (month == 3) or (month == 4) or (month == 5):
        return 'Autumn'
    elif (month == 6) or (month == 7) or (month == 8):
        return 'Winter'
    elif (month == 9) or (month == 10) or (month == 11):
        return 'Spring'

# Map the dates to seasons and drop the Date column
import pandas as pd

# Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Apply the function to the 'Date' column
df['Season'] = df['Date'].apply(date_to_season)

# Drop the 'Date' column
df = df.drop(columns=['Date'])

df.head()


# Define the feature and target dataframes
# Define the feature dataframe (X) and target dataframe (y)
X = df.drop(columns='RainToday', axis=1)
y = df['RainToday']

# How balanced are the classes?
# Display the counts of each class.
y.value_counts()
# Output:
#RainToday
#No     5766
#Yes    1791
#Name: count, dtype: int64

y.value_counts()

# Split data into training and test sets, ensuring target stratification
X_train, X_test, y_train, y_test = train_test_split(..., ..., test_size=0.2, stratify=..., random_state=42)

# Split the data into training and test sets, ensuring target stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Define preprocessing transformers for numerical and categorical features
# Automatically detect numerical and categorical columns and assign them to separate numeric and categorical features¶
numeric_features = X_train.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()


# Define separate transformers for both feature types and combine them into a single preprocessing transformer
# Scale the numeric features
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

# One-hot encode the categoricals
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Combine the transformers into a single preprocessing column transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Define the transformers
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Combine the transformers into a single preprocessing column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),  # Apply StandardScaler to numeric features
        ('cat', categorical_transformer, categorical_features)  # Apply OneHotEncoder to categorical features
    ]
)

# Create a pipeline by combining the preprocessing with a Random Forest classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# Create a pipeline by combining the preprocessing with a Random Forest classifier
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),  # Apply the preprocessor created in Exercise 7
    ('classifier', RandomForestClassifier(random_state=42))  # Use RandomForestClassifier as the model
])

# Define a parameter grid to use in a cross validation grid search model optimizer
param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5]
}

# Pipeline usage in crossvalidation
# Recall that the pipeline is repeatedly used within the crossvalidation by fitting on each internal training fold and
# predicting on its corresponding validation fold
# Perform grid search cross-validation and fit the best model to the training data
# Select a cross-validation method, ensuring target stratification during validation
cv = StratifiedKFold(n_splits=5, shuffle=True)

# Instantiate and fit GridSearchCV to the pipeline
from sklearn.model_selection import GridSearchCV

# Define the parameter grid for RandomForestClassifier
param_grid = {
    'classifier__n_estimators': [100, 200],  # Number of trees in the forest
    'classifier__max_depth': [None, 10, 20],  # Maximum depth of the tree
    'classifier__min_samples_split': [2, 5],  # Minimum samples required to split a node
    'classifier__min_samples_leaf': [1, 2]  # Minimum samples required at a leaf node
}

# Instantiate and fit GridSearchCV to the pipeline
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,  # 5-fold cross-validation
    scoring='accuracy',  # We are optimizing for accuracy
    verbose=2  # Print detailed logs of the grid search process
)

# Fit the grid search to the training data
grid_search.fit(X_train, y_train)


# Print the best parameters and best crossvalidation score
print("\nBest parameters found: ", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

# Display your model's estimated score
test_score = grid_search.score(X_test, y_test)
print("Test set score: {:.2f}".format(test_score))

# Get the model predictions from the grid search estimator on the unseen data
y_pred = grid_search.predict(X_test)

# Print the classification report
from sklearn.metrics import classification_report

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot the confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Compute and display the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap='Blues')  # Optional: use a blue color map for aesthetics
plt.title('Confusion Matrix')
plt.show()


# Extract the feature importances
# Extract feature importances from the trained Random Forest classifier
feature_importances = grid_search.best_estimator_['classifier'].feature_importances_

# Now let's extract the feature importances and plot them as a bar graph.
# Combine numeric and categorical feature names
feature_names = numeric_features + list(grid_search.best_estimator_['preprocessor']
                                        .named_transformers_['cat']
                                       # .named_steps['onehot']
                                        .get_feature_names_out(categorical_features))

feature_importances = grid_search.best_estimator_['classifier'].feature_importances_

importance_df = pd.DataFrame({'Feature': feature_names,
                              'Importance': feature_importances
                             }).sort_values(by='Importance', ascending=False)

N = 20  # Change this number to display more or fewer features
top_features = importance_df.head(N)

# Plotting
plt.figure(figsize=(10, 6))
plt.barh(top_features['Feature'], top_features['Importance'], color='skyblue')
plt.gca().invert_yaxis()  # Invert y-axis to show the most important feature on top
plt.title(f'Top {N} Most Important Features in predicting whether it will rain today')
plt.xlabel('Importance Score')
plt.show()


# ============
# Try another model - LogisticRegression

# Update the pipeline and the parameter grid
# Update the pipeline and the parameter grid and train a Logistic Regression model and compare the performance of the two models.
# You'll need to replace the clasifier with LogisticRegression. We have supplied the parameter grid for you.
from sklearn.linear_model import LogisticRegression

# 1. Replace RandomForestClassifier with LogisticRegression in the pipeline
pipeline.set_params(classifier=LogisticRegression(random_state=42))

# 2. Update the GridSearchCV estimator to use the new pipeline
grid_search.estimator = pipeline

# 3. Define the new parameter grid for Logistic Regression
param_grid = {
    'classifier__solver': ['liblinear'],
    'classifier__penalty': ['l1', 'l2'],
    'classifier__class_weight': [None, 'balanced']
}

# 4. Update the param_grid in the grid_search
grid_search.param_grid = param_grid

# 5. Fit the updated pipeline with LogisticRegression
grid_search.fit(X_train, y_train)

# 6. Make predictions on the test set
y_pred = grid_search.predict(X_test)

# Compare the results to your previous model.
# Display the clasification report and the confusion matrix for the new model and compare your results with the previous model.
print(classification_report(y_test, y_pred))

# Generate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')

# Set the title and labels
plt.title('Titanic Classification Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Show the plot
plt.tight_layout()
plt.show()
