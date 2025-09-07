import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score

# Define a function to display evaluation metrics
# We'll be evaluating several models, so its a good idea to make an evaluation function that includes the common metrics and explaned variances etc.
def regression_results(y_true, y_pred, regr_type):
    # Regression metrics
    ev = explained_variance_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print('Evaluation metrics for ' + regr_type + ' Linear Regression')
    print('explained_variance: ', round(ev, 4))
    print('r2: ', round(r2, 4))
    print('MAE: ', round(mae, 4))
    print('MSE: ', round(mse, 4))
    print('RMSE: ', round(np.sqrt(mse), 4))
    print()

# Generate a simple dataset with one feature
# We'll create a simple data set with a linear relationship between the target and a single feature.
# We'll add some noise to the target to simulate real data. We'll also create a data set from this one that has outliers added.
# Then you'll compare the three linear regression model performances on the two datasets, with and without sigiifcant outliers added.
# Generate synthetic data
noise=1
np.random.seed(42)
X = 2 * np.random.rand(1000, 1)
y = 4 + 3 * X + noise*np.random.randn(1000, 1)  # Linear relationship with some noise
y_ideal =  4 + 3 * X
# Specify the portion of the dataset to add outliers (e.g., the last 20%)
y_outlier = pd.Series(y.reshape(-1).copy())

# Identify indices where the feature variable X is greater than a certain threshold
threshold = 1.5  # Example threshold to add outliers for larger feature values
outlier_indices = np.where(X.flatten() > threshold)[0]

# Add outliers at random locations within the specified portion
num_outliers = 5  # Number of outliers to add
selected_indices = np.random.choice(outlier_indices, num_outliers, replace=False)

# Modify the target values at these indices to create outliers (add significant noise)
y_outlier[selected_indices] += np.random.uniform(50, 100, num_outliers)

# Plot the data with outliers and the ideal fit line
plt.figure(figsize=(12, 6))

# Scatter plot of the original data with outliers
plt.scatter(X, y_outlier, alpha=0.4,ec='k', label='Original Data with Outliers')
plt.plot(X, y_ideal,  linewidth=3, color='g',label='Ideal, noise free data')

plt.xlabel('Feature (X)')
plt.ylabel('Target (y)')
plt.title('')
plt.legend()
plt.show()


# Plot the data without the outliers and the ideal fit line
plt.figure(figsize=(12, 6))

# Scatter plot of the original data with outliers
plt.scatter(X, y, alpha=0.4,ec='k', label='Original Data without Outliers')
plt.plot(X, y_ideal,  linewidth=4, color='g',label='Ideal, noise free data')

plt.xlabel('Feature (X)')
plt.ylabel('Target (y)')
plt.title('')
plt.legend()
plt.show()

# Fit Ordinary, Ridge, and Lasso regression models and use them to make predicitions on the original, outlier-free data
# Fit a simple linear regression model
lin_reg = LinearRegression()
lin_reg.fit(X, y_outlier)
y_outlier_pred_lin = lin_reg.predict(X)

# Fit a ridge regression model (regularization to control large coefficients)
ridge_reg = Ridge(alpha=1)
ridge_reg.fit(X, y_outlier)
y_outlier_pred_ridge = ridge_reg.predict(X)

# Fit a lasso regression model (regularization to control large coefficients)
lasso_reg = Lasso(alpha=.2)
lasso_reg.fit(X, y_outlier)
y_outlier_pred_lasso = lasso_reg.predict(X)

# Print the regression results
regression_results(y, y_outlier_pred_lin, 'Ordinary')
regression_results(y, y_outlier_pred_ridge, 'Ridge')
regression_results(y, y_outlier_pred_lasso, 'Lasso')


# Plot the data and the predictions for comparison
# Let's see how well the predictions match expected ideal values.

plt.figure(figsize=(12, 6))

# Scatter plot of the original data with outliers
plt.scatter(X, y, alpha=0.4,ec='k', label='Original Data')

# Plot the ideal regression line (noise free data)
plt.plot(X, y_ideal,  linewidth=2, color='k',label='Ideal, noise free data')

# Plot predictions from the simple linear regression model
plt.plot(X, y_outlier_pred_lin,  linewidth=5, label='Linear Regression')

# Plot predictions from the ridge regression model
plt.plot(X, y_outlier_pred_ridge, linestyle='--', linewidth=2, label='Ridge Regression')

# Plot predictions from the lasso regression model
plt.plot(X, y_outlier_pred_lasso,  linewidth=2, label='Lasso Regression')

plt.xlabel('Feature (X)')
plt.ylabel('Target (y)')
plt.title('Comparison of Predictions with Outliers')
plt.legend()
plt.show()


# Build the models and the prediction plots from the same data, excluding the outliers
# Fit a simple linear regression model
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred_lin = lin_reg.predict(X)

# Fit a ridge regression model (regularization to control large coefficients)
ridge_reg = Ridge(alpha=1)
ridge_reg.fit(X, y)
y_pred_ridge = ridge_reg.predict(X)

# Fit a lasso regression model (regularization to control large coefficients)
lasso_reg = Lasso(alpha=0.2)
lasso_reg.fit(X, y)
y_pred_lasso = lasso_reg.predict(X)

# Print the regression results
regression_results(y, y_pred_lin, 'Ordinary')
regression_results(y, y_pred_ridge, 'Ridge')
regression_results(y, y_pred_lasso, 'Lasso')


# Plot the data and the predictions
plt.figure(figsize=(12, 8))

# # Scatter plot of the original data
plt.scatter(X, y, alpha=0.4,ec='k', label='Original Data')

# Plot the ideal regression line (noise free data)
plt.plot(X, y_ideal,  linewidth=2, color='k',label='Ideal, noise free data')

# Plot predictions from the simple linear regression model
plt.plot(X, y_pred_lin,  linewidth=5, label='Linear Regression')

# Plot predictions from the ridge regression model
plt.plot(X, y_pred_ridge, linestyle='--',linewidth=2, label='Ridge Regression')

# Plot predictions from the lasso regression model
plt.plot(X, y_pred_lasso,  linewidth=2, label='Lasso Regression')

plt.xlabel('Feature (X)')
plt.ylabel('Target (y)')
# plt.ylim((0,20))
plt.title('Comparison of predictions with no outliers')
plt.legend()
plt.show()

# Multiple regression regularization and Lasso feature selction
# Now that you have explored regularization for simple, one-dimensional regression, let's take a deeper look at a
# multiple regression modelling scenario. You'll again compare performances of the three inear regression methods
# and then use the Lasso result to select important features to use in another modelling pass.
# Create a high dimensional synthetic dataset with a small number of informative features using make_regression

from sklearn.datasets import make_regression

X, y, ideal_coef = make_regression(n_samples=100, n_features=100, n_informative=10, noise=10, random_state=42, coef=True)

# Get the ideal predictions based on the informative coefficients used in the regression model
ideal_predictions = X @ ideal_coef

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test, ideal_train, ideal_test = train_test_split(X, y, ideal_predictions, test_size=0.3, random_state=42)

# Initialize and fit the linear regression models and use them to predict the target.
lasso = Lasso(alpha=0.1)
ridge = Ridge(alpha=1.0)
linear = LinearRegression()

# Fit the models
lasso.fit(X_train, y_train)
ridge.fit(X_train, y_train)
linear.fit(X_train, y_train)

# Predict on the test set
y_pred_linear = linear.predict(X_test)
y_pred_ridge = ridge.predict(X_test)
y_pred_lasso = lasso.predict(X_test)

# Print the regression results
regression_results(y_test, y_pred_linear, 'Ordinary')
regression_results(y_test, y_pred_ridge, 'Ridge')
regression_results(y_test, y_pred_lasso, 'Lasso')

# Plot the predictions vs actuals
# Let's get some more insight into the perfomance of these models.
fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)

axes[0, 0].scatter(y_test, y_pred_linear, color="red", label="Linear")
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
axes[0, 0].set_title("Linear Regression")
axes[0, 0].set_xlabel("Actual", )
axes[0, 0].set_ylabel("Predicted", )

axes[0, 2].scatter(y_test, y_pred_lasso, color="blue", label="Lasso")
axes[0, 2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
axes[0, 2].set_title("Lasso Regression", )
axes[0, 2].set_xlabel("Actual", )

axes[0, 1].scatter(y_test, y_pred_ridge, color="green", label="Ridge")
axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
axes[0, 1].set_title("Ridge Regression", )
axes[0, 1].set_xlabel("Actual", )

axes[0, 2].scatter(y_test, y_pred_lasso, color="blue", label="Lasso")
axes[0, 2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
axes[0, 2].set_title("Lasso Regression", )
axes[0, 2].set_xlabel("Actual", )

# Line plots for predictions compared to actual and ideal predictions
axes[1, 0].plot(y_test, label="Actual", lw=2)
axes[1, 0].plot(y_pred_linear, '--', lw=2, color='red', label="Linear")
axes[1, 0].set_title("Linear vs Ideal", )
axes[1, 0].legend()

axes[1, 1].plot(y_test, label="Actual", lw=2)
# axes[1,1].plot(ideal_test, '--', label="Ideal", lw=2, color="purple")
axes[1, 1].plot(y_pred_ridge, '--', lw=2, color='green', label="Ridge")
axes[1, 1].set_title("Ridge vs Ideal", )
axes[1, 1].legend()

axes[1, 2].plot(y_test, label="Actual", lw=2)
axes[1, 2].plot(y_pred_lasso, '--', lw=2, color='blue', label="Lasso")
axes[1, 2].set_title("Lasso vs Ideal", )
axes[1, 2].legend()

plt.tight_layout()
plt.show()

# Plot the predictions vs actuals
# Let's get some more insight into the perfomance of these models.

fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)

axes[0, 0].scatter(y_test, y_pred_linear, color="red", label="Linear")
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
axes[0, 0].set_title("Linear Regression")
axes[0, 0].set_xlabel("Actual", )
axes[0, 0].set_ylabel("Predicted", )

axes[0, 2].scatter(y_test, y_pred_lasso, color="blue", label="Lasso")
axes[0, 2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
axes[0, 2].set_title("Lasso Regression", )
axes[0, 2].set_xlabel("Actual", )

axes[0, 1].scatter(y_test, y_pred_ridge, color="green", label="Ridge")
axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
axes[0, 1].set_title("Ridge Regression", )
axes[0, 1].set_xlabel("Actual", )

axes[0, 2].scatter(y_test, y_pred_lasso, color="blue", label="Lasso")
axes[0, 2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
axes[0, 2].set_title("Lasso Regression", )
axes[0, 2].set_xlabel("Actual", )

# Line plots for predictions compared to actual and ideal predictions
axes[1, 0].plot(y_test, label="Actual", lw=2)
axes[1, 0].plot(y_pred_linear, '--', lw=2, color='red', label="Linear")
axes[1, 0].set_title("Linear vs Ideal", )
axes[1, 0].legend()

axes[1, 1].plot(y_test, label="Actual", lw=2)
# axes[1,1].plot(ideal_test, '--', label="Ideal", lw=2, color="purple")
axes[1, 1].plot(y_pred_ridge, '--', lw=2, color='green', label="Ridge")
axes[1, 1].set_title("Ridge vs Ideal", )
axes[1, 1].legend()

axes[1, 2].plot(y_test, label="Actual", lw=2)
axes[1, 2].plot(y_pred_lasso, '--', lw=2, color='blue', label="Lasso")
axes[1, 2].set_title("Lasso vs Ideal", )
axes[1, 2].legend()

plt.tight_layout()
plt.show()


# Model coefficients
# Let's take a look at the coefficients for each model fit
# Model coefficients
linear_coeff = linear.coef_
ridge_coeff = ridge.coef_
lasso_coeff = lasso.coef_


# Plot the coefficients
x_axis = np.arange(len(linear_coeff))
x_labels = np.arange(min(x_axis),max(x_axis),10)
plt.figure(figsize=(12, 6))

plt.scatter(x_axis, ideal_coef,  label='Ideal', color='blue', ec='k', alpha=0.4)
plt.bar(x_axis - 0.25, linear_coeff, width=0.25, label='Linear Regression', color='blue')
plt.bar(x_axis, ridge_coeff, width=0.25, label='Ridge Regression', color='green')
plt.bar(x_axis + 0.25, lasso_coeff, width=0.25, label='Lasso Regression', color='red')

plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')
plt.title('Comparison of Model Coefficients')
plt.xticks(x_labels)
plt.legend()
plt.show()


# Plot the coefficient residuals
x_axis = np.arange(len(linear_coeff))

plt.figure(figsize=(12, 6))

plt.bar(x_axis - 0.25, ideal_coef - linear_coeff, width=0.25, label='Linear Regression', color='blue')
plt.bar(x_axis, ideal_coef - ridge_coeff, width=0.25, label='Ridge Regression', color='green')
# plt.bar(x_axis + 0.25, ideal_coef - lasso_coeff, width=0.25, label='Lasso Regression', color='red')
plt.plot(x_axis, ideal_coef - lasso_coeff, label='Lasso Regression', color='red')

plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')
plt.title('Comparison of Model Coefficient Residuals')
plt.xticks(x_labels)
plt.legend()
plt.show()


# Display two filtered versions of the resulting dataframe:
# 1.	Only those features identified as important by Lasso
# 2.	Only the nonzero ideal coefficient indices

threshold = 5 # selected by inspection of residuals plot

# Create a dataframe containing the Lasso model and ideal coefficients
feature_importance_df = pd.DataFrame({
    'Lasso Coefficient': lasso_coeff,
    'Ideal Coefficient': ideal_coef
})

# Mark the selected features
feature_importance_df['Feature Selected'] = feature_importance_df['Lasso Coefficient'].abs() > threshold


print("Features Identified as Important by Lasso:")
display(feature_importance_df[feature_importance_df['Feature Selected']])

print("\nNonzero Ideal Coefficient Indices")
display(feature_importance_df[feature_importance_df['Ideal Coefficient']>0])


# Use the threshold to select the most important features for use in modelling.
# Also split your data into train, test sets, including the ideal targets. How many features did you end up selecting?
important_features = feature_importance_df[feature_importance_df['Feature Selected']].index
# Filter features
X_filtered = X[:, important_features]
print("Shape of the filtered feature set:", X_filtered.shape)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test, ideal_train, ideal_test = train_test_split(X_filtered, y, ideal_predictions, test_size=0.3, random_state=42)

# Fit and apply the three models to the selected features
# Initialize the models
lasso = Lasso(alpha=0.1)
ridge = Ridge(alpha=1.0)
linear = LinearRegression()

# Fit the models
lasso.fit(X_train, y_train)
ridge.fit(X_train, y_train)
linear.fit(X_train, y_train)

# Predict on the test set
y_pred_linear = linear.predict(X_test)
y_pred_ridge = ridge.predict(X_test)
y_pred_lasso = lasso.predict(X_test)

# Print the regression performance results
regression_results(y_test, y_pred_linear, 'Ordinary')
regression_results(y_test, y_pred_ridge, 'Ridge')
regression_results(y_test, y_pred_lasso, 'Lasso')

# Regenerate the same plots as before and compare the results
# Plot the predictions vs actuals
fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)

axes[0, 0].scatter(y_test, y_pred_linear, color="red", label="Linear")
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
axes[0, 0].set_title("Linear Regression", )
axes[0, 0].set_xlabel("Actual", )

axes[0, 2].scatter(y_test, y_pred_lasso, color="blue", label="Lasso")
axes[0, 2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
axes[0, 2].set_title("Lasso Regression", )
axes[0, 2].set_xlabel("Actual", )
axes[0, 2].set_ylabel("Predicted", )

axes[0, 1].scatter(y_test, y_pred_ridge, color="green", label="Ridge")
axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
axes[0, 1].set_title("Ridge Regression", )
axes[0, 1].set_xlabel("Actual", )

axes[0, 2].scatter(y_test, y_pred_lasso, color="blue", label="Lasso")
axes[0, 2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
axes[0, 2].set_title("Lasso Regression", )
axes[0, 2].set_xlabel("Actual", )
axes[0, 2].set_ylabel("Predicted", )

# Line plots for predictions compared to actual and ideal predictions
axes[1, 0].plot(y_test, label="Actual", lw=2)
axes[1, 0].plot(y_pred_linear, '--', lw=2, color='red', label="Linear")
axes[1, 0].set_title("Linear vs Ideal", )
axes[1, 0].legend()

axes[1, 1].plot(y_test, label="Actual", lw=2)
axes[1, 1].plot(y_pred_ridge, '--', lw=2, color='green', label="Ridge")
axes[1, 1].set_title("Ridge vs Ideal", )
axes[1, 1].legend()

axes[1, 2].plot(y_test, label="Actual", lw=2)
axes[1, 2].plot(y_pred_lasso, '--', lw=2, color='blue', label="Lasso")
axes[1, 2].set_title("Lasso vs Ideal", )
axes[1, 2].legend()

plt.tight_layout()
plt.show()
