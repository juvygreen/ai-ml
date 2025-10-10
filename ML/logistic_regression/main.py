import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import log_loss
import matplotlib.pyplot as plt

%matplotlib inline

import warnings
warnings.filterwarnings('ignore')


# This data set provides you information about customer preferences, services opted, personal details, etc. which helps you predict customer churn.
# churn_df = pd.read_csv("ChurnData.csv")
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv"
churn_df = pd.read_csv(url)
churn_df

churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'churn']]
churn_df['churn'] = churn_df['churn'].astype('int')
churn_df

# For modeling the input fields X and the target field y need to be fixed. Since that the target to be predicted is 'churn',
# the data under this field will be stored under the variable 'y'.
# We may use any combination or all of the remaining fields as the input. Store these values in the variable 'X'.
X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
X[0:5]  #print the first 5 values


y = np.asarray(churn_df['churn'])
y[0:5] #print the first 5 values

# It is also a norm to standardize or normalize the dataset in order to have all the features at the same scale.
# This helps the model learn faster and improves the model performance. We may make use of StandardScalar function in the Scikit-Learn library.
X_norm = StandardScaler().fit(X).transform(X)
X_norm[0:5]

# The trained model has to be tested and evaluated on data which has not been used during training.
# Therefore, it is required to separate a part of the data for testing and the remaining for training.
# For this, we may make use of the train_test_split function in the scikit-learn library.
X_train, X_test, y_train, y_test = train_test_split( X_norm, y, test_size=0.2, random_state=4)

## Logistic Regression Classifier modeling
# Let's build the model using __LogisticRegression__ from the Scikit-learn package and fit our model with train data set.
LR = LogisticRegression().fit(X_train,y_train)
# Fitting, or in simple terms training, gives us a model that has now learnt from the traning data and can be used to predict the output variable. Let us predict the churn parameter for the test data set.
yhat = LR.predict(X_test)
yhat[:10]
yhat_prob = LR.predict_proba(X_test)
yhat_prob[:10]


# Since the purpose here is to predict the 1 class more acccurately,
# you can also examine what role each input feature has to play in the prediction of the 1 class. Consider the code below.
coefficients = pd.Series(LR.coef_[0], index=churn_df.columns[:-1])
coefficients.sort_values().plot(kind='barh')
plt.title("Feature Coefficients in Logistic Regression Churn Model")
plt.xlabel("Coefficient Value")
plt.show()

# Log loss (Logarithmic loss), also known as Binary Cross entropy loss, is a function that generates a loss value based on
# the class wise prediction probabilities and the actual class labels. The lower the log loss value,
# the better the model is considered to be.
log_loss(y_test, yhat_prob)
