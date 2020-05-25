# -*- coding: utf-8 -*-

"""
1. Problem Definition:

Given the medical parameters and details about a patient, can we predict whether or not they have heart disease?


2. Data/ Features

1. age - age in years
 
2. sex - (1 = male; 0 = female)
 
3. cp - chest pain type

  0: Typical angina: chest pain related decrease blood supply to the heart
  1: Atypical angina: chest pain not related to heart
  2: Non-anginal pain: typically esophageal spasms (non heart related)
  3: Asymptomatic: chest pain not showing signs of disease
 
4. trestbps - resting blood pressure (in mm Hg on admission to the hospital) anything above 130-140 is typically cause for concern

5. chol - serum cholestoral in mg/dl
 
  serum = LDL + HDL + .2 * triglycerides
  above 200 is cause for concern

6. fbs - (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
  
  '>126' mg/dL signals diabetes

7. restecg - resting electrocardiographic results

  0: Nothing to note
  1: ST-T Wave abnormality
  can range from mild symptoms to severe problems
  signals non-normal heart beat
  2: Possible or definite left ventricular hypertrophy
  Enlarged heart's main pumping chamber
 
8. thalach - maximum heart rate achieved

9. exang - exercise induced angina (1 = yes; 0 = no)
 
10. oldpeak - ST depression induced by exercise relative to rest looks at stress of heart during excercise unhealthy heart will stress more

11. slope - the slope of the peak exercise ST segment
 
  0: Upsloping: better heart rate with excercise (uncommon)
  1: Flatsloping: minimal change (typical healthy heart)
  2: Downslopins: signs of unhealthy heart
 
12. ca - number of major vessels (0-3) colored by flourosopy
  
  colored vessel means the doctor can see the blood passing through
  the more blood movement the better (no clots)
 
13. thal - thalium stress result
  
  1,3: normal
  6: fixed defect: used to be defect but ok now
  7: reversable defect: no proper blood movement when excercising
 
14. target - have disease or not (1=yes, 0=no) (= the predicted attribute)

"""

# %% Importing Libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
%matplotlib qt

sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")

# %% Reading Data Set

df = pd.read_csv('G:/My Files/Work/Data Science Resources/Data Science Projects/Heart Disease Prediction/heart.csv')

df.head()

# %% Some info about the dataset

df.info()

df.shape

df.describe()

# %% Checking for imbalanced dataset

df.target.value_counts()

df.target.value_counts().plot(kind="bar", color=["salmon", "lightblue"])

# %% Checking for missing values

df.isna().sum()

# %% Number of unique values

categorical_val = []
continous_val = []

for column in df.columns:
    print('\n')
    print(column," : ", df[column].nunique())
    if len(df[column].unique()) <= 10:
        categorical_val.append(column)
    else:
        continous_val.append(column)

# %% PLotting based on categorical values
        
plt.figure(figsize=(15, 15))

for i, column in enumerate(categorical_val, 1):
    plt.subplot(3, 3, i)
    df[df["target"] == 0][column].hist(bins=35, color='blue', label='Have Heart Disease = NO', alpha=0.6)
    df[df["target"] == 1][column].hist(bins=35, color='red', label='Have Heart Disease = YES', alpha=0.6)
    plt.legend()
    plt.xlabel(column)

# %% Plotting based on numerical values
    
plt.figure(figsize=(15, 15))

for i, column in enumerate(continous_val, 1):
    plt.subplot(3, 2, i)
    df[df["target"] == 0][column].hist(bins=35, color='blue', label='Have Heart Disease = NO', alpha=0.6)
    df[df["target"] == 1][column].hist(bins=35, color='red', label='Have Heart Disease = YES', alpha=0.6)
    plt.legend()
    plt.xlabel(column)

# %% Correlation heatmap

corrmat = df.corr()
corr_feat = corrmat.index
plt.figure(figsize = (15,15))
sns.heatmap(df[corr_feat].corr(),annot = True, cmap = 'coolwarm')


# %% Correlation Values with Target

df.drop('target', axis=1).corrwith(df.target).plot(kind='bar', 
           figsize=(12, 8),title="Correlation with target")

"""
fbs and chol are the lowest correlated with the target variable.
All other variables have a significant correlation with the target variable.

"""

# %% Data Preprocessing

"""
After exploring the dataset, I observed that I need to convert some categorical variables 
into dummy variables and scale all the values before training the Machine Learning models. 
First, I'll use the get_dummies method to create dummy columns for categorical variables.
"""

df.columns

dataset = pd.get_dummies(df, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang','slope', 'ca', 'thal'])


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# %% Scaling the numerical features

scaler = StandardScaler()
columns_to_scale = ['age','trestbps','chol','thalach','oldpeak']
dataset[columns_to_scale] = scaler.fit_transform(dataset[columns_to_scale])

# %% Train Test Split

X = dataset.drop('target',axis = 1)
y = dataset['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

"""
Now we've got our data split into training and test sets, it's time to build a machine learning model.

We're going to try different machine learning models:
 
 1. Logistic Regression
 2. K-Nearest Neighbours Classifier
 3. Decision Tree Classifier
 4. Random Forest Classifier

"""

from sklearn.model_selection import cross_val_score

# %% Logistic Regression

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)


score = cross_val_score(log_reg, X, y, cv =10)
score.mean()

# %% Predictions

predictions = log_reg.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test,predictions))

print('\n')

print(classification_report(y_test,predictions))

# %% K-Nearest Neighbours Classifier

from sklearn.neighbors import KNeighborsClassifier

knn_scores = []

for k in range(1,41):
    knn = KNeighborsClassifier(n_neighbors= k)
    score = cross_val_score(knn, X, y, cv=10)   
    knn_scores.append(score.mean())
    
plt.figure(figsize = (15,8))
plt.plot([k for k in range(1,41)], knn_scores, color = 'red')

for i in range(1,41):
    plt.text(i, knn_scores[i-1], (i, knn_scores[i-1]))

plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Scores')
plt.title('K Neighbors Classifier scores for different K values')
plt.tight_layout()

knn = KNeighborsClassifier(n_neighbors = 12)
knn.fit(X_train,y_train)

score = cross_val_score(knn, X, y, cv =10)
score.mean()

# %% Predictions

predictions = knn.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test,predictions))

print(classification_report(y_test,predictions))

# %% Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()

dtree.fit(X_train,y_train)

score = cross_val_score(dtree, X, y, cv =10)
score.mean()

# %% Predictions

pred = dtree.predict(X_test)

print(confusion_matrix(y_test,pred))

print(classification_report(y_test,pred))

# %% Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators = 200)

rfc.fit(X_train,y_train)

score = cross_val_score(rfc, X, y, cv =10)
score.mean()

# %% Predictions

predictions = rfc.predict(X_test)

print(confusion_matrix(y_test,predictions))

print(classification_report(y_test,predictions))


# %% Deployment 

"""
Now we are going to create a pickle file to deploy the model. We would need to pass 13 features according 
to the data to get a prediction.
We have found that K-Nearest Neighbours is a good choice.

"""

import pickle

# %% Saving model to disk

pickle.dump(knn, open('model.pkl','wb'))

# %% Loading model to compare the results

model = pickle.load(open('model.pkl','rb'))


print(model.predict([[-0.2,-0.5,.8,.9,.34,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]]))








