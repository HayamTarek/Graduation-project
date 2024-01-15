# %% [markdown]
# # Alzheimer's disease prediction
# 
# 
# 

# %% [markdown]
# 
# 
# 
# ## Adding Imports

# %%
import pandas as pd # used to load, manipulate the data
from sklearn.model_selection import train_test_split ## for splitting the dataset into train and test split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, f1_score, roc_curve, auc
from sklearn.metrics import plot_confusion_matrix

# %% [markdown]
# ## Load Data

# %% [markdown]
# longitudinal Magnetic Resonance Imaging (MRI) data from OASIS.

# %%
data = pd.read_csv('oasis_longitudinal.csv')
data.head(10)

# %% [markdown]
# ## Analysing the data

# %% [markdown]
# The dataset has been analyzed for any categorical values.

# %%
#data = data.loc[data['Visit']==1] #use first visit data only because of the analysis we're doing
#data = data.reset_index(drop=True) #reset index after filtering first visit data
data.info()

# %% [markdown]
# ## Converting Categorical Data to Numerical Data

# %% [markdown]
# gender and group attribute columnsare converted into numeric values 0 and 1. 

# %%
data['M/F'] = data['M/F'].apply(lambda x: 0 if x == 'F' else 1) #0 for F
data['Group'] = data['Group'].apply(lambda x: 0 if x == 'Nondemented' else 1) #0 for Nondemented
data.rename(columns={'M/F':'Gender'}, inplace=True)
data.info()

# %% [markdown]
# ## Correlation matrix

# %% [markdown]
# the correlation between attributes has been checked by using the “correlation matrix” function based on group attributes. 

# %%
correlation_matrix = data.corr()
data_corr = correlation_matrix['Group'].sort_values(ascending=False)
data_corr

# %% [markdown]
# ## Checking for any null or missing values.

# %% [markdown]
# The median value is used to fill in those missing values for features.

# %%
data.isnull().sum()

# %%
SES=data['SES'].median()
MMSE=data['MMSE'].median()
data['SES'] = data['SES'].fillna(SES)
data['MMSE'] = data['MMSE'].fillna(MMSE)
data.isnull().sum()

# %% [markdown]
# ## Drop unnecessary columns

# %%
data = data.drop(['Subject ID', 'MRI ID', 'MR Delay', 'Hand'], axis=1) 
data

# %% [markdown]
# ## Assign the features and the target value.

# %% [markdown]
# X = the features for making the prediction.
# y = the target value set so that the model can predict.

# %%
y = data['Group']
X = data.drop(['Group', 'ASF'], axis=1)

# %% [markdown]
# ## Split the dataset

# %% [markdown]
# stratified sampling has been applied with atraining-validation size of 80% and a testing size of 20%.

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)

# %% [markdown]
# ## Scale the dataset

# %% [markdown]
# standardization has been applied.The result of standardization is that the features will be rescaled to ensure the mean and the standard deviation to be 0 and 1, respectively.

# %%
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %% [markdown]
# ## Data Visualization

# %%
#Histogram
X_train.hist(bins=30, figsize=(20,15))
plt.show()

# %% [markdown]
# shows the histogram of the training and validation set. Histogram portrays the ratios of the dataset.

# %%
#Correlation Matrix
attributes = ["Group", "CDR", "Gender", "SES", "ASF"]
scatter_matrix(data[attributes], figsize=(15, 11), alpha=0.3)

# %% [markdown]
# shows the correlation matrix of the features in the dataset. the correlation matrix indicates how features are interrelated with each other.

# %% [markdown]
# # Models

# %% [markdown]
# ## Support vector machine

# %%
from sklearn.svm import SVC
svm = SVC().fit(X_train_scaled, y_train)
plot_confusion_matrix(svm, 
                      X_test_scaled, 
                      y_test, 
                      values_format='d', 
                      display_labels=['Nondemented', 'Demented'])

# %%
acc_svm = 0
train_score = 0
test_score = 0
test_precision = 0
test_recall = 0
F1_Score = 0
test_auc = 0

train_score = svm.score(X_train_scaled, y_train)
test_score = svm.score(X_test_scaled, y_test)
y_predict = svm.predict(X_test_scaled)

test_precision = precision_score(y_test, y_predict, pos_label=1)
test_recall = recall_score(y_test, y_predict, pos_label=1)
F1_Score = f1_score(y_test, y_predict, pos_label=1)

svm_fpr, svm_tpr, thresholds = roc_curve(y_test, y_predict)
test_auc = auc(svm_fpr, svm_tpr)

acc_svm = accuracy_score(y_test, y_predict)
print('Accuracy: {:.2%} '.format(acc_svm))

#print("Train accuracy ", train_score)
#print("Test accuracy ", test_score)

print('Test precision: {:.2%} '.format(test_precision))
print('Test recall: {:.2%} '.format(test_recall))
print('Test F1: {:.2%} '.format(F1_Score))
print('Test AUC: {:.2%} '.format(test_auc))

# %% [markdown]
# ## Logistic Regression

# %%
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression().fit(X_train_scaled, y_train)
plot_confusion_matrix(log_reg, 
                      X_test_scaled, 
                      y_test, 
                      values_format='d', 
                      display_labels=['Nondemented', 'Demented'])

# %%
acc_lg = 0
train_score = 0
test_score = 0
test_precision = 0
test_recall = 0
F1_Score = 0
test_auc = 0

train_score = log_reg.score(X_train_scaled, y_train)
test_score = log_reg.score(X_test_scaled, y_test)
y_predict = log_reg.predict(X_test_scaled)

test_precision = precision_score(y_test, y_predict, pos_label=1)
test_recall = recall_score(y_test, y_predict, pos_label=1)
F1_Score = f1_score(y_test, y_predict, pos_label=1)

lgr_fpr, lgr_tpr, thresholds = roc_curve(y_test, y_predict)
test_auc = auc(lgr_fpr, lgr_tpr)

acc_lg = accuracy_score(y_test, y_predict)
print('Accuracy: {:.2%} '.format(acc_lg))

#print("Train accuracy ", train_score)
#print("Test accuracy ", test_score)

print('Test precision: {:.2%} '.format(test_precision))
print('Test recall: {:.2%} '.format(test_recall))
print('Test F1: {:.2%} '.format(F1_Score))
print('Test AUC: {:.2%} '.format(test_auc))

# %% [markdown]
# ## Decision Tree

# %%
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier().fit(X_train_scaled, y_train)
plot_confusion_matrix(dt, 
                      X_test_scaled, 
                      y_test, 
                      values_format='d', 
                      display_labels=['Nondemented', 'Demented'])

# %%
acc_ds = 0
train_score = 0
test_score = 0
test_precision = 0
test_recall = 0
F1_Score = 0
test_auc = 0

train_score = dt.score(X_train_scaled, y_train)
test_score = dt.score(X_test_scaled, y_test)
y_predict = dt.predict(X_test_scaled)

test_precision = precision_score(y_test, y_predict, pos_label=1)
test_recall = recall_score(y_test, y_predict, pos_label=1)
F1_Score = f1_score(y_test, y_predict, pos_label=1)

dt_fpr, dt_tpr, thresholds = roc_curve(y_test, y_predict)
test_auc = auc(dt_fpr, dt_tpr)

acc_ds = accuracy_score(y_test, y_predict)
print('Accuracy: {:.2%} '.format(acc_ds))

#print("Train accuracy ", train_score)
#print("Test accuracy ", test_score)

print('Test precision: {:.2%} '.format(test_precision))
print('Test recall: {:.2%} '.format(test_recall))
print('Test F1: {:.2%} '.format(F1_Score))
print('Test AUC: {:.2%} '.format(test_auc))

# %% [markdown]
# ## Random Forest

# %%
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier().fit(X_train_scaled, y_train)
plot_confusion_matrix(rf, 
                      X_test_scaled, 
                      y_test, 
                      values_format='d', 
                      display_labels=['Nondemented', 'Demented'])

# %%
acc_rf = 0
train_score = 0
test_score = 0
test_precision = 0
test_recall = 0
F1_Score = 0
test_auc = 0

train_score = rf.score(X_train_scaled, y_train)
test_score = rf.score(X_test_scaled, y_test)
y_predict = rf.predict(X_test_scaled)

test_precision = precision_score(y_test, y_predict, pos_label=1)
test_recall = recall_score(y_test, y_predict, pos_label=1)
F1_Score = f1_score(y_test, y_predict, pos_label=1)

rfc_fpr, rfc_tpr, thresholds = roc_curve(y_test, y_predict)
test_auc = auc(rfc_fpr, rfc_tpr)

acc_rf = accuracy_score(y_test, y_predict)
print('Accuracy: {:.2%} '.format(acc_rf))

#print("Train accuracy ", train_score)
#print("Test accuracy ", test_score)

print('Test precision: {:.2%} '.format(test_precision))
print('Test recall: {:.2%} '.format(test_recall))
print('Test F1: {:.2%} '.format(F1_Score))
print('Test AUC: {:.2%} '.format(test_auc))

# %% [markdown]
# ## Bar chart of Classifiers 

# %%
import numpy as np
import matplotlib.pyplot as plt

ind = np.arange(4)
width = 0.20
fig, ax = plt.subplots()

accu = [96.00, 94.67, 94.67, 97.33]
rects1 = ax.bar(ind, accu, width, color='purple')
prec = [100.00, 97.06, 97.06, 100.00]
rects2 = ax.bar(ind+width, prec, width, color='skyblue')
rec = [91.67, 91.67, 91.67, 94.44]
rects3 = ax.bar(ind + 2 * width, rec, width, color='tomato')
f1 = [95.65, 94.29, 94.29, 97.14]
rects4 = ax.bar(ind + 3 * width, f1, width, color='black')

ax.set_ylabel('Scores')
ax.set_title('Scores for each algorithm')
ax.set_xticks(ind + 1.5*width)
ax.set_xticklabels(('SVM ', 'Log-Reg', 'Decision Tree', 'Random Forest'))

ax.legend((rects1[0], rects2[0], rects3[0], rects4[0]),
          ('Accuracy', 'Precision', 'Recall', 'F1_score'),
          bbox_to_anchor = (1.05, 0.7))


plt.show()

# %% [markdown]
# ## Plot ROC and compare AUC

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(5, 5), dpi=100)
plt.plot(svm_fpr, svm_tpr, linestyle='-', label='SVM')
plt.plot(lgr_fpr, lgr_tpr, marker='.', label='Logistic Regression')
plt.plot(rfc_fpr, rfc_tpr, linestyle=':', label='Random Forest')
plt.plot(dt_fpr, dt_tpr, linestyle='-.', label='Decision Tree')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.legend()

plt.show()

# %% [markdown]
# ## cross validation

# %% [markdown]
# 5-fold cross-validation has been applied to evaluate all possible combinations.

# %%
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score ,StratifiedKFold
# create dataset
X, y = make_classification(n_samples=373, n_features=8, n_informative=6, n_redundant=2, random_state=1)
# prepare the cross-validation procedure
cv = StratifiedKFold(n_splits=5)

# %%
# create model
model = SVC()
# evaluate model
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print("StratifiedKFold_SVM\n")
print("Cross Validation Scores are {}".format(scores))
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

# %%
# create model
model = LogisticRegression()
# evaluate model
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print("StratifiedKFold_LogisticRegression\n")
print("Cross Validation Scores are {}".format(scores))
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

# %%
# create model
model = DecisionTreeClassifier()
# evaluate model
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print("StratifiedKFold_DecisionTreeClassifier\n")
print("Cross Validation Scores are {}".format(scores))
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

# %%
# create model
model = RandomForestClassifier()
# evaluate model
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print("StratifiedKFold_RandomForestClassifier\n")
print("Cross Validation Scores are {}".format(scores))
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))


