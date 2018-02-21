
# coding: utf-8

# In[126]:


import pandas as pd
import random
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score


# In[127]:


#input data 
creditData=pd.read_csv('/Users/Mandy/Study/SpringBoard/Capstone 1/Data/Modified Dataset.csv', header=0)


# In[128]:


#subset macro-economic data
limit_bal=creditData.iloc[:,1]
sex=creditData.iloc[:,2]
education=creditData.iloc[:,3]
marriage=creditData.iloc[:,4]
age=creditData.iloc[:,5]
pay_0=creditData.iloc[:,6]
pay_2=creditData.iloc[:,7]
pay_3=creditData.iloc[:,8]
pay_4=creditData.iloc[:,9]
pay_5=creditData.iloc[:,10]
pay_6=creditData.iloc[:,11]
bill_amt1=creditData.iloc[:,12]
bill_amt2=creditData.iloc[:,13]
bill_amt3=creditData.iloc[:,14]
bill_amt4=creditData.iloc[:,15]
bill_amt5=creditData.iloc[:,16]
bill_amt6=creditData.iloc[:,17]
pay_amt1=creditData.iloc[:,18]
pay_amt2=creditData.iloc[:,19]
pay_amt3=creditData.iloc[:,20]
pay_amt4=creditData.iloc[:,21]
pay_amt5=creditData.iloc[:,22]
pay_amt6=creditData.iloc[:,23]
interest_rate=creditData.iloc[:,24]
employed_persons=creditData.iloc[:,25]
cpi=creditData.iloc[:,26]
consumer_confidence=creditData.iloc[:,27]
default_next_month=creditData.iloc[:,28]


# In[129]:


# randomly pick the monthly datas to each cells for economic datas

for index,rowValue in interest_rate.iteritems():
        interest_rate[index]=random.choice([1.88,2,2.13])

for index,rowValue in employed_persons.iteritems():
        employed_persons[index]=random.choice([9916,9929,9931,9933,9949,9969])
        
for index,rowValue in cpi.iteritems():
        cpi[index]=random.choice([92.02,92.53,92.82,93.9,94.6,94.73])

for index,rowValue in consumer_confidence.iteritems():
        consumer_confidence[index]=random.choice([73.15,73.06,74.79,73.15,73.38,71.42])


# In[130]:


#Data Exploration
default_next_month.value_counts()


# In[131]:


creditData.groupby('Default_Next_Month').mean()


# In[132]:


creditData.groupby('SEX').mean()


# In[133]:


creditData.groupby('MARRIAGE').mean()


# In[134]:


creditData.groupby('Interest_Rate').mean()


# In[135]:


creditData.groupby('Employed_Persons').mean()


# In[136]:


creditData.groupby('CPI').mean()


# In[137]:


creditData.groupby('Consumer_Confidence').mean()


# In[138]:


get_ipython().magic('matplotlib inline')
#bar chart of age grouped by default
pd.crosstab(age,default_next_month).plot(kind='bar')
plt.title('Age Distribution by  Credit Default Status')
plt.xlabel('Age')
plt.ylabel('Frequency')


# In[139]:


#use dmatrics to prepare dependent variable matric and the independent variable vector
y,X=dmatrices('Default_Next_Month~LIMIT_BAL+SEX+EDUCATION+MARRIAGE+AGE+PAY_0+PAY_2+PAY_3+PAY_4+PAY_5+PAY_6+BILL_AMT1+BILL_AMT2+BILL_AMT3+BILL_AMT4+BILL_AMT5+BILL_AMT6+PAY_AMT1+PAY_AMT2+PAY_AMT3+PAY_AMT4+PAY_AMT5+PAY_AMT6+Interest_Rate+Employed_Persons+CPI+Consumer_Confidence',creditData, return_type='dataframe')
#flattern y into a 1-D array
y=np.ravel(y)
#then seperate it into training and test sets
X_training,X_test,y_training,y_test = train_test_split(X,y, test_size=0.3, random_state =0)


# In[140]:


#fit logistic regression into the training set
classifier = LogisticRegression(random_state=0)
classifier.fit(X_training,y_training)


# In[141]:


#predict the default status for each sample in test set
predicted_status=classifier.predict(X_test)
print (predicted_status)


# In[142]:


#generate default probabilities based on test set
probs = classifier.predict_proba(X_test)
print(probs)


# In[143]:


# Model evaluation metrics
print (metrics.accuracy_score(y_test,predicted_status))
print (metrics.roc_auc_score(y_test,predicted_status))


# In[144]:


print(metrics.confusion_matrix(y_test,predicted_status))
print (metrics.classification_report(y_test,predicted_status))


# In[145]:


#10-fold cross-validation 
scores = cross_val_score(LogisticRegression(),X,y,scoring='accuracy',cv=10)
print(scores)
print(scores.mean())

