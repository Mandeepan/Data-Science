
# coding: utf-8

# In[50]:


import pandas as pd
import random
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score


# In[2]:


#input data 
creditData=pd.read_csv('/Users/Mandy/Study/SpringBoard/Capstone 1/Data/Modified Dataset.csv', header=0)


# In[3]:


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


# In[4]:


# randomly pick the monthly datas to each cells for economic datas

for index,rowValue in interest_rate.iteritems():
        interest_rate[index]=random.choice([1.88,2,2.13])

for index,rowValue in employed_persons.iteritems():
        employed_persons[index]=random.choice([9916,9929,9931,9933,9949,9969])
        
for index,rowValue in cpi.iteritems():
        cpi[index]=random.choice([92.02,92.53,92.82,93.9,94.6,94.73])

for index,rowValue in consumer_confidence.iteritems():
        consumer_confidence[index]=random.choice([73.15,73.06,74.79,73.15,73.38,71.42])


# In[5]:


#Data Exploration
default_next_month.value_counts()


# In[6]:


creditData.groupby('Default_Next_Month').mean()


# In[7]:


creditData.groupby('SEX').mean()


# In[8]:


creditData.groupby('MARRIAGE').mean()


# In[9]:


creditData.groupby('Interest_Rate').mean()


# In[10]:


creditData.groupby('Employed_Persons').mean()


# In[11]:


creditData.groupby('CPI').mean()


# In[12]:


creditData.groupby('Consumer_Confidence').mean()


# In[19]:


creditData['x_bin']=pd.cut(x=creditData.AGE, bins=[20,30,40,50,60,70]) 


# In[14]:


get_ipython().magic('matplotlib inline')
#bar chart of age grouped by default
pd.crosstab(creditData.x_bin,default_next_month).plot(kind='bar')
plt.title('Age Distribution by  Credit Default Status')
plt.xlabel('Age')
plt.ylabel('Frequency')


# In[20]:


#Separate the dataset into dependent and independent variables
X=creditData.drop('Default_Next_Month',axis=1)
X=X.drop('ID',axis=1)
X=X.drop('x_bin',axis=1) 
y=creditData['Default_Next_Month']

#then seperate it into training set, validation set and test set
X_train,X_val_test,y_train,y_val_test = train_test_split(X,y, test_size=0.3)
X_val,X_test,y_val,y_test=train_test_split(X_val_test,y_val_test,test_size=0.5) 


# In[21]:


#######MODEL ONE: LOGISTIC REGRESSION MODEL######
LR_classifier = LogisticRegression()
LR_classifier.fit(X_train,y_train)


# In[55]:


#If the evaluation score suggests not overfitting, predict the default status for each sample in test set
LR_evaluated=LR_classifier.score(X_val,y_val)
print("The accuracy of Logistic Regression Model on validation set is :",LR_evaluated)
if LR_evaluated >0.97:
    print (" The Logistic Regression Model is overfitting in this case.")
else:
    LR_predicted=LR_classifier.predict(X_test)
    #generate default probabilities based on test set
    LR_probs_default=np.sum(LR_predicted)/len(LR_predicted)
    print("The Default Probability based on Logistic Regression Model is : ", LR_probs_default)
    LR_accuracy=LR_classifier.score(X_test,y_test)
    print("The accuracy of Logistic Regression Model on test set is : ",LR_accuracy)
    print(metrics.confusion_matrix(y_test,LR_predicted))
    print (metrics.classification_report(y_test,LR_predicted))
    #10-fold cross-validation 
    LR_cross_val_scores = cross_val_score(LogisticRegression(),X,y,scoring='accuracy',cv=10)
    print("The 10 fold cross validation score based on Logistic Regression Model is: ",LR_cross_val_scores.mean())


# In[51]:


######MODEL TWO: DECISION TREE MODEL#####
DT_classifier = DecisionTreeRegressor()
DT_classifier.fit(X_train,y_train) 


# In[83]:


#If the evaluation score suggests not overfitting, predict the default status for each sample in test set
DT_evaluated_diff=DT_classifier.predict(X_val)-y_val
DT_evaluated_count=pd.value_counts(DT_evaluated_diff)
DT_evaluated= DT_evaluated_count[0]/len(DT_evaluated_diff) 
print("The accuracy of Decision Tree Model on validation set is :",DT_evaluated)
if DT_evaluated >0.97:
    print (" The Decision Tree Model is overfitting in this case.")
else:
    DT_predicted=DT_classifier.predict(X_test)
    #generate default probabilities based on test set
    DT_probs_default=np.sum(DT_predicted)/len(DT_predicted)
    print("The Default Probability based on Decision Tree Model is : ", DT_probs_default)
    DT_test_diff=DT_predicted-y_test
    DT_test_count=pd.value_counts(DT_test_diff)
    DT_accuracy= DT_test_count[0]/len(DT_test_diff) 
    print("The accuracy of Decision Tree Model on test set is : ",DT_accuracy)
    print(metrics.confusion_matrix(y_test,DT_predicted))
    print (metrics.classification_report(y_test,DT_predicted))
    #10-fold cross-validation 
    DT_cross_val_scores = cross_val_score(DecisionTreeRegressor(),X,y,scoring='accuracy',cv=10)
    print("The 10 fold cross validation score based on Decision Tress Model is: ",DT_cross_val_scores.mean())


# In[80]:




