
# coding: utf-8

# In[6]:


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
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

#set the random seed as zero
np.random.seed(0)


# In[7]:


#input data 
creditData=pd.read_csv('/Users/Mandy/Study/SpringBoard/Capstone 1/Data/Modified Dataset.csv', header=0)


# In[8]:


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


# In[9]:


# randomly pick the monthly datas to each cells for economic datas

for index,rowValue in interest_rate.iteritems():
        interest_rate[index]=random.choice([1.88,2,2.13])

for index,rowValue in employed_persons.iteritems():
        employed_persons[index]=random.choice([9916,9929,9931,9933,9949,9969])
        
for index,rowValue in cpi.iteritems():
        cpi[index]=random.choice([92.02,92.53,92.82,93.9,94.6,94.73])

for index,rowValue in consumer_confidence.iteritems():
        consumer_confidence[index]=random.choice([73.15,73.06,74.79,73.15,73.38,71.42])


# In[10]:


#Data Exploration
default_next_month.value_counts()


# In[11]:


creditData.groupby('Default_Next_Month').mean()


# In[12]:


creditData.groupby('SEX').mean()


# In[13]:


creditData.groupby('MARRIAGE').mean()


# In[14]:


creditData.groupby('Interest_Rate').mean()


# In[15]:


creditData.groupby('Employed_Persons').mean()


# In[16]:


creditData.groupby('CPI').mean()


# In[17]:


creditData.groupby('Consumer_Confidence').mean()


# In[18]:


creditData['x_bin']=pd.cut(x=creditData.AGE, bins=[20,30,40,50,60,70]) 


# In[19]:


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
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.15)
 


# In[21]:


#######MODEL ONE: LOGISTIC REGRESSION MODEL######
LR_classifier = LogisticRegression()
LR_classifier.fit(X_train,y_train)


# In[22]:


#10-fold cross-validation,if the evaluation score suggests not overfitting, predict the default status for each sample in test set
LR_cross_val_scores = cross_val_score(LR_classifier,X_train,y_train,scoring='accuracy',cv=10)
print("The 10 fold cross validation score based on Logistic Regression Model is: ",'%.3f'%LR_cross_val_scores.mean())

if LR_cross_val_scores.mean() >0.97:
    print (" The Logistic Regression Model is overfitting in this case.")
else:
    LR_predicted=LR_classifier.predict(X_test)
    #generate default probabilities based on test set
    LR_prob_default=np.sum(LR_predicted)/len(LR_predicted)
    print("The Default Probability based on Logistic Regression Model is :",'%.3f'%LR_prob_default)
    LR_accuracy=LR_classifier.score(X_test,y_test)
    print("The accuracy of Logistic Regression Model on test set is : ",'%.3f'%LR_accuracy)
    print(metrics.confusion_matrix(y_test,LR_predicted))
    print (metrics.classification_report(y_test,LR_predicted))
    


# In[23]:


#create a dataframe to store evaluation info of different models
evaluation = pd.DataFrame({'Model':['Logistic Regression'],'Default_Probability':[LR_prob_default],'Cross_Validation_Accuracy':[LR_cross_val_scores.mean()],'Test_Accuracy':[LR_accuracy]})
evaluation=evaluation[['Model','Default_Probability','Cross_Validation_Accuracy','Test_Accuracy']]
print(evaluation)


# In[24]:


#########MODEL TWO: RANDOM FOREST MODEL#########
RF_classifier = RandomForestClassifier(random_state=0)


# In[38]:


# 10 Fold Cross Validation over the entire training set
kf=KFold(n_splits=10,shuffle=False)
RF_cross_val_score=0
for train_index, test_index in kf.split(X_train):
    X_train_subset, X_test_subset=X_train.iloc[train_index],X_train.iloc[test_index]
    y_train_subset, y_test_subset=y_train.iloc[train_index],y_train.iloc[test_index]
    RF_classifier.fit(X_train_subset,y_train_subset)
    RF_cross_val_score=RF_cross_val_score+RF_classifier.score(X_test_subset,y_test_subset)
RF_cross_val_score=RF_cross_val_score/10
print("The 10 fold cross validation score based on Naive Bayes Model(Non Scaled) is: ",'%.3f'%RF_cross_val_score)


# In[26]:


# if the score is less than 0.975, then build the classifier with the entire training set
if RF_cross_val_score >0.97:
    print (" The Random Forest Model is overfitting in this case.")
else:
    RF_classifier.fit(X_train,y_train)
    RF_predicted=RF_classifier.predict(X_test)
    RF_prob_default=np.sum(RF_predicted)/len(RF_predicted)
    print("The Default Probability based on Random Forest Model is :",'%.3f'%RF_prob_default)
    RF_accuracy=RF_classifier.score(X_test,y_test)
    print("The accuracy of Random Forest Model on test set is : ",'%.3f'%RF_accuracy)
    print(metrics.confusion_matrix(y_test,RF_predicted))
    print(metrics.classification_report(y_test,RF_predicted))


# In[27]:


features_weight=RF_classifier.fit(X_test,y_test).feature_importances_
feature_index = np.where(features_weight==max(features_weight))
feature_index=int(feature_index[0][0])
print("The most important feature is:",(list(X_test)[feature_index]))


# In[28]:


#output the result into the existing evaluation dataframe to compare with other model
new_evaluation=pd.DataFrame({'Model':["Random Forest"],'Default_Probability':[RF_prob_default],'Cross_Validation_Accuracy':[RF_cross_val_score],'Test_Accuracy':[RF_accuracy]})
evaluation = evaluation.append(new_evaluation)
evaluation=evaluation[['Model','Default_Probability','Cross_Validation_Accuracy','Test_Accuracy']]
print(evaluation)


# In[58]:


########MODEL THREE: NAIVE BAYES #######################
### Non Scaled ####
#10-fold cross validation for the entire standardized training data without scaling/normalization/standarization
NB_classifier=GaussianNB()
kf=KFold(n_splits=10,shuffle=False)
NB_cross_val_score=0
for train_index, test_index in kf.split(X_train):
    X_train_subset, X_test_subset=X_train.iloc[train_index],X_train.iloc[test_index]
    y_train_subset, y_test_subset=y_train.iloc[train_index],y_train.iloc[test_index]
    NB_classifier.fit(X_train_subset,y_train_subset)
    NB_cross_val_score=NB_cross_val_score+NB_classifier.score(X_test_subset,y_test_subset)
    
NB_cross_val_score=NB_cross_val_score/10
print("The 10 fold cross validation score based on Naive Bayes Model is: ",'%.3f'%NB_cross_val_score)


# In[35]:


if NB_cross_val_score >0.97:
    print ("The Naive Bayes Model is overfitting in this case.")
else:
    NB_classifier.fit(X_train,y_train)
    NB_predicted=NB_classifier.predict(X_test)
    NB_prob_default=np.sum(NB_predicted)/len(NB_predicted)
    print("The Default Probability based on Naive Bayes Model is :",'%.3f'%NB_prob_default)
    NB_accuracy=NB_classifier.score(X_test,y_test)
    print("The accuracy of Naive Bayes Model on test set is : ",'%.3f'%NB_accuracy)


# In[36]:


#output the result into the existing evaluation dataframe to compare with other models
new_evaluation=pd.DataFrame({'Model':["Naive Bayes_NonScaled"],'Default_Probability':[NB_prob_default],'Cross_Validation_Accuracy':[NB_cross_val_score],'Test_Accuracy':[NB_accuracy]})
evaluation = evaluation.append(new_evaluation)
evaluation=evaluation[['Model','Default_Probability','Cross_Validation_Accuracy','Test_Accuracy']]
print(evaluation)


# In[56]:


#10-fold cross validation for the entire standardized training set
NB_cross_val_score_scaled=0
for NBtrain_index, NBtest_index in kf.split(X_train):
    X_train_subset, X_test_subset=X_train.iloc[NBtrain_index],X_train.iloc[NBtest_index]
    y_train_subset, y_test_subset=y_train.iloc[NBtrain_index],y_train.iloc[NBtest_index]
    #standardization
    X_train_scaled_subset=(X_train_subset-X_train_subset.mean())/X_train_subset.std()
    y_train_scaled_subset=(y_train_subset-y_train_subset.mean())/y_train_subset.std()
    X_test_scaled_subset=(X_test_subset-X_test_subset.mean())/X_test_subset.std()
    y_test_scaled_subset=(y_test_subset-y_test_subset.mean())/y_test_subset.std()
    # turn the dataframe subset, which is 2D array into 1D array
    y_train_scaled_subset=np.asarray(y_train_scaled_subset,dtype="|S6")
    y_test_scaled_subset=np.asarray(y_test_scaled_subset,dtype="|S6")
    NB_classifier.fit(X_train_scaled_subset,y_train_scaled_subset)
    NB_predicted_subset=NB_classifier.predict(X_test_scaled_subset)
    print(NB_predicted_subset)
    print(y_test_scaled_subset)
    print("####################################")
    NB_cross_val_score_scaled=NB_cross_val_score_scaled+NB_classifier.score(X_test_scaled_subset,y_test_scaled_subset)

    NB_cross_val_score_scaled=NB_cross_val_score_scaled/10
print("The 10 fold cross validation score based on Naive Bayes Model(Scaled) is: ",'%.3f'%NB_cross_val_score_scaled)


# In[61]:


if NB_cross_val_score >0.97:
    print (" The Naive Bayes Model (scaled) is overfitting in this case.")
else:
    #standardization
    X_train_scaled=(X_train-X_train.mean())/X_train.std()
    y_train_scaled=(y_train-y_train.mean())/y_train.std()
    X_test_scaled=(X_test-X_test.mean())/X_test.std()
    y_test_scaled=(y_test-y_test.mean())/y_test.std()
    # turn the dataframe subset, which is 2D array into 1D array
    y_train_scaled=np.asarray(y_train_scaled,dtype="|S6")
    y_test_scaled=np.asarray(y_test_scaled,dtype="|S6") 
    NB_classifier.fit(X_train_scaled,y_train_scaled)
    NB_predicted_scaled=NB_classifier.predict(X_test_scaled)
    NB_predicted_default=pd.value_counts(pd.Series(NB_predicted_scaled))[1]
    NB_predicted_nondefault=pd.value_counts(pd.Series(NB_predicted_scaled))[0] 
    NB_prob_default_scaled=NB_predicted_default/(NB_predicted_default+NB_predicted_nondefault)
    print("The Default Probability based on Naive Bayes Model is :",'%.3f'%NB_prob_default_scaled)
    #print(pd.value_counts(pd.Series(NB_predicted_scaled)))
    #print(y_test_scaled.)
    #print(accuracy_score(y_test_scaled,NB_predicted_scaled))
    NB_accuracy=NB_classifier.score(X_test_scaled,y_test_scaled)
    print("The accuracy of Naive Bayes Model on test set is : ",'%.3f'%NB_accuracy)


# In[ ]:


#output the result into the existing evaluation dataframe to compare with other models
#new_evaluation=pd.DataFrame({'Model':["Naive Bayes"],'Default_Probability':[NB_prob_default],'Cross_Validation_Accuracy':[NB_cross_val_score],'Test_Accuracy':[NB_accuracy]})
#evaluation = evaluation.append(new_evaluation)
#evaluation=evaluation[['Model','Default_Probability','Cross_Validation_Accuracy','Test_Accuracy']]
#print(evaluation)

