
# coding: utf-8

# In[1]:


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
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

#set the random seed as zero
np.random.seed(0)


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


# In[13]:


creditData['x_bin']=pd.cut(x=creditData.AGE, bins=[20,30,40,50,60,70]) 


# In[14]:


get_ipython().magic('matplotlib inline')
#bar chart of age grouped by default
pd.crosstab(creditData.x_bin,default_next_month).plot(kind='bar')
plt.title('Age Distribution by  Credit Default Status')
plt.xlabel('Age')
plt.ylabel('Frequency')


# In[15]:


#Separate the dataset into dependent and independent variables
X=creditData.drop('Default_Next_Month',axis=1)
X=X.drop('ID',axis=1)
X=X.drop('x_bin',axis=1) 
y=creditData['Default_Next_Month']

#then seperate it into training set, validation set and test set
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.15)
 


# In[16]:


#######MODEL ONE: LOGISTIC REGRESSION MODEL######
LR_classifier = LogisticRegression()


# In[17]:


#10-fold cross-validation,if the evaluation score suggests not overfitting, predict the default status for each sample in test set
LR_cross_val_scores = cross_val_score(LR_classifier,X_train,y_train,n_jobs=-1,scoring='accuracy',cv=10)
print("The 10 fold cross validation score based on Logistic Regression Model is: %0.3f(+-%0.3f)"%(LR_cross_val_scores.mean(),LR_cross_val_scores.std()*2))

if LR_cross_val_scores.mean() >0.97:
    print (" The Logistic Regression Model is overfitting in this case.")
else:
    LR_classifier.fit(X_train,y_train)
    LR_predicted=LR_classifier.predict(X_test)
    #generate default probabilities based on test set
    LR_prob_default=np.sum(LR_predicted)/len(LR_predicted)
    print("The Default Probability based on Logistic Regression Model is :",'%.3f'%LR_prob_default)
    LR_accuracy=LR_classifier.score(X_test,y_test)
    print("The accuracy of Logistic Regression Model on test set is : ",'%.3f'%LR_accuracy)


# In[18]:


#create a dataframe to store evaluation info of different models
evaluation = pd.DataFrame({'Model':['Logistic Regression'],'Default_Probability':[LR_prob_default],'Cross_Validation_Accuracy':[LR_cross_val_scores.mean()],'Test_Accuracy':[LR_accuracy]})
evaluation=evaluation[['Model','Default_Probability','Cross_Validation_Accuracy','Test_Accuracy']]
print(evaluation)


# In[19]:


#########MODEL TWO: RANDOM FOREST MODEL#########
RF_classifier = RandomForestClassifier(random_state=0)


# In[20]:


# 10 Fold Cross Validation over the entire training set
RF_cross_val_scores=cross_val_score(RF_classifier,X_train,y_train,cv=10,n_jobs=-1,scoring='accuracy')
print("The 10 fold cross validation score based on Random Forest Model is: %0.3f(+/-%0.3f)"%(RF_cross_val_scores.mean(),RF_cross_val_scores.std()*2))

# if the score is less than 0.975, then build the classifier with the entire training set
if RF_cross_val_scores.mean() >0.97:
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


# In[21]:


features_weight=RF_classifier.fit(X_test,y_test).feature_importances_
feature_index = np.where(features_weight==max(features_weight))
feature_index=int(feature_index[0][0])
print("The most important feature is:",(list(X_test)[feature_index]))


# In[22]:


#output the result into the existing evaluation dataframe to compare with other model
new_evaluation=pd.DataFrame({'Model':["Random Forest"],'Default_Probability':[RF_prob_default],'Cross_Validation_Accuracy':[RF_cross_val_scores.mean()],'Test_Accuracy':[RF_accuracy]})
evaluation = evaluation.append(new_evaluation)
evaluation=evaluation[['Model','Default_Probability','Cross_Validation_Accuracy','Test_Accuracy']]
print(evaluation)


# In[23]:


########MODEL THREE: NAIVE BAYES -Non Scaled #######################
NB_classifier=GaussianNB()
#10-fold cross validation for the entire standardized training data without scaling/normalization/standarization
NB_NonScaled_cross_val_scores=cross_val_score(NB_classifier,X_train,y_train,cv=10,n_jobs=-1,scoring='accuracy')
print("The 10 fold cross validation score based on Naive Bayes Model(Non Scaled) is: %0.3f(+/-%0.3f)"%(NB_NonScaled_cross_val_scores.mean(),NB_NonScaled_cross_val_scores.std()*2))


# In[24]:


if NB_NonScaled_cross_val_scores.mean() >0.97:
    print ("The Naive Bayes Model (Non Scaled) is overfitting in this case.")
else:
    NB_classifier.fit(X_train,y_train)
    NB_NonScaled_predicted=NB_classifier.predict(X_test)
    NB_NonScaled_prob_default=np.sum(NB_NonScaled_predicted)/len(NB_NonScaled_predicted)
    print("The Default Probability based on Naive Bayes Model(Non Scaled) is :",'%.3f'%NB_NonScaled_prob_default)
    NB_NonScaled_accuracy=NB_classifier.score(X_test,y_test)
    print("The accuracy of Naive Bayes Model(Non Scaled) on test set is : ",'%.3f'%NB_NonScaled_accuracy)


# In[25]:


#output the result into the existing evaluation dataframe to compare with other models
new_evaluation=pd.DataFrame({'Model':["Naive Bayes_NonScaled"],'Default_Probability':[NB_NonScaled_prob_default],'Cross_Validation_Accuracy':[NB_NonScaled_cross_val_scores.mean()],'Test_Accuracy':[NB_NonScaled_accuracy]})
evaluation = evaluation.append(new_evaluation)
evaluation=evaluation[['Model','Default_Probability','Cross_Validation_Accuracy','Test_Accuracy']]
print(evaluation)


# In[26]:


###################MODEL FOUR: NAIVE BAYES ( SCALED) ############################
#Standardization
scaler=StandardScaler()
NB_classifier_scaled=GaussianNB()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.fit_transform(X_test)


# In[27]:


#10-fold cross validation for the entire standardized training set
NB_Scaled_cross_val_scores=cross_val_score(NB_classifier_scaled,X_train_scaled,y_train,cv=10,n_jobs=-1,scoring='accuracy')
print("The 10 fold cross validation score based on Naive Bayes Model(Scaled) is: %0.3f(+/-%0.3f)" %(NB_Scaled_cross_val_scores.mean(),NB_Scaled_cross_val_scores.std()*2))


# In[28]:


if NB_Scaled_cross_val_scores.mean() >0.97:
    print (" The Naive Bayes Model (scaled) is overfitting in this case.")
else:
    NB_classifier_scaled.fit(X_train_scaled,y_train)
    NB_Scaled_predicted=NB_classifier_scaled.predict(X_test_scaled)
    NB_Scaled_prob_default=np.sum(NB_Scaled_predicted)/len(NB_Scaled_predicted)
    print("The Default Probability based on Naive Bayes Model(Scaled) is :",'%.3f'%NB_Scaled_prob_default)
    NB_Scaled_accuracy=NB_classifier_scaled.score(X_test_scaled,y_test)
    print("The accuracy of Naive Bayes Model(Scaled) on test set is : ",'%.3f'%NB_Scaled_accuracy)


# In[29]:


#output the result into the existing evaluation dataframe to compare with other models
new_evaluation=pd.DataFrame({'Model':["Naive Bayes_Scaled"],'Default_Probability':[NB_Scaled_prob_default],'Cross_Validation_Accuracy':[NB_Scaled_cross_val_scores.mean()],'Test_Accuracy':[NB_Scaled_accuracy]})
evaluation = evaluation.append(new_evaluation)
evaluation=evaluation[['Model','Default_Probability','Cross_Validation_Accuracy','Test_Accuracy']]
print(evaluation)


# In[30]:


############### MODEL FIVE: VOTING CLASSIFIER WITH NON-SCALED DATA #############
#build a voting classifier based on different models' accuracies
VT_classifier_nonscaled=VotingClassifier(estimators=[('Logistic Regression',LR_classifier),('Random Forest',RF_classifier),('Naive Bayes',NB_classifier)],voting='soft',weights=[LR_accuracy,RF_accuracy,NB_NonScaled_accuracy])
# 10-fold cross validation
VT_NonScaled_cross_val_scores=cross_val_score(VT_classifier_nonscaled,X_train,y_train,cv=10,scoring='accuracy')
print("The 10 fold cross validation score based on Voting Classifier(Non-Scaled) is: %0.3f(+/-%0.3f)" %(VT_NonScaled_cross_val_scores.mean(),VT_NonScaled_cross_val_scores.std()*2))


# In[31]:


if VT_NonScaled_cross_val_scores.mean() >0.97:
    print ("The Voting Classifier (Non Scaled) is overfitting in this case.")
else:
    VT_classifier_nonscaled.fit(X_train,y_train)
    VT_NonScaled_predicted=VT_classifier_nonscaled.predict(X_test)
    VT_NonScaled_prob_default=np.sum(VT_NonScaled_predicted)/len(VT_NonScaled_predicted)
    print("The Default Probability based on Voting Classifier(Non Scaled) is :",'%.3f'%VT_NonScaled_prob_default)
    VT_NonScaled_accuracy=VT_classifier_nonscaled.score(X_test,y_test)
    print("The accuracy of Voting Classifier(Non Scaled) on test set is : ",'%.3f'%VT_NonScaled_accuracy)


# In[32]:


#output the result into the existing evaluation dataframe to compare with other models
new_evaluation=pd.DataFrame({'Model':["Voting Classifier_NonScaled"],'Default_Probability':[VT_NonScaled_prob_default],'Cross_Validation_Accuracy':[VT_NonScaled_cross_val_scores.mean()],'Test_Accuracy':[VT_NonScaled_accuracy]})
evaluation = evaluation.append(new_evaluation)
evaluation=evaluation[['Model','Default_Probability','Cross_Validation_Accuracy','Test_Accuracy']]
print(evaluation)


# In[33]:


################ MODEL SIX: VOTING CLASSIFIER WITH SCALED DATA ##############
#build a voting classifier based on different models' accuracies
VT_classifier_scaled=VotingClassifier(estimators=[('Logistic Regression',LR_classifier),('Random Forest',RF_classifier),('Naive Bayes',NB_classifier_scaled)],voting='soft',weights=[LR_accuracy,RF_accuracy,NB_Scaled_accuracy])
# 10-fold cross validation
VT_Scaled_cross_val_scores=cross_val_score(VT_classifier_scaled,X_train,y_train,cv=10,scoring='accuracy')
print("The 10 fold cross validation score based on Voting Classifier(Scaled) is: %0.3f(+/-%0.3f)" %(VT_Scaled_cross_val_scores.mean(),VT_Scaled_cross_val_scores.std()*2))


# In[34]:


if VT_Scaled_cross_val_scores.mean() >0.97:
    print ("The Voting Classifier (Scaled) is overfitting in this case.")
else:
    VT_classifier_scaled.fit(X_train_scaled,y_train)
    VT_Scaled_predicted=VT_classifier_scaled.predict(X_test_scaled)
    VT_Scaled_prob_default=np.sum(VT_Scaled_predicted)/len(VT_Scaled_predicted)
    print("The Default Probability based on Voting Classifier(Scaled) is :",'%.3f'%VT_Scaled_prob_default)
    VT_Scaled_accuracy=VT_classifier_scaled.score(X_test_scaled,y_test)
    print("The accuracy of Voting Classifier(Scaled) on test set is : ",'%.3f'%VT_Scaled_accuracy)


# In[35]:


#output the result into the existing evaluation dataframe to compare with other models
new_evaluation=pd.DataFrame({'Model':["Voting Classifier_Scaled"],'Default_Probability':[VT_Scaled_prob_default],'Cross_Validation_Accuracy':[VT_Scaled_cross_val_scores.mean()],'Test_Accuracy':[VT_Scaled_accuracy]})
evaluation = evaluation.append(new_evaluation)
evaluation=evaluation[['Model','Default_Probability','Cross_Validation_Accuracy','Test_Accuracy']]
print(evaluation)

