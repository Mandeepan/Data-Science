#Credit Approval Model for Banks

The following proposal is for my 1st capstone project for Springboard Data Science course.
 
Mandy Pan
January, 2018

1.Problem Definition
This research aims to detect potential default by considering both credit card customers’ personal information and macroeconomic environment. It will help banks reduce risk exposure and avoid losses of offering credit card to untrustworthy customers.

2. Data Source
This project will test on the modified dataset, which combines an existing dataset from UCI Machine Learning Repository and several extra macroeconomic indicators from TradingEconomics. The existing dataset contains the credit amount for the customer’s credit card, gender, age, education, marital status, historical payments and default payments during April and September 2005. The addition component includes the Consumer Price Index, Consumer Confidence Index and the Interest Rate (by central bank) etc. The average of those indicators between April and September in 2005.  The sample size is 30000, and 70% of the data will be used for training, 15% for validation and 15% for model testing. 

3. Major Steps (To be changed if necessary)
3.1 Data Pre-processing and Exploratory Data Analysis 
The existing 30000 observations between April 2005 and September 2005 and doesn’t include any indications of specific timing for customers receiving their credit cards. However, the macroeconomic indicators are time-variated. Hence, given the 6 monthly data points of each indicator, the program will randomly pick 1 data point. Although with random simulation, the final data result may not reflect the true fact. The macrocosmic data still plays an important role in affecting the creditability of individuals.

3.2 Training methods
Three major methods will be used, Logistic Regression Model, Decision Tree methods and Random Forest Classifier

4.Validation Method
To avoid wasting data, the research will apply K-fold cross validation.  

5.Interpretation
With three methods, the testing data set should predict three different results. If a customer is predicted to be default by at least two methods, that customer’s credit card will be cancelled by the bank.

6. Deliverables
This research will come out with a modified dataset and a paper 
