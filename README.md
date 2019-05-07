# Competition 2 | 3 Musketeers
#### BA545 | Spring 2019 
#### Kevin Hickey | Martin Browne | Stuart Weinstein

**Analytical Question & Goal:**

What is the likelihood that a credit card client will default and ultimately force the closure of their account?  We were tasked with predictiing the likelihood of this worst case outcome for each ID in the test set.  To do this we must predict the binary value of the default variable.



**Files in the repository:**

- The Final Project notebook explains our work process and understanding: [Final_project](Final_project.ipynb)


- The Data Dictionary: **![](Data_Dictionary_picture.JPG)**
*Which we updated after creating our engineered features link to complete data dictionary*:  **[Data Dictionary](complete_data_dic.xlsx)** 


 - The Modules PY file: created custom definitions to read the data file, rename the columns, generate charts, evaluate the baseline model, and an XGBoost model for evaluation.  These modules mean the code is reproduceable in each model or pipeline we attempt.  **[Modules](Modules.py)** 
 
 - Intial Model report submission **[Initial_models](Initial_models.ipynb)** 
 - Data Audit report submission **[data_audit_report](data_audit_report.ipynb)**
 
**Project Scope:**
This project is strictly limited to identifying existing client's who are at risk of default based strictly on the data provided from the issuer.

**Business Understanding:**
Our client needs to evaluate their customer base to predict which accounts they will need to close due to default.

**Source Data:**
The data we were given was also found in the UCI Machine Learning Repository. It is information of customer payment history from a Taiwanese based company seeking to accurately predict the probability of customer defualt. The dataset is comprised of customer demographics, such as age, education, gender, as well as payment amounts and payment history
for the company on a 6 period basis. Here are the components of the data that was given to us:
 
    - Credit Limit
    - Gender 
    - Level of Education 
    - Marital Status 
    - Age 
    - 6 Scored Repayment Status going back 9 months based on the delays for 6 sequential statement periods 
    - Billing Statements for 6 sequential statement periods
    - Statement Amount for 6 sequential statement periods 
    
Here are the names of the fields from the data source from the source: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients


**EDA:** 
Summary EDA is provided and also shown in the final project notebook [Final_project](Final_project.ipynb)


-	Created box-plots of the columns
-	Observed -2 values in the "PAY" columns which was not listed in the data dictionary. Performed a quick model evaulation (using Naieve Bayes and a logistic regression) of different methods to test the significant.
-	Left as is(-2), and imputed using the column mode. Changed value to -1(assumed that all negative values meant the customer payment on time), and combined with 0 values (which is also not described in the data dictionary) creating a "other" category
-	The performance of all 4 methods were almost identical, therefore it was decided that the -2 values were not siginificant in our models performance. We also note that the -2 values are only appromimately 7% of the data
-	Plotted a correlation matrix
-	Plotted a PCA analyisis of the data


**Data Preparation:** 
The first thing we did was to read in the data into a suitable format. This involved reading the Excel file and naming the columns into something more intuitive than the original column names.

There are no missing data points, so **imputation was not necessary**.

Next, after noting the discrepencies in the values for the classes for  'EDUCATION', 'SEX', and 'MARRIAGE' features, we renamed and replaced several values. For instance, the 'EDUCATION' feature had a majority of  "Graduate School" and "University" class, and a very limited amount of the other classes, so we dicided to keep the 2 former classes and bin the latter classes into a category of "Other". A similar process was done for the other two features

We noted, undefined values in some of the features. -2 and "0" appear frequently in the "pay_" features. We tested a few strategies to see if adjusting this values effected our inital results. We eventually decided to bin together these two values into a new logical feature, which is used in the final dataset, as shown later in the presentation

**Feature Engineering:**
After discussions with Dr. Tao we engineered ratio features to help with the correlation between some of the features we observed earlier. Some of the ratios are time weighted (giving more significance to the most recent periods). Further explanation is provided in the [Final_project](Final_project.ipynb)

**A note about correlation:**
Many of our initial correlated features were engineered into new features were decorrelated. However, in the creation of new features, we created newly correlated features. Our newly created 'PERCENT_OF_LIMIT_BAL' featues are very highly correlated with each other, and our binned logical variables are also very correlated with each other. With this new information, it appears that we can eliminate some of these new feautures. Since this new information would cause us to change our procedure below and would take considerable time to revise, we won't change the results from below onward. We acknolwedge our mistake of missing this crucial step; perhaps in the future we can return to this to correct and reapply our steps below with a better version of our data. Still, we would like to continue our process below. 


**Modeling:**
The results of our models are shown in the [Final_project](Final_project.ipynb)

The best performing model was the XGBoost model created by Kevin. 


Here is the notebook that holds the code used to create the final best performing model.  [Final_model_process.ipynb](Kevin/Final_model_process.ipynb) We will go through this notebook now. 

**Evaluation:**
After developing our best performing model, we decided to utilize a range of feature selection methods to try and improve our results. The processes and methods to do so are described in [link](Kevin/Feature_selection.ipynb) notebook. 

**Conclusion:**
To conclude, our best model was an xgboost model with  hyperparameters tuned to the values as in 'xgboost3'. By implementing various library's we were able to see that the features that contribute most to prediction of default are a client's delayed payments for more than one month.  We suggest that managers carefully track and tag any potential delay in payment of a client as early as possible.