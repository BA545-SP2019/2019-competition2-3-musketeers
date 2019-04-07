# Charles F. Dolan School of Business

# BA 545 – Data Mining, Competition#2, SP 2019

## 3 musketeers

**Goal:** 
The goal for this part of the project was to  perform Exploratory Data Analysis (EDA) and pre-process the data as best we could.We also want some initial basic models to show that our pre-processing was successful.


**Explanation of Work:**

#  

The Team began the project by examining each of the attributes and getting an understanding from the website . [Link to website](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients).


We then created our Data Dictionary: **![](/Data_Dictionary_picture.JPG)**

### EDA

 - Firstly, we read the csv file and split out the X and Y data
 - Checked for missing values(none)
 - We observed that the Y data is imbalanced, we will deal with this later
 - We investigated some unusual values, recoded and/or renamed them in the MARRIAGE, SEX and EDUCATION columns
 - Got logical ("dummy") variables for varaibles
 - Created box-plots of the columns
 - We observed -2 values in the "PAY" columns, -2 is not listed in the data dictionary. We were unsure what best way to deal with the -2 values so we performed a quick model evaulation (using Naieve Bayes and a logistic regression) of different methods to test their significant.
     -  we left them as is(-2), we imputed them using the column mode, we changed them to -1(assumed that all negative values meant the customer payment on time), and we combined them with 0 (which is also not described in the data dictionary) creating a "other" category
     - The performance of all 4 methods were almost identical, therefore it was decided that the -2 values were not siginificant in our models performance. We also note that the -2 values are only appromimately 7% of the data
 - Plotted a PCA analyisis of the data
 - 