def read_data(path):
    
    import pandas as pd
    
    df = pd.read_excel(path, header=0)

    df.columns = df.iloc[0,:]

    df.rename(index=str, columns={"default payment next month": "Y"}, inplace=True)

    y = df.Y

    df.drop('ID', inplace=True, axis = 0)
    y.drop('ID', inplace=True, axis = 0)
    #df.drop(columns = 'Y', inplace = True)
    df = df.astype('int32')
    y = y.astype('int32')
    
    #replace 'PAY_0' with 'PAY_1' in order to allign with 'BILL_AMT1' and 'PAY_AMT1'
    df.rename(index=str, columns = {'PAY_0': 'PAY_1'}, inplace=True)

    return df,  y

def proc_cat_df(df):
    
    import pandas as pd
    
    #change the 'EDUCATION' values
    df['EDUCATION'] = df['EDUCATION'].replace({1: 'Graduate School', 2: 'University', 3: 'Other', 4: 'Other', 5: 'Other', 6: 'Other', 0: 'Other'})

    #change the 'SEX' values
    df['SEX'] = df['SEX'].replace({1: 'Male', 2:'Female'})

    #change the 'MARRIAGE' values
    df['MARRIAGE'] = df['MARRIAGE'].replace({2: 'Non-married', 1: 'Married', 3: 'Non-married', 0: 'Non-married'})
    
    df = pd.get_dummies(df)
    
    return df


def make_bar_charts(df, features, n_rows, n_cols):
    import pandas 
    import matplotlib.pyplot as plt
    
    fig = plt.figure()
    data = df
    for idx, variable in enumerate(features):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1)
        data[variable].value_counts().plot(kind='bar', ax=ax)
        ax.set_title(variable)
    plt.tight_layout()
    plt.show()
        
def make_hist(df, features, n_rows, n_cols, n_bins):
    import pandas 
    import matplotlib.pyplot as plt
    
    fig = plt.figure()
    data = df
    for idx, variable in enumerate(features):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1)
        data[variable].hist(bins=n_bins)
        ax.set_title(variable)
    plt.tight_layout()
    plt.show() 
    
    
    
def evaluate_baseline(df, clf):
    '''evalueates a df with a Guassian Naive Bayes Model.
        Input: a dataframe
               NB = True is want to run a simple Naive Bayes model
               NB= False if want to run a simple Logistic Regression Model
        Output: 10-fold cross validation f1 score and AUC score.
    
    '''
    import pandas as pd
    import numpy as np
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, KFold, cross_val_score
    from sklearn.metrics import classification_report, confusion_matrix
    from imblearn.pipeline import make_pipeline, Pipeline
    from imblearn.over_sampling import RandomOverSampler, SMOTE

    X = df.drop('Y', axis = 1)
    y = df['Y']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 2019)

    #begin oversampling pipeline
    if clf == 'NB':
        clf = GaussianNB()
    else:
        clf = LogisticRegression()
    
    oversampler = SMOTE(random_state = 2019)
    pipeline = make_pipeline(oversampler, clf)
    
    pipeline.fit(X_train, y_train)
    
    #10-fold cross validation
    kfold = KFold(n_splits=10, random_state=2019)
    results = cross_val_score(pipeline, X_test, y_test, cv=kfold, scoring='f1')
    print('10-fold f1 scores:')
    print(results)
    print()
    print('corss-validation f1 score:', np.mean(results))
    
    
def XGBoost_evaluate(df):
    '''baseline xgboos classifier for investigating creating new sequential data.
    Input: a dataframe with desired features. Target feature must be labled as "Y".
    Ouput: 50-fold cross validation with auc and mean-avg-precision scores.
    The process is a follows:
    #1: split data into a train & test split
    #2: create SMOTE-applied data training sets
    #3: create an XGBoost model
    #4. train on SMOTE-applied data
    #5. 10-fold Cross-validation f1 scores
    '''
    import xgboost as xgb
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split, KFold, cross_val_score
    import sklearn.metrics
    from imblearn.over_sampling import SMOTE

    #1
    X = df.drop(columns = ['Y'])
    y = df['Y']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2019)

    #2
    oversampler = SMOTE(random_state = 2019)
    X_train_oversampled, y_train_oversampled = oversampler.fit_resample(X_train, y_train)

    X_train_oversampled = pd.DataFrame(X_train_oversampled, columns = X_train.columns)
    y_train_oversampled = pd.Series(y_train_oversampled)

    X_train_dmatrix = xgb.DMatrix(X_train_oversampled)
    y_train_dmatrix = xgb.DMatrix(y_train_oversampled)

    #3
    xgb_clf = xgb.XGBClassifier(max_depth=5, n_estimators=100, colsample_bytree=0.3, learning_rate=0.1, n_jobs=-1)

    #4
    xgb_clf.fit(X_train_oversampled, y_train_oversampled)

    #5
    kfold = KFold(n_splits=10, random_state=2019)
    results = cross_val_score(xgb_clf, X_test, y_test, cv=kfold, scoring = 'f1')

    print('10-fold f1 scores:')
    print(results)
    print()
    print('corss-validation f1 score:', np.mean(results))

def plot_auc_map(cv_results):
    """plots the auc-score and map-score from the XGBoost_evaluate function
    Input: cross_validation results from XGBoost_evaluate function
    """
    cv_results[['test-auc-mean', 'test-map-mean']].plot()
    plt.xlabel('Round')
    plt.ylabel('Score')
    plt.annotate(xy = (30,.75), s='final auc-mean score: ' +str(cv_results['test-auc-mean'][49]))