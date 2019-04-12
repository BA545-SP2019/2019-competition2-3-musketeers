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
    
    
    
def evaluate_baseline(df, NB = True):
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

    X = df.drop('Y', axis = 1)
    y = df['Y']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 2019)

    #begin oversampling
    oversample = pd.concat([X_train,y_train],axis=1)
    max_size = oversample['Y'].value_counts().max()
    lst = [oversample]

    for class_index, group in oversample.groupby('Y'):
        lst.append(group.sample(max_size-len(group), replace=True))
    X_train = pd.concat(lst)
    y_train=pd.DataFrame.copy(X_train['Y'])
    del X_train['Y']

    if NB:
        clf = GaussianNB()
    else:
        clf = LogisticRegression()

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    
    #10-fold cross validation
    kfold = KFold(n_splits=10, random_state=2019)
    results = cross_val_score(clf, X_train, y_train, cv=kfold, scoring='f1')
    print(results)
    print()
    print('corss-validation f1 score:', np.mean(results))
    #results_auc = cross_val_score(clf, X_train, y_train, cv=kfold, scoring='roc_auc')
    #print()
    #print('cross-validation auc score:', np.mean(results_auc))