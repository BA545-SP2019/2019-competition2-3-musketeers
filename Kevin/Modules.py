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