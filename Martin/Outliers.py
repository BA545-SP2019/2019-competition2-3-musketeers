def OutliersUpperBand(df):
    '''Calcualtes the IQR for each column then replaces any values outside that with the upper or lower band'''
    
    import pandas as pd
    import numpy as np
    
    for i in df.describe().columns:
        Q1=df.describe().at['25%',i]
        Q3=df.describe().at['75%',i]
        IQR=Q3 - Q1
        LTV=Q1 - 1.5 * IQR
        UTV=Q3 + 1.5 * IQR
        x=np.array(df[i])
        p=[]
        for j in x:
            if j < LTV:
                p.append(LTV)
            elif j > UTV:
                p.append(UTV)
            else:
                p.append(j)
        df[i]=p
    return df


def OutliersMean(df):
    '''Calcualtes the IQR for each column then replaces any values outside that with the mean'''
    
    import pandas as pd
    import numpy as np
    
    for i in df.describe().columns:
        Q1=df.describe().at['25%',i]
        Q3=df.describe().at['75%',i]
        IQR=Q3 - Q1
        LTV=Q1 - 1.5 * IQR
        UTV=Q3 + 1.5 * IQR
        x=np.array(df[i])
        p=[]
        for j in x:
            if j < LTV:
                p.append(df[i].mean())
            elif j > UTV:
                p.append(df[i].mean())
            else:
                p.append(j)
        df[i]=p
    return df

def OutliersMedian(df):
    '''Calcualtes the IQR for each column then replaces any values outside that with the median'''
    
    import pandas as pd
    import numpy as np
    
    for i in df.describe().columns:
        Q1=df.describe().at['25%',i]
        Q3=df.describe().at['75%',i]
        IQR=Q3 - Q1
        LTV=Q1 - 1.5 * IQR
        UTV=Q3 + 1.5 * IQR
        x=np.array(df[i])
        p=[]
        for j in x:
            if j < LTV:
                p.append(df[i].median())
            elif j > UTV:
                p.append(df[i].median())
            else:
                p.append(j)
        df[i]=p
    return df