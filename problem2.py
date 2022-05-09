import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
def problem2():
    bank = pd.read_csv('bank-additional-full.csv', sep = ';')
    #Converting dependent variable categorical to dummy
    y = pd.get_dummies(bank['y'], columns = ['y'], prefix = ['y'], drop_first = True)
    y=y.values.reshape(-1, 1)
    bank_client = bank.iloc[: , 0:7]
    labelencoder_X = LabelEncoder()
    bank_client['job']      = labelencoder_X.fit_transform(bank_client['job']) 
    bank_client['marital']  = labelencoder_X.fit_transform(bank_client['marital']) 
    bank_client['education']= labelencoder_X.fit_transform(bank_client['education']) 
    bank_client['default']  = labelencoder_X.fit_transform(bank_client['default']) 
    bank_client['housing']  = labelencoder_X.fit_transform(bank_client['housing']) 
    bank_client['loan']     = labelencoder_X.fit_transform(bank_client['loan'])

    bank_client.loc[bank_client['age'] <= 32, 'age'] = 1
    bank_client.loc[(bank_client['age'] > 32) & (bank_client['age'] <= 47), 'age'] = 2
    bank_client.loc[(bank_client['age'] > 47) & (bank_client['age'] <= 70), 'age'] = 3
    bank_client.loc[(bank_client['age'] > 70) & (bank_client['age'] <= 98), 'age'] = 4

    bank_related = bank.iloc[: , 7:11]
    bank[(bank['duration'] == 0)]

    labelencoder_X = LabelEncoder()
    bank_related['contact']     = labelencoder_X.fit_transform(bank_related['contact']) 
    bank_related['month']       = labelencoder_X.fit_transform(bank_related['month']) 
    bank_related['day_of_week'] = labelencoder_X.fit_transform(bank_related['day_of_week']) 

    bank_related.loc[bank_related['duration'] <= 102, 'duration'] = 1
    bank_related.loc[(bank_related['duration'] > 102) & (bank_related['duration'] <= 180)  , 'duration']    = 2
    bank_related.loc[(bank_related['duration'] > 180) & (bank_related['duration'] <= 319)  , 'duration']   = 3
    bank_related.loc[(bank_related['duration'] > 319) & (bank_related['duration'] <= 644.5), 'duration'] = 4
    bank_related.loc[bank_related['duration']  > 644.5, 'duration'] = 5

    bank_se = bank.loc[: , ['emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']]
    bank_o = bank.loc[: , ['campaign', 'pdays','previous', 'poutcome']]
    bank_o['poutcome'].replace(['nonexistent', 'failure', 'success'], [1,2,3], inplace  = True)
    bank_final= pd.concat([bank_client, bank_related, bank_se, bank_o], axis = 1)
    bank_final = bank_final[['age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
                        'contact', 'month', 'day_of_week', 'duration', 'emp.var.rate', 'cons.price.idx', 
                        'cons.conf.idx', 'euribor3m', 'nr.employed', 'campaign', 'pdays', 'previous', 'poutcome']]
    
    from sklearn.model_selection import train_test_split
    X_train , X_test, y_train, y_test = train_test_split(bank_final, y.ravel(), test_size = 0.2, random_state = 42)

    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    from sklearn.model_selection import RepeatedKFold
    from sklearn.model_selection import cross_val_score
    cv=RepeatedKFold(n_splits=5,n_repeats=5,random_state=True)

    from sklearn.linear_model import LogisticRegression
    all_scores=[]
    for x in np.logspace(-4,4,20):
        model=LogisticRegression(C=x)
        model.fit(X_train,y_train)
        scores=cross_val_score(model, X_train, y_train.ravel(),scoring="roc_auc", cv=cv)
        all_scores.append(scores.mean())

    plt.plot(np.logspace(-4,4,20,endpoint=True),all_scores,'-gD')
    plt.xscale("log")
    plt.show()

problem2()