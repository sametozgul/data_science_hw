import pandas as pd
import numpy as np

def problem1(fileName):
    data=pd.read_excel(fileName)
# data.head()
# data.info()
# data.shape
# data.isnull().values.any()
# data.describe()

    X = data.drop(['Y1', 'Y2'], axis=1).values
    y_1= ((data['Y1']).values).reshape(-1, 1)
    y_2= ((data['Y2']).values).reshape(-1, 1)

    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split,GridSearchCV
    outputs=[y_1,y_2]
    for y_x in outputs:
        X_train , X_test , y_train , y_test = train_test_split(X , y_x , test_size=0.2 , random_state =42)
        from sklearn.preprocessing import StandardScaler
        scaler=StandardScaler()
        X_train=scaler.fit_transform(X_train)
        X_test=scaler.transform(X_test)
        from sklearn.model_selection import RepeatedKFold,cross_val_score
        cv=RepeatedKFold(n_splits=10,n_repeats=10,random_state=True)
        from numpy import mean
        from numpy import std
        params=[0.001,0.01,0.1, 1.0, 10.0]
        for alpha in params:
            model=Ridge(alpha=alpha)
            model.fit(X_train,y_train)
            # print(model.score(X_train,y_train))

        model=Ridge(alpha=0.001)
        model.fit(X_train,y_train)
        scores = cross_val_score(model, X_train, y_train,scoring="neg_mean_absolute_error", cv=cv)
        print(-scores.mean(),scores.std())

    

problem1("ENB2012_data.xlsx")
