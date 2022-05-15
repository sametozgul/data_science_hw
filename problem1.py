import pandas as pd
import numpy as np

def problem1(fileName):
    data=pd.read_excel(fileName)
    X = data.drop(['Y1', 'Y2'], axis=1).values
    y_1= ((data['Y1']).values).reshape(-1, 1)
    y_2= ((data['Y2']).values).reshape(-1, 1)
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split,GridSearchCV
    from sklearn.ensemble import RandomForestRegressor
    param_grid = {
        'max_depth': [50,150,250],
        'min_samples_leaf': [1,2,3],
        'min_samples_split': [2,3],
        'n_estimators': [10,50,100,250,500]
    }
    ridge_params={
        'alpha':[0.001,0.01,0.1, 1.0, 10.0]
    }
    outputs=[y_1,y_2]
    ridge_means=[]
    ridge_std=[]
    forest_means=[]
    forest_std=[]
    for index,y_x in  enumerate(outputs):
        for score in ["neg_mean_absolute_error","neg_mean_squared_error"]:
            X_train , X_test , y_train , y_test = train_test_split(X , y_x , test_size=0.2 , random_state =42)
            from sklearn.preprocessing import StandardScaler
            scaler=StandardScaler()
            X_train=scaler.fit_transform(X_train)
            X_test=scaler.transform(X_test)
            from sklearn.model_selection import RepeatedKFold,cross_val_score
            cv=RepeatedKFold(n_splits=10,n_repeats=10,random_state=True)
            temp_ridge=Ridge()
            grid_search_ridge=GridSearchCV(estimator = temp_ridge, param_grid = ridge_params, cv = cv, n_jobs = -1)
            grid_search_ridge.fit(X_train,y_train.ravel())
            model=Ridge(**grid_search_ridge.best_params_)
            model.fit(X_train,y_train)
            scores_ridge = cross_val_score(model, X_train, y_train,scoring=score, cv=cv)
            ridge_means.append(-scores_ridge.mean())
            ridge_std.append(scores_ridge.std())
            # print("Ridge","Y",index+1," ",score,"-----",-scores.mean(),scores.std())
            rf = RandomForestRegressor()
            grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = cv, n_jobs = -1)
            grid_search.fit(X_train,y_train.ravel())
            last_model=RandomForestRegressor(**grid_search.best_params_)
            scores_randomforest= cross_val_score(last_model, X_train, y_train.ravel(),scoring=score, cv=cv)
            forest_means.append(-scores_randomforest.mean())
            forest_std.append(scores_randomforest.std())
            # print("RandomForest","Y",index+1," ",score,"-----",-scores.mean(),scores.std())
    from prettytable import PrettyTable
    x=PrettyTable(["-","Mean Absolute Error","*","Mean Square Error","/"])
    x.add_row(["Output","RandomForest","RidgeRegression","RandomForest","RidgeRegression"])
    x.add_row(["Y1",str(forest_means[0])+u"\u00B1"+str(forest_std[0]),str(ridge_means[0])+u"\u00B1"+str(ridge_std[0]),
                    str(forest_means[1])+u"\u00B1"+str(forest_std[1]),str(ridge_means[1])+u"\u00B1"+str(ridge_std[1])])
    x.add_row(["Y2",str(forest_means[2])+u"\u00B1"+str(forest_std[2]),str(ridge_means[2])+u"\u00B1"+str(ridge_std[2]),
                    str(forest_means[3])+u"\u00B1"+str(forest_std[3]),str(ridge_means[3])+u"\u00B1"+str(ridge_std[3])])
    print(x)
problem1("ENB2012_data.xlsx")
