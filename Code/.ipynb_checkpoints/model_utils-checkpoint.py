import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

import statsmodels.api as sm
import statsmodels.formula.api as smf


def cross_val(X, y):
    '''
    input: X(dataframe) of features, y(series) of target
    output: Mean cross validation for simple linear, ridge, and lasso models
    '''
    X, X_test, y, y_test = train_test_split(
        X, y, test_size=.2,
        random_state=10)  #hold out 20% of the data for final testing

    #this helps with the way kf will generate indices below
    X_np, y_np = np.array(X), np.array(y)

    #run the CV
    kf = KFold(n_splits=5, shuffle=True, random_state=71)
#     cv_lm_r2s, cv_lm_reg_ridge_r2s, cv_lm_reg_lasso_r2s, cv_lm_poly_r2s = [], [], [], []  # collect the validation results for the models (R^2)
    cv_lm_r2s_adj, cv_lm_reg_ridge_r2s_adj, cv_lm_reg_lasso_r2s_adj, cv_lm_poly_r2s_adj  = [], [], [], [] # collect the validation results for the models (adjusted R^2) 
    cv_lm_RMSE, cv_lm_reg_ridge_RMSE, cv_lm_reg_lasso_RMSE, cv_lm_poly_RMSE = [], [], [], []
    
    for train_ind, val_ind in kf.split(X_np, y_np):
        X_train, y_train = X_np[train_ind], y_np[train_ind]
        X_val, y_val = X_np[val_ind], y_np[val_ind]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
       
        # SIMPLE LINEAR REGRESSION
        lm = LinearRegression()
        
        ## fitting unscaled features
#         lm.fit(X_train, y_train)
#         cv_lm_r2s.append(lm.score(X_val, y_val))
#         cv_lm_r2s_adj.append(1 - (1-lm.score(X_val, y_val))*(len(y_val)-1)/(len(y_val)-X_val.shape[1]-1))
#         cv_lm_RMSE.append(np.sqrt(mean_squared_error(y_true=y_val, y_pred=lm.predict(X_val))))
        
        ## fitting scaled features
        lm.fit(X_train_scaled, y_train)
#         cv_lm_r2s.append(lm.score(X_val_scaled, y_val))
        cv_lm_r2s_adj.append(1 - (1-lm.score(X_val_scaled, y_val))*(len(y_val)-1)/(len(y_val)-X_val_scaled.shape[1]-1))
        cv_lm_RMSE.append(np.sqrt(mean_squared_error(y_true=y_val, y_pred=lm.predict(X_val_scaled))))
    
        
        # RIDGE REGRESSION
        lm_reg_ridge = RidgeCV()

        ## fitting unscaled features
#         lm_reg_ridge.fit(X_train, y_train)
#         cv_lm_reg_ridge_r2s_.append(lm_reg_ridge.score(X_val, y_val))
#         cv_lm_reg_ridge_r2s_adj.append(1 - (1-lm_reg_ridge.score(X_val, y_val))*(len(y_val)-1)/(len(y_val)-X_val.shape[1]-1))
#         cv_lm_reg_ridge_RMSE.append(np.sqrt(mean_squared_error(y_true=y_val, y_pred=lm_reg_ridge.predict(X_val))))

        ## fitting scaled features
        lm_reg_ridge.fit(X_train_scaled, y_train)
#         cv_lm_reg_ridge_r2s.append(lm_reg_ridge.score(X_val_scaled, y_val))
        cv_lm_reg_ridge_r2s_adj.append(1 - (1-lm_reg_ridge.score(X_val_scaled, y_val))*(len(y_val)-1)/(len(y_val)-X_val_scaled.shape[1]-1))
        cv_lm_reg_ridge_RMSE.append(np.sqrt(mean_squared_error(y_true=y_val, y_pred=lm_reg_ridge.predict(X_val_scaled))))
            
        # LASSO REGRESSION
        lm_reg_lasso = LassoCV()
        
        ## fitting unscaled features
#         lm_reg_lasso.fit(X_train, y_train)
#         cv_lm_reg_lasso_r2s.append(lm_reg_lasso.score(X_val, y_val))  
#         cv_lm_reg_lasso_r2s_adj.append(1 - (1-lm_reg_lasso.score(X_val, y_val))*(len(y_val)-1)/(len(y_val)-X_val.shape[1]-1))
#         cv_lm_reg_lasso_RMSE.append(np.sqrt(mean_squared_error(y_true=y_val, y_pred=lm_reg_lasso.predict(X_val))))

        ## fitting scaled features
        lm_reg_lasso.fit(X_train_scaled, y_train)
#         cv_lm_reg_lasso_r2s.append(lm_reg_lasso.score(X_val_scaled, y_val))
        cv_lm_reg_lasso_r2s_adj.append(1 - (1-lm_reg_lasso.score(X_val_scaled, y_val))*(len(y_val)-1)/(len(y_val)-X_val_scaled.shape[1]-1))
        cv_lm_reg_lasso_RMSE.append(np.sqrt(mean_squared_error(y_true=y_val, y_pred=lm_reg_lasso.predict(X_val_scaled))))
    
        # POLYNOMIAL REGRESSION
        poly = PolynomialFeatures(degree=2)
        
        X_train_poly = poly.fit_transform(X_train)
        X_val_poly = poly.transform(X_val)
        
        ## fitting unscaled features
        lm_poly = LinearRegression()
        
        lm_poly.fit(X_train_poly, y_train)
#         cv_lm_poly_r2s.append(lm_poly.score(X_val_poly, y_val))
        cv_lm_poly_r2s_adj.append(1 - (1-lm_poly.score(X_val_poly, y_val))*(len(y_val)-1)/(len(y_val)-X_val_poly.shape[1]-1))
        cv_lm_poly_RMSE.append(np.sqrt(mean_squared_error(y_true=y_val, y_pred=lm_poly.predict(X_val_poly))))
        
        
#     print('Simple linear regression scores: ', cv_lm_r2s)
#     print('Simple linear regression scores (adjusted): ', cv_lm_r2s_adj)
#     print('Ridge scores: ', cv_lm_reg_ridge_r2s, '\n')
#     print('Ridge scores (adjusted): ', cv_lm_reg_ridge_r2s_adj)
#     print('Lasso scores: ', cv_lm_reg_lasso_r2s, '\n')
#     print('Lasso scores (adjusted): ', cv_lm_reg_lasso_r2s_adj)
#     print('Polynomical regression scores: ',cv_lm_poly_r2s)
#     print('Polynomical regression scores (adjusted): ',cv_lm_poly_r2s_adj, '\n')

#     print(f'Simple linear mean cv r^2: {np.mean(cv_lm_r2s):.4f} +- {np.std(cv_lm_r2s):.4f}')
    print(f'Simple linear mean cv adjusted r^2: {np.mean(cv_lm_r2s_adj):.4f} +- {np.std(cv_lm_r2s_adj):.4f}')                            
#     print(f'Ridge mean cv r^2: {np.mean(cv_lm_reg_ridge_r2s):.4f} +- {np.std(cv_lm_reg_ridge_r2s):.4f}')
    print(f'Ridge mean cv adjusted r^2: {np.mean(cv_lm_reg_ridge_r2s_adj):.4f} +- {np.std(cv_lm_reg_ridge_r2s_adj):.4f}')
#     print(f'Lasso mean cv r^2: {np.mean(cv_lm_reg_lasso_r2s):.4f} +- {np.std(cv_lm_reg_lasso_r2s):.4f}')
    print(f'Lasso mean cv adjusted r^2: {np.mean(cv_lm_reg_lasso_r2s_adj):.4f} +- {np.std(cv_lm_reg_lasso_r2s_adj):.4f}')
#     print(f'Polynomial mean cv r^2: {np.mean(cv_lm_poly_r2s):.4f} +- {np.std(cv_lm_poly_r2s):.4f}')  
    print(f'Polynomial mean cv adjusted r^2: {np.mean(cv_lm_poly_r2s_adj):.4f} +- {np.std(cv_lm_poly_r2s_adj):.4f}', '\n') 
 
    print(f'Simple linear mean cv RMSE: {np.mean(cv_lm_RMSE):.4f} +- {np.std(cv_lm_RMSE):.4f}')                                                     
    print(f'Ridge mean cv RMSE: {np.mean(cv_lm_reg_ridge_RMSE):.4f} +- {np.std(cv_lm_reg_ridge_RMSE):.4f}')                            
    print(f'Lasso mean cv RMSE: {np.mean(cv_lm_reg_lasso_RMSE):.4f} +- {np.std(cv_lm_reg_lasso_RMSE):.4f}') 
    print(f'Polynomial mean cv RMSE: {np.mean(cv_lm_poly_RMSE):.4f} +- {np.std(cv_lm_poly_RMSE):.4f}', '\n')  
    
    print(f'Ridge alpha: {lm_reg_ridge.alpha_}')
    print(f'Lasso alpha: {lm_reg_lasso.alpha_}')
    
    Lasso_coef = pd.DataFrame(lm_reg_lasso.coef_, index=X.columns, columns=['Lasso Coef'])
    return Lasso_coef

def correlation_heatmap(df):
    '''
    input: a dataframe
    output: a correlation heatmap of all the features and target
    '''
    plt.figure(figsize=(15,12))
    sns.heatmap(df.corr(), cmap='viridis', annot=True)

def distribution_plot(df, column_name):
    '''
    input: a dataframe and the name of a column to create a distribution plot for
    output: a distribution plot
    '''
    sns.displot(df[column_name], bins=35)
    
def logtransform(df, columnlist):
    '''
    input: a dataframe and a list of features
    output: new columns with log of feature
    '''
    for elem in columnlist:
        df[f'log{elem}'] = df[elem].apply(np.log1p)  #log(x+1) due to zeros in features

    return df

def scatterplot(x, y, df):
    '''
    input: a dataframe and two features to create a scatterplot for
    output: a scatter plot
    '''
    sns.scatterplot(x=x, y=y, data=df)
    
def residuals(X, y, df):
    '''
    input: A dataframe of features and a series containing target values
    output: A dataframe with Predictions and Residuals columns added, sorted by Residuals in a descending order
    '''
    lr = LinearRegression()
    fit = lr.fit(X, y)

    df['Predictions'] = fit.predict(X)
    df['Residuals'] = abs(y - df['Predictions'])

    return df.sort_values('Residuals', ascending=False)

