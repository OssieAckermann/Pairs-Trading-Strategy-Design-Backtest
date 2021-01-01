#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 16:20:10 2020

@author: Austin
"""

import warnings
warnings.filterwarnings('ignore')
from colorit import *
init_colorit()

import numpy as np
import pandas as pd


import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

class LinearRegression():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.addconst()
        self.fit()
        
    def addconst(self):
        self.x['constant'] = np.ones(len(self.x))
        
    def fit(self):
        self.coef = np.linalg.inv(self.x.T.dot(self.x)).dot(self.x.T.dot(self.y))
        
        
        self.error = self.y.values - self.x.dot(self.coef).values
        self.residual_cov = self.error.T.dot(self.error)/ len(self.y)
        self.coef_cov = np.kron(np.linalg.inv(self.x.T.dot(self.x))
                                ,self.residual_cov )
        self.coef_std = np.sqrt(self.y.shape[0]/(self.y.shape[0]-self.x.shape[1])*np.diag(self.coef_cov))
        self.tstats = self.coef/self.coef_std.reshape(self.coef.shape)
        
    def report(self):
        self.coef = pd.DataFrame(self.coef, 
                                 index = self.x.columns, 
                                 columns = self.y.columns)
        self.coef_std = pd.DataFrame(self.coef_std, 
                                     index = self.x.columns, 
                                     columns = self.y.columns)
        self.tstats = pd.DataFrame(self.tstats, 
                                   index = self.x.columns, 
                                   columns = self.y.columns)
        return pd.concat({'Estimate Coefficient':self.coef,
                          'SD of Estimate':self.coef_std,
                          't-Statistic':self.tstats,}, axis = 1)
    
class VAR():
    """
    This is class contains all the 
    statistical means I need for this
    pair trading project
    """
    def __init__(self, df, lag, ):
        """
        x and y are pandas dataframe
        """
        self.df = df
        self.lag =lag
        self.coef = np.array([])
        self.columns_name = self.name_lag()
        self.data = self.process_data()
        self.x = self.data[self.indepvar_name]
        self.y = self.data[df.columns]
        self.addconst()
        self.fit()
        
        
    def name_lag(self,):
        rlst = list(self.df.columns)
        self.indepvar_name = []
        for j in range(self.lag):
            for i in range(len(self.df.columns)):
                rlst.append('(Lag_'+str(j+1)+', '+self.df.columns[i]+')')
                self.indepvar_name.append('(Lag_'+str(j+1)+', '+self.df.columns[i]+')')
        return rlst
        
    def process_data(self,):
        lst = [self.df]
        for i in range(self.lag):
            lst.append(df.shift(i+1))
        r_df = pd.concat(lst,axis = 1)
        r_df.columns = self.columns_name
        return r_df.fillna(0).reset_index(drop=True)
    
    def addconst(self):
        self.x['constant'] = np.ones(len(self.x))
        self.indepvar_name.append('constant')

    def fit(self):
        self.coef = np.linalg.inv(self.x.T.dot(self.x)).dot(self.x.T.dot(self.y))
        self.coef = pd.DataFrame(self.coef, 
                                 index = self.x.columns,
                                columns = self.y.columns)
        self.error = self.y - self.x.dot(self.coef)
        self.residual_cov = self.error.T.dot(self.error)/ len(self.y)
        self.coef_cov = np.kron(np.linalg.inv(self.x.T.dot(self.x))
                                ,self.residual_cov )
        self.coef_std = np.sqrt(self.y.shape[0]/(self.y.shape[0]-self.x.shape[1])*np.diag(self.coef_cov))
        self.tstats = self.coef/self.coef_std.reshape(self.coef.shape)
        
        self.coef_std = pd.DataFrame(self.coef_std.reshape(self.coef.shape),
                                    index =self.x.columns,
                                   columns = self.y.columns)
        
        self.p_value = pd.DataFrame(2*(1-norm.cdf(abs(self.tstats))), 
                     index = self.indepvar_name, 
                     columns = self.df.columns)
        
    
    def report(self):
        return pd.concat({'Estimate Coefficient':self.coef,
                          'SD of Estimate':self.coef_std,
                          't-Statistic':self.tstats,}, axis = 1)
        
    def AIC(self):
        return np.log(np.linalg.det(self.residual_cov)) + 2*self.x.shape[1]*self.y.shape[1]/self.y.shape[0]
    
    def BIC(self):
        return np.log(np.linalg.det(self.residual_cov)) + np.log(self.y.shape[0])*self.x.shape[1]*self.y.shape[1]/self.y.shape[0]
        
    def IC(self, lag):
        ic = pd.DataFrame([[VAR(df, p+1).AIC(), VAR(df, p+1).BIC()] for p in range(lag)], 
             index=[p+1 for p in range(lag)],
             columns = ['AIC','BIC'],)
        ic.index.name = 'Lag'
        return ic
    
    def stability(self):
        eig_v,_ = np.linalg.eig(self.coef.drop('constant'))
        if False in (eig_v < 1):
            print('Stability status: '+ color('[x] UNSTABLE',Colors.red))
        else:
            print('Stability status: ' + color('[o] STABLE',(9, 86, 146)))
        
