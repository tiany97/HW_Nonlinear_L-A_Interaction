# -*- coding: utf-8 -*-
"""
Created on Fri May 08 16:36:36 2020

@author: Tian Yinglin
"""

print ('start')
import shap
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2 
from sklearn.datasets import make_regression
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression
import matplotlib.pylab as pl
import numpy as np
import pandas as pd
import xarray as xr
from scipy import signal
import copy
from netCDF4 import Dataset, num2date, date2num
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import matplotlib.patches as patches
import cartopy
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import cmaps
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib import cm
from scipy.optimize import curve_fit


from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from sklearn.model_selection import train_test_split, cross_val_score
from scipy.stats import linregress
# import ruptures as rpt
import random

import scipy
import scipy.stats as ss
import scipy.signal
from scipy.interpolate import griddata
from scipy.signal import argrelmin
from scipy.ndimage import interpolation
from scipy.stats import genextreme as gev
import scipy.stats.mstats as mstats
import math
from math import log10, floor
import os
import dask.diagnostics as dd
import dask.array as da
from dask.diagnostics import ProgressBar
pbar = ProgressBar() 
import seaborn as sns
from sklearn import linear_model
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, make_scorer

from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

import xgboost as xgb

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.datasets import load_boston

import lightgbm as lgb
import os 
import sys
from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_squared_log_error,median_absolute_error,r2_score,explained_variance_score

from multiprocessing import Pool
import sys
import multiprocessing
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
print(sys.argv)
all_number= int(sys.argv[1])

def custom_objective(preds, train_data):
    labels = train_data.get_label()
    residual = (labels - preds).astype("float")
    grad = -2 * residual
    hess = 2 * np.ones_like(labels)
    return grad, hess
def custom_r2_metric(preds, train_data):
    labels = train_data.get_label()
    r2 = r2_score(labels, preds)
    return 'r2', r2, True
def remove_seasonal(var,yearN,stepN):
    var_re = np.zeros_like(var)*0.0
    for i_box in range(stepN):
        temp1 = np.array([var[stepN*i_year+i_box] for i_year in range(yearN)])
        if (i_box>=2)&(i_box<=stepN-3):
            temp = []
            for i_year in range(0, yearN):
                temp.extend(var[stepN*i_year+i_box-2:stepN*i_year+i_box+3] )
            temp = np.array(temp)
        elif i_box<2:
            temp = []
            temp.extend(var[0:5])
            for i_year in range(1, yearN):
                temp.extend(var[stepN*i_year+i_box-2:stepN*i_year+i_box+3] )
            temp = np.array(temp)
        else:
            temp = []
            for i_year in range(0, yearN-1):
                temp.extend(var[stepN*i_year+i_box-2:stepN*i_year+i_box+3] )
            temp.extend(var[-5:])
            temp = np.array(temp)
            
        temp1 = temp1 - np.array(len(temp1)*[np.nanmean(temp,axis=0)])
        #temp = signal.detrend(temp,axis=0)

        for i_year in range(yearN):
            var_re[stepN*i_year+i_box]  = temp1[i_year]
    return var_re

def detrend(var,yearN,stepN):
    var_detrend = np.zeros_like(var)*0.0
    for i_year in range(yearN):
        temp = np.array(var[stepN*i_year:stepN*i_year+stepN] )
        if i_year in range(2,yearN-2):
            temp1 = np.nanmean(np.array(var[stepN*(i_year-2):stepN*(i_year+3)] ),axis=0)
        elif i_year in range(2):
            temp1 = np.nanmean(np.array(var[:stepN*5] ),axis=0)
        else:
            temp1 = np.nanmean(np.array(var[-stepN*5:] ),axis=0)
        
        temp = temp - np.array(len(temp)*[temp1])
        #temp = signal.detrend(temp,axis=0)
        var_detrend[stepN*i_year:stepN*i_year+stepN]  = temp
    return var_detrend

def standard(var,yearN,stepN):
    return (var-np.nanmean(var))/np.nanstd(var)
def standard_re(var,yearN,stepN):
    return (var-np.nanmean(var))/np.nanstd(var),np.nanmean(var),np.nanstd(var)

# Define the custom quantile loss function for scoring
def quantile_loss(y_true, y_pred, alpha=0.9):
    diff = y_true - y_pred
    return np.mean(np.maximum(alpha * diff, (alpha - 1) * diff))

# Create a custom scorer using make_scorer
quantile_loss_scorer = make_scorer(quantile_loss, greater_is_better=False, alpha=0.9)


########################## CAL SHAP
def run_point(i_lat,i_lon, X_all,y_all,t2m_std1 , t2m_ave1):

    ## basic information
    print('basic information')
    lat_grid = lat_era5[i_lat]
    lon_grid = lon_era5[i_lon]
    print(lat_grid,lon_grid)

    # Define the model
    # Define the hyperparameter search space
    param_space = {
        'learning_rate': Real(0.001, 0.3, prior='log-uniform'),
        'n_estimators': Integer(100, 1000),
        'max_depth': Integer(2, 12),
        'min_samples_split': Integer(2, 20),
        'min_samples_leaf': Integer(1, 20),
        #'max_features': Categorical(['auto', 'sqrt', 'log2']),
        'max_features': Real(0.5, 1.0, prior='uniform'),
        'subsample': Real(0.6, 1.0, prior='uniform')
    }

    # # Define the regressor with the 'Mean squared error' function
    gbr = GradientBoostingRegressor(loss='ls', random_state=42)
    print('hyperparameter')
    # Perform Bayesian optimization
    opt = BayesSearchCV(
        estimator=gbr,
        search_spaces=param_space,
        n_iter=50,  # Number of iterations to perform
        cv=5,  # Number of cross-validation folds
        n_jobs=1,  # Use all available cores
        # scoring=quantile_loss_scorer, # Custom scorer for Mean squared error
        scoring='neg_mean_squared_error',
        random_state=42,
        verbose=0
    )


    # Perform the hyperparameter optimization
    opt.fit(X_all, y_all)
    best_params_b = []
    for key, value in opt.best_params_.items():
        print(f'{key}: {value}')
        best_params_b.append(value)
    best_params = best_params_b

    np.save('/p/projects/climber3/ytian/SHAP_results/GradientBoostingRegressor_b_3d_MAE_5fold_Tmax/best_params_b_lat'+str(lat_grid)+'_lon'+str(lon_grid),best_params_b)   

    model_evaluate_run_mean = []
    y_predict_all_run_mean = []
    shap_interaction_values_run_mean = []

    for i_run in range(1,101):
        ## 5-fold cross-validation indices
        train_indices = []
        test_indices = []
        exclude_positions = []
        indices_all = np.arange(len(X_all))
        test_len = np.int (len(X_all)*0.2)
        for i_round in range (4):
            indices_now = [item for i, item in enumerate(indices_all) if i not in exclude_positions]
            indices = random.sample(indices_now, test_len)
            exclude_positions.extend(indices)
            test_indices.append(indices)
            train_indices.append([item for i, item in enumerate(indices_all) if i not in indices])
            
        indices = [item for i, item in enumerate(indices_all) if i not in exclude_positions]
        train_indices.append([item for i, item in enumerate(indices_all) if i not in indices])
        test_indices.append(indices)

        ## run 5-fold
        shap_interaction_values_all = np.zeros((len(X_all),2,2))
        y_predict_all = np.zeros(len(X_all))
        nn = 0
        for i_fold in range(5):
            nn=nn+1
            print('fold_'+str(nn))
            X_train = X_all.loc[train_indices[i_fold]]
            X_test =  X_all.loc[test_indices[i_fold]]
            y_train = [y_all[ii] for ii in train_indices[i_fold]]
            y_test =  [y_all[ii] for ii in test_indices[i_fold]]
            print('data_prepare_'+str(nn))


            # run XGBoost model
            model_c = GradientBoostingRegressor(loss='ls',learning_rate=best_params[0],max_depth =best_params[1], max_features=best_params[2],min_samples_leaf = best_params[3],min_samples_split= best_params[4],n_estimators= best_params[5],subsample=best_params[6],random_state=42)

            model_c.fit(X_train, y_train)
            print('run GradientBoostingRegressor_fold_'+str(nn))
            # predict
            y_pred = model_c.predict(X_test)
            y_predict_all[test_indices[i_fold]] = y_pred

            # interprete
            shap_interaction_values = shap.TreeExplainer(model_c).shap_interaction_values(X_test)
            shap_interaction_values_all[test_indices[i_fold]] = shap_interaction_values

        # evaluate
        y_pred = copy.deepcopy(y_predict_all)
        y_test = copy.deepcopy(np.array(y_all))
        test_score1=  mean_absolute_error(y_test*t2m_std1+t2m_ave1, y_pred*t2m_std1+t2m_ave1)
        test_score2 = mean_squared_error(y_test*t2m_std1+t2m_ave1, y_pred*t2m_std1+t2m_ave1)
        test_score3 =  median_absolute_error(y_test*t2m_std1+t2m_ave1, y_pred*t2m_std1+t2m_ave1)
        test_score4  =  r2_score(y_test*t2m_std1+t2m_ave1, y_pred*t2m_std1+t2m_ave1)
        test_score5 =  explained_variance_score(y_test*t2m_std1+t2m_ave1, y_pred*t2m_std1+t2m_ave1)
        test_score6 =   quantile_loss(y_test*t2m_std1+t2m_ave1, y_pred*t2m_std1+t2m_ave1, alpha=0.9)


        model_evaluate_run_mean.append([test_score1 ,test_score2,test_score3,test_score4 ,test_score5,test_score6])
        y_predict_all_run_mean.append(y_predict_all) 
        shap_interaction_values_run_mean.append(shap_interaction_values_all  )

    model_evaluate_run_mean = np.nanmean(model_evaluate_run_mean,axis=0)
    y_predict_all_run_mean = np.nanmean(y_predict_all_run_mean,axis=0)
    shap_interaction_values_run_mean = np.nanmean(shap_interaction_values_run_mean,axis=0)

    np.save('/p/projects/climber3/ytian/SHAP_results/GradientBoostingRegressor_b_3d_MAE_5fold_Tmax/model_evaluate_lat'+str(lat_grid)+'_lon'+str(lon_grid),model_evaluate_run_mean )   
    np.save('/p/projects/climber3/ytian/SHAP_results/GradientBoostingRegressor_b_3d_MAE_5fold_Tmax/y_predict_lat'+str(lat_grid)+'_lon'+str(lon_grid),y_predict_all_run_mean )   
    np.save('/p/projects/climber3/ytian/SHAP_results/GradientBoostingRegressor_b_3d_MAE_5fold_Tmax/shap_interaction_values_lat'+str(lat_grid)+'_lon'+str(lon_grid),shap_interaction_values_run_mean )   


if __name__ == '__main__':
    
    # Number of computation to parallelize

    # print(sys.argv)
    # all_number= int(sys.argv[1])
    ##########################

    NCData = Dataset(r'/p/projects/climber3/ytian/data/ERA5_globe/land_sea_mask_2023_01_01_1x1.nc')
    land_mask = np.squeeze(NCData.variables['var172'][:])[all_number*10:all_number*10+10,:]
    lon_era5= NCData.variables['lon'][:][:]
    lat_era5 = NCData.variables['lat'][:][all_number*10:all_number*10+10]
    LON_era5, LAT_era5 = np.meshgrid(lon_era5, lat_era5)
    print(NCData.variables['lon'][:][:][:10])
    print(NCData.variables['lat'][:][:][:10])
    NCData.close()
    
    

    print ('T2m')
    NCData = Dataset(r'/p/projects/climber3/ytian/data/ERA5_globe/ERAland_1950_2023_T2mmax_1.x1._3daymean.nc')
    t2m_all = np.squeeze(NCData.variables['var167'][:])[:,all_number*10:all_number*10+10,:]
    time = NCData.variables['time']
    dates = list(num2date(time[:], time.units, time.calendar))
    year_era5_all = np.array([date.year for date in dates])
    mon_era5_all = np.array([date.month for date in dates])
    day_era5_all = np.array([date.day for date in dates])
    print(NCData.variables['lon'][:][:][:10])
    print(NCData.variables['lat'][:][:][:10])
    NCData.close()

    yearN = 74
    stepN = int(len(year_era5_all)/yearN)

    print ('SM')
    NCData = Dataset(r'/p/projects/climber3/ytian/data/ERA5_globe/ERAland_1950_2023_swvl1_1.x1._3daymean.nc')
    SM_all = np.squeeze(NCData.variables['var39'][:])[:,all_number*10:all_number*10+10,:]
    time = NCData.variables['time']
    dates = list(num2date(time[:], time.units, time.calendar))
    year_era5_all = np.array([date.year for date in dates])
    mon_era5_all = np.array([date.month for date in dates])
    day_era5_all = np.array([date.day for date in dates])
    print(NCData.variables['lon'][:][:][:10])
    print(NCData.variables['lat'][:][:][:10])
    NCData.close()


    print ('Z500')
    NCData = Dataset(r'/p/projects/climber3/ytian/data/ERA5_globe/ERA5_1950_2023_Z500_1.x1._3daymean.nc')
    Z500_all = np.squeeze(NCData.variables['var129'][:])[:,all_number*10:all_number*10+10,:]/9.81
    time = NCData.variables['time']
    dates = list(num2date(time[:], time.units, time.calendar))
    year_era5_all = np.array([date.year for date in dates])
    mon_era5_all = np.array([date.month for date in dates])
    day_era5_all = np.array([date.day for date in dates])
    print(NCData.variables['lon'][:][:][:10])
    print(NCData.variables['lat'][:][:][:10])
    NCData.close()



    ################### read t2m, Z500, SM

    year_era5_all = np.array(year_era5_all)
    mon_era5_all = np.array(mon_era5_all)
    day_era5_all = np.array(day_era5_all )
    t2m_all = np.array(t2m_all)
    SM_all = np.array(SM_all)
    Z500_all = np.array(Z500_all)

    ################### 
    if all_number<7:
        JJA_index = np.where(np.in1d(mon_era5_all,[6,7,8]))[0]
    elif all_number>=11:
        JJA_index = np.where(np.in1d(mon_era5_all,[12,1,2]))[0]
    else:
        JJA_index = np.where(np.in1d(mon_era5_all,range(1,13)))[0]
    


    i_lat_all =[]
    i_lon_all = []

    X_all_all = []
    y_all_all = []

    X_std_all = []
    X_ave_all = []
    processes = []
    for i_lat in range(land_mask.shape[0]):
        for i_lon in range(land_mask.shape[1]):
             # Start the core parallelization over index
            lat_grid = lat_era5[i_lat]
            lon_grid = lon_era5[i_lon]

            if np.isnan(np.array(SM_all[:,i_lat,i_lon])).any():
                continue
            if np.isnan(np.array(t2m_all[:,i_lat,i_lon])).any():
                continue
            if np.isnan(np.array(Z500_all[:,i_lat,i_lon])).any():
                continue
            if land_mask[i_lat,i_lon] <0.1:
                continue          
            if os.path.exists('/p/projects/climber3/ytian/SHAP_results/GradientBoostingRegressor_b_3d_MAE_5fold_Tmax/shap_interaction_values_lat'+str(lat_grid)+'_lon'+str(lon_grid)+'.npy'):
                continue
            
            print(lat_grid,lon_grid )


            t2m_region = np.array([i for i in t2m_all[:,i_lat,i_lon]])
            if len(t2m_region[np.isnan(t2m_region)])>0:
                print(lat_grid,lon_grid,'NAN' )
                continue
            t2m_region_re = copy.deepcopy(remove_seasonal(t2m_region,yearN,stepN))
            t2m_region_re_std = copy.deepcopy(standard(t2m_region_re,yearN,stepN))
            t2m_region_re_de = copy.deepcopy(detrend(t2m_region_re,yearN,stepN))
            t2m_region_re_de_std = copy.deepcopy(standard(t2m_region_re_de,yearN,stepN))
            t2m_ave1 = copy.deepcopy(standard_re(t2m_region_re_de,yearN,stepN))[1]
            t2m_std1 = copy.deepcopy(standard_re(t2m_region_re_de,yearN,stepN))[2]


            Z500_region = np.array([i for i in Z500_all[:,i_lat,i_lon]])
            if len(Z500_region[np.isnan(Z500_region)])>0:
                print(lat_grid,lon_grid,'NAN' )
                continue
            Z500_region_re = copy.deepcopy(remove_seasonal(Z500_region,yearN,stepN))
            Z500_region_re_std = copy.deepcopy(standard(Z500_region_re,yearN,stepN))
            Z500_region_re_de = copy.deepcopy(detrend(Z500_region_re,yearN,stepN))
            Z500_region_re_de_std = copy.deepcopy(standard(Z500_region_re_de,yearN,stepN))


            SM_region = np.array([i for i in SM_all[:,i_lat,i_lon]])
            if len(SM_region[np.isnan(SM_region)])>0:
                print(lat_grid,lon_grid,'NAN' )
                continue
            SM_region_re = copy.deepcopy(remove_seasonal(SM_region,yearN,stepN))
            SM_region_re_std = copy.deepcopy(standard(SM_region_re,yearN,stepN))
            SM_region_re_de = copy.deepcopy(detrend(SM_region_re,yearN,stepN))
            SM_region_re_de_std = copy.deepcopy(standard(SM_region_re_de,yearN,stepN))

            i_lat_all.append(i_lat)
            i_lon_all.append(i_lon)
            d = {'Z500': list(Z500_region_re_de_std[JJA_index ]),'SM': list(SM_region_re_de_std[JJA_index ])}
            df = pd.DataFrame(data=d)
            X_all = df
            y_all = t2m_region_re_de_std[JJA_index ]
            # create a train/test split
            # X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
            
            X_all_all.append(X_all)
            y_all_all.append(y_all)     

            X_std_all.append(t2m_std1)
            X_ave_all.append(t2m_ave1)      

    print('data prepared' )

    with Pool(processes=64) as p:
        p.starmap(run_point, [(i_lat,i_lon,X_all,y_all,t2m_std1,t2m_ave1) for  (i_lat,i_lon,X_all,y_all,t2m_std1,t2m_ave1) in  zip(i_lat_all,i_lon_all,X_all_all,y_all_all,X_std_all,X_ave_all) ]  )



