#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Zhang Xi

model Learning, prediction and visulization  of a single twitter cascade.
"""

import pandas as pd
import numpy as np
import json
import copy
from itertools import combinations
from matplotlib import pyplot as plt
import argparse
from pathlib import Path, PosixPath
from typing import Dict, List, Tuple

from function_prediction import  get_prediction_results
from function_learning_discriminative import discriminative_projected_gradient_descent


# code snippet obtained from https://github.com/kratzert/lstm_for_pub/blob/master/main.py 
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



# code snippet obtained from https://github.com/kratzert/lstm_for_pub/blob/master/main.py
def get_args() -> Dict:
    """Parse input arguments
    Returns
    -------
    dict
        Dictionary containing the run config.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_file_path',
        default='data/seismic/example_cascade_2.json',
        type=str,
        help="Path to the file that contains the realization")
    parser.add_argument(
        '--censoring_time',
        type=str,
        choices=["10min","30min","1h","2h","3h","4h", "5h", "6h"],
        default="1h",
        help=" t_c, Censoring time enter number followed immediately by the time granularity. For example '1h', '20h','1d', '5d' ")
    
    # parser.add_argument(
    #     '--prediction_time_intervals',
    #     type=str,
    #     default=['1h', '4h', '8h', '12h', '16h', '20h','1d', '2d', '3d', '4d', '5d', '6d'] ,
    #     help=" Delta_t, prediction time interval  (in minutes) ")

    parser.add_argument(
        '--run_example',
        type=str2bool,
        default=False,
        help="Boolean variable to indicate if example is to be run")

    parser.add_argument(
        '--output_path',
        type=str,
        default="./",
        help="Location to store the output bar plot")

    
    cfg = vars(parser.parse_args())

   
    return cfg


def sec2scale(granularity_hours_scale_setup, time_str):
    
    granularity_hours = granularity_hours_scale_setup[time_str]
    
    gs = 60*60*granularity_hours
    
    return gs
    

def learn(censoring_time, cascade, gs, method = 'censored'):
    
    print('model learning')
    theta_min = pd.Series({'gamma': 1e-6, 'beta':1e-6, 'kappa': 1e-6} )
    theta_max = pd.Series({'gamma': 10.0, 'beta':100.0, 'kappa': 10.0 } )
    theta_init = pd.Series({'gamma': 0.1, 'beta': 0.1, 'kappa':0.1})
        
    data = copy.deepcopy(cascade)
    data['hw'] = [(e[0]/gs, e[1]) for e in data['hw'] ]
    
    tc = pd.Timedelta(censoring_time).total_seconds() / gs
    events = np.array(data['hw'])
    obs_events = events[events[:,0]<=tc]
    
    if method == 'all':     
        # all pairs
        grid_times = np.append(obs_events[:,0], tc) 
        time_pairs = np.array(list(combinations(grid_times, 2)))
    elif method == 'censored':
        # only predict at tc
        time_pairs = np.column_stack((obs_events[:,0], tc* np.ones(len(obs_events))))
        
    elif method == 'fix_num':
        N = 20
        temp_1 = np.column_stack((obs_events[:,0], tc* np.ones(len(obs_events))))
        temp_2 = np.array(list(combinations(obs_events[:,0], 2)))
        
        if (len(temp_1) + len(temp_2) < N):
            time_pairs =  np.concatenate((temp_1, temp_2), axis=0)
        else:
            if len(obs_events) < N:
                make_ups = np.random.choice(len(temp_2), size= N-len(temp_1), replace=False)
                time_pairs = np.concatenate((temp_1, temp_2[make_ups,:]), axis=0)
            else:
                time_pairs = temp_1[np.random.choice(len(temp_1), size= N, replace=False), :]
    
    
    theta_learned, obj_list = discriminative_projected_gradient_descent(data, tc,\
                  np.array(theta_init), np.array(theta_min), np.array(theta_max),\
                  time_pairs = time_pairs, step_size_init = 1e-2)

    
    return theta_learned
    


def predict(delta_times, cascade, censoring_time, theta_learned, gs, get_bounds=False):
        
    data = copy.deepcopy(cascade)
    data['hw'] = [(e[0]/gs, e[1]) for e in data['hw'] ]
    
    tc = pd.Timedelta(censoring_time).total_seconds() / gs
    delta_t = [pd.Timedelta(delta_time).total_seconds() / gs for delta_time in delta_times] 
    
    pred_times = np.array(delta_t) + tc   
    r_df = get_prediction_results(data, theta_learned, tc, pred_times, get_bounds= get_bounds, eb = 0.05)
    
    results_df = r_df.drop(columns=['pred_time'])
    results_df['delta_time'] = delta_times
    results_df = results_df.set_index('delta_time')
    
    return results_df

def visulization_plot(figName, cascade, censoring_time, theta_learned, last_pred_time ='7d', get_bounds=False, fontsize=12):
    data = copy.deepcopy(cascade)
    data['hw'] = [(e[0]/gs, e[1]) for e in data['hw'] ]
    
    tc = pd.Timedelta(censoring_time).total_seconds() / gs
    t_end = pd.Timedelta(last_pred_time).total_seconds() / gs
    
    pts_1 = np.linspace(0, tc, 10)
    pts_2 = np.linspace(tc, t_end, 90)
    pts = np.concatenate([pts_1, pts_2])
    
    s_df = get_prediction_results(data, theta_learned, tc, pts, get_bounds= get_bounds, eb = 0.05)
    
    
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(s_df['pred_time'], s_df['pred_count'], 'r', label='predicted count')
    ax.plot(s_df['pred_time'], s_df['true_count'], 'g', label='true count')    

    ax.set_xticks([tc, t_end])
    ax.set_xticklabels(['tc={}'.format(censoring_time), last_pred_time ], fontsize=fontsize)
    ax.legend(loc='lower right',fontsize=fontsize)
    ax.set_xlabel('time', fontsize=fontsize)
    ax.set_ylabel('count',fontsize=fontsize)
    if get_bounds:
        ax.fill_between(s_df['pred_time'], s_df['lower_bound'] , s_df['upper_bound'],facecolor='lightgrey')
        ax.set_title('Predicted Counts with 95% Intervals', fontsize=fontsize)
    else:
        ax.set_title('Predicted Counts', fontsize=fontsize)

    plt.savefig(figName, dpi=300, transparent=True, bbox_inches='tight')
    
    return s_df

def run_example():
    dataPath = 'data/seismic/'
    with open(dataPath + 'example_cascade_2.json', 'r') as f:
        cascade = json.load(f)

    granularity_hours_scale_setup = {'10min': 2, '30min': 6, '1h': 9, '2h': 19,\
                                      '3h': 24, '4h': 24, '5h':24, '6h':24}

    #--------------------Specify censoring time--------------------------------   
    # following the format of pd.Timedelta. For example, '10min', '1h', '2d'
    censoring_time = '1h'
    gs = sec2scale(granularity_hours_scale_setup, censoring_time)  

    #--------------------Learn-------------------------------------------------    
    # To learn the model parameters
    theta_learned = learn(censoring_time, cascade, gs)

    #--------------------Specify prediction intervals--------------------------
    # following the format of pd.Timedelta. For example, '10min', '1h'
            
    delta_times = ['1h', '4h', '8h', '12h', '16h', '20h', \
                    '1d', '2d', '3d', '4d', '5d', '6d'] 
        
    #--------------------Predict-----------------------------------------------
    if theta_learned['gamma'] >=1:
        print('the cascade will be viral')
    else:     
        pred_results = predict(delta_times, cascade, censoring_time, theta_learned, gs, get_bounds=False)


    #--------------------Visulization------------------------------------------
    s_df = visulization_plot(cfg['output_path'] + 'example_figure.png', cascade, censoring_time, theta_learned, last_pred_time ='1d', get_bounds=False)


if __name__ == '__main__':
    
    cfg = get_args()
    if cfg['run_example'] == True:
        run_example()
    else:
        datafile = cfg['data_file_path']
        with open(datafile , 'r') as f:
            cascade = json.load(f)

        granularity_hours_scale_setup = {'10min': 2, '30min': 6, '1h': 9, '2h': 19,\
                                        '3h': 24, '4h': 24, '5h':24, '6h':24}

        #--------------------Specify censoring time--------------------------------   
        # following the format of pd.Timedelta. For example, '10min', '1h', '2d'
        censoring_time = cfg['censoring_time']
        gs = sec2scale(granularity_hours_scale_setup, censoring_time)  

        #--------------------Learn-------------------------------------------------    
        # To learn the model parameters
        theta_learned = learn(censoring_time, cascade, gs)

        #--------------------Specify prediction intervals--------------------------
        # following the format of pd.Timedelta. For example, '10min', '1h'
                
        delta_times = ['1h', '4h', '8h', '12h', '16h', '20h', \
                        '1d', '2d', '3d', '4d', '5d', '6d'] 
            
        #--------------------Predict-----------------------------------------------
        if theta_learned['gamma'] >=1:
            print('the cascade will be viral')
        else:     
            pred_results = predict(delta_times, cascade, censoring_time, theta_learned, gs, get_bounds=False)


        #--------------------Visulization------------------------------------------
        s_df = visulization_plot('example_figure.png', cascade, censoring_time, theta_learned, last_pred_time ='1d', get_bounds=False)









