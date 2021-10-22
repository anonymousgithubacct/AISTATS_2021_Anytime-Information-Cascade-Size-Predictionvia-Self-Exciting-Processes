
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 01:16:14 2021

@author: XXXXX XX
"""
import numpy as np
import pandas as pd
from functools import partial
from numba import jit
import random
from itertools import combinations

#==============================================================================
#====================Basic Construction Functions==============================
#==============================================================================
# def get_pairs(ori_list):
    
#     ori_list.sort()
    
#     # all_pairs = []
#     fl = []
#     sl = []
#     for i, vi in enumerate(ori_list[:-1]):    
#         # all_pairs.append([vi, ori_list[i+1:]])
#         fl.append(vi)
#         sl.append(ori_list[i+1:])
#     return fl, sl



def get_mark_dist(realization, tc, include_tc = True):
    # find current empirical mark distribution
    if include_tc == True:
        obs_events = [e for e in realization['hw'] if e[0]<= tc]
    else:
        obs_events = [e for e in realization['hw'] if e[0]< tc]

    obs_df = pd.DataFrame(obs_events, columns=['event_time', 'event_mark'])    
    obs_df['mark_count'] =  obs_df.groupby('event_mark')['event_mark'].transform('count') 
    mark_dist = obs_df[['event_mark', 'mark_count']].drop_duplicates()
    mark_dist['mark_prob'] = mark_dist['mark_count'] /  mark_dist['mark_count'].sum()
    
    mark_dist_df = mark_dist[['event_mark', 'mark_prob']]
    mark_dist_array = np.array(mark_dist_df)
    
    return mark_dist_array, mark_dist_df

@jit(nopython=True)
def p_func(gamma, beta, tij):
    
    if gamma == 1:
        value = gamma * beta*tij
    else:
        value = (1 - np.exp(-beta*(1-gamma)*tij))*gamma / (1-gamma) 
    return value

@jit(nopython=True)
def p_grad_cal(gamma, beta, tij):
    if tij<0:
        print('ERROR')
        
    if gamma == 1:
        pd_gamma = beta*tij
        pd_beta = gamma*tij
    else:
        pd_gamma = (1-np.exp(-beta*(1-gamma)*tij)) / np.power(1-gamma, 2) -\
                    gamma * beta * tij * np.exp(-beta*(1-gamma)*tij) / (1-gamma) 
                    
        pd_beta = gamma*tij*np.exp(-beta*(1-gamma)*tij)
        
    return pd_gamma, pd_beta

        
@jit(nopython=True)
def q_func(kappa, beta, ti, i_hist, mark_dist):
        
    a = np.dot(i_hist[:,1]**kappa, np.exp(-beta* np.subtract(ti,  i_hist[:,0])))
    f1 = np.sum((mark_dist[:,0]**kappa)*mark_dist[:,1])
        
    return a / f1

@jit(nopython=True)
def q_grad_cal(kappa, beta, ti, i_hist, mark_dist):
    
    i_hist_times = i_hist[:,0]
    i_hist_marks = i_hist[:,1]

    f1 = np.sum( np.power(mark_dist[:,0], kappa)*mark_dist[:,1])
    
    fd_1 = np.sum( np.log(mark_dist[:,0]) * np.power(mark_dist[:,0], kappa)*mark_dist[:,1] )
        
    qd_kappa_1 = np.dot(np.log(i_hist_marks)*np.power(i_hist_marks, kappa) , \
                        np.exp(-beta* np.subtract(ti, i_hist_times)) ) / f1   
        
    qd_kappa_2 = np.dot(np.power(i_hist_marks, kappa), \
                        np.exp(-beta* np.subtract(ti, i_hist_times))) * fd_1 / np.power(f1,2)
        
    qd_kappa = qd_kappa_1 - qd_kappa_2

    qd_beta = - np.dot(np.power(i_hist_marks, kappa), np.subtract(ti, i_hist_times) * \
                       np.exp(-beta* np.subtract(ti, i_hist_times))) / f1
    
    return qd_kappa, qd_beta

def param_update(theta, step_size, grad_theta, theta_min, theta_max):
    
    theta_next = theta - grad_theta * step_size
    
    theta_next = np.clip(theta_next, theta_min, theta_max)
            
    return theta_next

#==============================================================================
#==============================================================================
#==============================================================================

@jit(nopython=True)
def obj_cal(theta, tc, obs_events ,mark_dist, time_pairs): 
        
    value = 0
    
    gamma = theta[0]
    beta = theta[1]
    kappa = theta[2]
    
    for ti in np.unique(time_pairs[:,0]):  
                    
        i_hist = obs_events[obs_events[:,0] <= ti]
        
        N_ti = len(i_hist) 
    
        qi_value = q_func(kappa, beta, ti, i_hist, mark_dist)
        
        tj_list = time_pairs[time_pairs[:,0]==ti][:,1]
    
        for tj in tj_list:
                        
            N_tj = len(obs_events[obs_events[:,0] <= tj])
            
            pij_value = p_func(gamma, beta, (tj-ti)) 
       
            value = value + (pij_value * qi_value -(N_tj-N_ti))**2
            
    value = value / len(time_pairs)
    
    return value

@jit(nopython=True)
def gradient_cal(theta, tc, obs_events, mark_dist, time_pairs):


    gamma = theta[0]
    beta = theta[1]
    kappa = theta[2]
    
    gamma_grad = 0
    beta_grad = 0
    kappa_grad = 0
    
    for ti in np.unique(time_pairs[:,0]):  

        i_hist = obs_events[obs_events[:,0] <= ti] #[e for e in obs_events if e[0]<=ti]
        N_ti = len(i_hist) 
        
        qi_value = q_func(kappa, beta, ti, i_hist, mark_dist)
        qi_grad_kappa, qi_grad_beta = q_grad_cal(kappa, beta, ti, i_hist, mark_dist)

        tj_list = time_pairs[time_pairs[:,0]==ti][:,1]

        for tj in tj_list:
            
            N_tj = len(obs_events[obs_events[:,0] <= tj])
            
            pij_value = p_func(gamma, beta, tj-ti)
            
            pij_grad_gamma, pij_grad_beta = p_grad_cal(gamma, beta, tj-ti)
            
            gamma_grad = gamma_grad + 2*(pij_value * qi_value - (N_tj-N_ti)) * qi_value * pij_grad_gamma
                              
            kappa_grad = kappa_grad + 2*(pij_value * qi_value - (N_tj-N_ti)) * pij_value * qi_grad_kappa
                               
            beta_grad = beta_grad + 2*(pij_value * qi_value - (N_tj-N_ti)) *\
                              (pij_value * qi_grad_beta + qi_value* pij_grad_beta)
                                                     
    return np.array([gamma_grad/len(time_pairs), beta_grad/len(time_pairs), kappa_grad/len(time_pairs)])




def discriminative_projected_gradient_descent(realization, tc, theta_init, theta_min, theta_max, time_pairs = None,\
                                              step_size_init=0.1, bt=0.7, max_iter = 20000, epsilon=1e-6):
    


    theta_init = np.array(theta_init)

    events = np.array(realization['hw'])
    obs_events = events[events[:,0]<=tc]
    
    if time_pairs is None:

        grid_times = np.append(obs_events[:,0], tc) 
        time_pairs = np.array(list(combinations(grid_times, 2)))
    
    
    # find current empirical mark distribution
    mark_dist, _ = get_mark_dist(realization, tc)

    update_func = partial(param_update, theta_min = np.array(theta_min), theta_max=np.array(theta_max))
    obj_func = partial(obj_cal, tc=tc, obs_events=obs_events, mark_dist=mark_dist, time_pairs=time_pairs)
    grad_func = partial(gradient_cal, tc=tc, obs_events= obs_events, mark_dist=mark_dist, time_pairs=time_pairs)       

    
    step_size_list = [step_size_init]
    obj_list = [obj_func(theta_init)]
    
    # initialization
    step_size = step_size_init
    theta_curr = theta_init
    norm_diff = epsilon + 1
    iter_c = 0
    
    while (iter_c < max_iter) and (norm_diff > epsilon):
    
        step_size = step_size_init
        
        grad_curr = grad_func(theta_curr)
        theta_next = update_func(theta_curr, step_size, grad_curr)
        
        next_obj =  obj_func(theta_next)
        inter_count = 0
        while (obj_list[-1] - next_obj <= 0):
            step_size = step_size * bt
            theta_next = update_func(theta_curr, step_size, grad_curr)
            next_obj = obj_func(theta_next)
            
            inter_count = inter_count + 1
            if inter_count > 100:
                theta_next = theta_curr
                break
        
        norm_diff = np.linalg.norm( theta_next - theta_curr)
        iter_c = iter_c + 1
        
        obj_list.append(next_obj)
        step_size_list.append(step_size)
        
        theta_curr = theta_next
                            
        
    theta_learned = theta_curr
    theta_learned = pd.Series(theta_learned, index=['gamma', 'beta', 'kappa'])
    
    f1 = np.sum( np.power(mark_dist[:,0], theta_learned[2])*mark_dist[:,1])
    learned_alpha = theta_learned[0] * theta_learned[1] / f1     
    
    theta_learned['alpha'] = learned_alpha
    
    return theta_learned, obj_list


















