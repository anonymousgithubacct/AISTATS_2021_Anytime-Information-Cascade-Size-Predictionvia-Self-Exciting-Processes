#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 15:55:13 2020

@author: XXXXX XX

"""

import pandas as pd
import numpy as np

from scipy.special import factorial



#==============================================================================
#================phi(t) = m^kappa * alpha * exp(-beta*t)=====================
#==================b(t) = 0 ===================================================
#======================== marks: g(m) = =======================================
#==============================================================================


def marked_exp_mean_count(t, alpha, beta, kappa, gamma, tc, obs_events):
    
    if t <= tc:
        value = len([e for e in obs_events if e[0] <= t])
    else: 

        n_alpha = gamma*beta
        
        temp_df = pd.DataFrame(obs_events, columns=['event_time', 'event_mark'])
        temp_df['value'] = temp_df.apply(lambda x: np.power(x['event_mark'], kappa)*\
                                         np.exp(-beta*(tc - x['event_time'])) ,axis=1)
        temp_val = temp_df['value'].sum() * alpha / (gamma*beta)
        
        if n_alpha == beta: 
            f1 = n_alpha*(t-tc)
        else:
            f1 = ( 1 - np.exp(-(beta-n_alpha)*(t-tc)) ) *gamma / (1-gamma)
        
        value = temp_val*f1 + len(obs_events)
            
    return value

def marked_generation_count_mean(t, k, alpha, beta, kappa, gamma, tc, init_obs):
      
    if t <= tc:
        value = 0
    else:
        n_alpha = gamma*beta
        
        temp_df = pd.DataFrame(init_obs, columns=['event_time', 'event_mark'])
        temp_df['value'] = temp_df.apply(lambda x: np.power(x['event_mark'], kappa)* np.exp(-beta*(tc - x['event_time'])) ,axis=1)
        temp_val = temp_df['value'].sum() * alpha / (gamma*beta)
        
    
        td = t-tc
        temp_sum = 0
        for j in np.arange(k+1):
            temp_sum = temp_sum + np.power(beta*td, j) * np.exp(-beta*td)/ factorial(j)
    
        value = temp_val * np.power((n_alpha/beta), k+1) * (1-temp_sum)
    
    return value

def marked_count_mean_approximation(t, maxG, alpha, beta, kappa, gamma, tc, init_obs):
    
    if t <= tc:
        value = len([e for e in init_obs if e[0] <= t])
    else: 
        value = len(init_obs)

        for k in np.arange(0, maxG):
            value = value + marked_generation_count_mean(t, k, alpha, beta, kappa, gamma, tc, init_obs)
            
    return value

def marked_generation_count_var(t, k, alpha, beta, kappa, gamma, nv, tc, init_obs):
    
    if t <= tc:
        value = 0.0
        
    else:
        temp_df = pd.DataFrame(init_obs, columns=['event_time', 'event_mark'])
        temp_df['value'] = temp_df.apply(lambda x: np.power(x['event_mark'], kappa)* np.exp(-beta*(tc - x['event_time'])) ,axis=1)
        temp_val = temp_df['value'].sum() 
                
        eta = temp_val * (alpha / beta) 
              
        lk = marked_generation_count_mean(t, k, alpha, beta, kappa ,gamma, tc, init_obs)
                          
        if gamma == 1:
            value = lk + np.power(lk,2) * (k-1) / eta
        else:
            value = lk + np.power(lk,2) * nv *\
                ( 1-np.power(gamma,k) )/(np.power(gamma,k-1) - np.power(gamma,k))/ (eta * np.power(gamma,2))
    return value



def marked_count_var_approximation(t, maxG, alpha, beta, kappa, gamma, nv, tc, init_obs):
    if t <= tc:
        var_approx = 0.0
    else:
        var_approx = 0.0
            
        count_mean = marked_exp_mean_count(t, alpha, beta, kappa, gamma, tc, init_obs)
        
        for k in np.arange(0, maxG):
            
            mean_k =  marked_generation_count_mean(t, k, alpha, beta, kappa, gamma, tc, init_obs)
            var_k = marked_generation_count_var(t, k, alpha, beta, kappa, gamma, nv, tc, init_obs)
            
            temp = sum([marked_generation_count_mean(t, i, alpha, beta, kappa, gamma, tc, init_obs) \
                        for i in np.arange(0, k+1)]) + len(init_obs)
            
            if mean_k == 0:
                para = 1   
            else:
                para = 1+ 2*(count_mean - temp)/mean_k
        
            var_approx = var_approx + para*var_k
            
    return var_approx 



if __name__ == '__main__':
    
    pass


























