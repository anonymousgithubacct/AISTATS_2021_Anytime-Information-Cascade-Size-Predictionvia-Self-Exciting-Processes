#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: XXXXX XX

"""

import pandas as pd
import numpy as np

from function_learning_discriminative import get_mark_dist
from function_conditional_moments import marked_exp_mean_count, marked_count_var_approximation


#==============================================================================
#==============================================================================
#==============================================================================
    

# used for getting results for a single cascades
def get_prediction_results(realization, theta_learned, tc, pred_times, get_bounds= False, N_max = 10,  eb = 0.05 ):
    
    # tc: right-censored time, observed events up to and include this time.
    # pts: list of times at which the learned and gt are being compared

    
    gamma_learned = theta_learned['gamma']
    beta_learned = theta_learned['beta']
    kappa_learned = theta_learned['kappa']
    alpha_learned = theta_learned['alpha']
    

    events = realization['hw']
    
    
    # ts = pts[0]
    obs_events = [e for e in realization['hw'] if e[0]<= tc]
    
    s_df = pd.DataFrame({'pred_time': pred_times})
    
    s_df['true_count'] = s_df.apply(lambda x: len([e for e in events if e[0] <= x['pred_time']]), axis=1)
    
    print('calculating conditional means')
    s_df['pred_count'] = s_df.apply(lambda x: marked_exp_mean_count(x['pred_time'], alpha_learned,\
                            beta_learned, kappa_learned, gamma_learned, tc, obs_events ) , axis=1)   
        
        
    if get_bounds:
        
        mark_dist, _ = get_mark_dist(realization, tc, include_tc = True)
        
        # f1 = np.sum( np.power(mark_dist[:,0], kappa_learned)*mark_dist[:,1])
        # learned_gamma = theta_learned['alpha'] * f1 / theta_learned['beta']
        
        f2 = np.sum( np.power(mark_dist[:,0], 2*kappa_learned)*mark_dist[:,1])
        nv = np.power(alpha_learned/beta_learned, 2) * f2

        print('calculating conditional variances')

        s_df['pred_var'] = s_df.apply(lambda x: marked_count_var_approximation(x['pred_time'], N_max, alpha_learned,\
                                beta_learned, kappa_learned, gamma_learned, nv, tc, obs_events ) , axis=1)   
    
                
        s_df['lower_bound']  = s_df.apply(lambda x: x['pred_count'] -  np.sqrt( ( x['pred_var'] / eb ) ) if x['pred_time'] > tc else x['true_count'] ,axis=1)
           
        s_df['lower_bound'] =  s_df['lower_bound'].mask( (s_df['lower_bound'] <len(obs_events)) & (s_df['pred_time']>tc), len(obs_events))

         
        s_df['upper_bound']  = s_df.apply(lambda x: x['pred_count'] + np.sqrt( (x['pred_var'] / eb ))  if x['pred_time'] > tc else x['true_count'] ,axis=1)

    return s_df

        
    

if __name__ == '__main__':
    
    pass


























