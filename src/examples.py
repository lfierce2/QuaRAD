#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Laura Fierce
"""

import run
import plot

emission_type = 'talk'

# =============================================================================
# RUN BASELINE SCENARIO
# =============================================================================
scenario_name = 'base'
griddings = ['vs_xz']
user_options = {'inactivates':[False],'Nts_jet':int(1e4),'Nts_reflections':int(1e4),'tau_thresh':3e-4,'tlim_thresh':1e-10}
scenario_dir = run.one_scenario(
        scenario_name,griddings,
        emission_type=emission_type,user_options=user_options,get_next_dir=True,
        trajectories_only=False,overwrite_trajectories=False,overwrite_gridding=True,
        output_dir='../output/',excel_filename='../input/inputs.xlsx')

# =============================================================================
# RUN QUADRATURE OPTIMIZATION CASES
# =============================================================================
scenario_name = 'base'
griddings = ['vs_x']
user_options = {
        'inactivates':[False],'Nts_jet':int(1e4),'Nts_reflections':int(1e4),
        'tau_thresh':3e-4,'tlim_thresh':1e-10,
        'N_quads':[[1,3,2],[1,1,1],[200,200,200]]}
scenario_dir__quad = run.one_scenario(
        scenario_name,griddings,
        emission_type=emission_type,user_options=user_options,get_next_dir=True,
        trajectories_only=False,overwrite_trajectories=False,overwrite_gridding=True,
        output_dir='../output/',excel_filename='../input/inputs.xlsx')


# =============================================================================
# RUN WEI & LI CASE
# =============================================================================
scenario_name = 'Wei&Li' # Note: excel sheet needs to be modified!
griddings = ['vs_xz']
user_options = {
        'inactivates':[False],'Nts_jet':int(1e4),'Nts_reflections':int(1e4),
        'tau_thresh':3e-4,'tlim_thresh':1e-10,'N_quads':[[1,1,1]]}
scenario_dir__verify = run.one_scenario(
        scenario_name,griddings,
        emission_type=emission_type,user_options=user_options,get_next_dir=True,
        trajectories_only=False,overwrite_trajectories=False,overwrite_gridding=True,
        output_dir='../output/',excel_filename='../input/inputs.xlsx')

# =============================================================================
# RUN ENSEMBLE OF SCENARIOS
# =============================================================================
N_jet = 1000
N_infection = 10
ensemble_name = 'orig'
griddings = ['vs_x']
user_options = {
        'inactivates':[False],
        'smask_types':['none'],
        'rmask_types':['none'],
        'acr_enhancements':[1.],
        'other_param_dicts':[{}],
        'max_cpu_time':120,'tau_thresh':3e-4,
        'tlim_thresh':1e-12,'tlims_log':[-6,3],'N_loops':3,
        'Nts_jet':int(1e4),'Nts_refelctions':int(1e4)}
ensemble_dir = run.ensemble(
        ensemble_name,N_jet,griddings,
        emission_type=emission_type,user_options=user_options,
        get_next_dir=True,N_infection=N_infection,
        trajectories_only=False,overwrite_trajectories=False,overwrite_gridding=False,
        output_dir='../output/',excel_filename='../input/inputs.xlsx')



## =============================================================================
## MAKE ALL FIGURES
## =============================================================================
fontsize=11
plot.fig1(scenario_dir,fontsize=fontsize)
plot.fig2(scenario_dir,fontsize=fontsize)
plot.fig3(scenario_dir,fontsize=fontsize,normalize=False,colorscale='log')
plot.fig4(ensemble_name='orig',emission_type=emission_type,fontsize=fontsize,output_dir=ensemble_dir + '../../',whis=[2.5,97.5])
plot.fig5(ensemble_name='orig',emission_type=emission_type,fontsize=fontsize,whis=[2.5,97.5])
plot.fig6(scenario_dir,fontsize=fontsize,normalize=False,colorscale='log')
plot.fig7(ensemble_dir,fontsize=fontsize,whis=[2.5,97.5])

plot.SIfig1(scenario_dir__quad,varname='p_infect',fontsize=fontsize)


plot.SIfig2(
        scenario_dir__verify,fontsize=fontsize,
        dispersion_type='jet',normalize=False,colorscale='log',qq_vals=[0,2])


