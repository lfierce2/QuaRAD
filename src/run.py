#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Laura Fierce
"""
import pandas as pd
import pickle
import numpy as np
import os

from pyDOE import lhs
from scipy.special import erfinv#, erf
import QuaRAD
import utils


def one_scenario(
        scenario_name,griddings,
        emission_type='talk',user_options={},get_next_dir=True,N_infection=1,
        trajectories_only=False,overwrite_trajectories=False,overwrite_gridding=False,
        output_dir='../output/',excel_filename='../input/inputs.xlsx',ensemble_name='orig'):
    
    X_i_dispersion,varnames_dispersion = get_inputs__onescenario(
            scenario_name, fixed_vars={}, emission_type=emission_type, excel_filename=excel_filename)
    
    X_infection,varnames_infection = get_inputs__ensemble__infection(
                    ensemble_name, N_infection, fixed_vars={'S_source':1.}, 
                    emission_type=emission_type, excel_filename=excel_filename)
    
    if not os.path.exists(output_dir + 'scenarios/'):
        os.mkdir(output_dir + 'scenarios/')
    scenario_dir = get_print_dir(scenario_name,output_dir + 'scenarios/',get_next_dir=get_next_dir)
    
    QuaRAD.run_model(
            scenario_dir,X_i_dispersion,varnames_dispersion,X_infection,varnames_infection,
            griddings=griddings,user_options=user_options,trajectories_only=trajectories_only,
            overwrite_trajectories=overwrite_trajectories,overwrite_gridding=overwrite_gridding)
    return scenario_dir

def ensemble(
        ensemble_name,N_scenarios,griddings,
        emission_type='talk',user_options={},get_next_dir=True,N_infection=10,
        trajectories_only=False,overwrite_trajectories=False,overwrite_gridding=False,
        output_dir='../output/',excel_filename='../input/inputs.xlsx'):
    
    if not os.path.exists(output_dir + 'ensembles/'):
        os.mkdir(output_dir + 'ensembles/')
        
    ensemble_dir = get_print_dir(ensemble_name,output_dir + 'ensembles/',get_next_dir=get_next_dir)
    if not os.path.exists(ensemble_dir):
        os.mkdir(ensemble_dir)
    
    X,varnames_dispersion = get_inputs__ensemble(
            ensemble_name, N_scenarios, fixed_vars={'S_source':1.}, 
            emission_type=emission_type, excel_filename=excel_filename)
    for ii,X_i_dispersion in enumerate(X):
        X_infection,varnames_infection = get_inputs__ensemble__infection(
                ensemble_name, N_infection, fixed_vars={'S_source':1.}, 
                emission_type=emission_type, excel_filename=excel_filename)
        print('need to switch back next line! right now... ii+50')
        scenario_dir = ensemble_dir + str(ii+50).zfill(4) + '/'
        QuaRAD.run_model(
                scenario_dir,X_i_dispersion,varnames_dispersion,X_infection,varnames_infection,
                griddings=griddings,
                user_options=user_options,
                trajectories_only=trajectories_only,
                overwrite_trajectories=overwrite_trajectories,
                overwrite_gridding=overwrite_gridding)
        
    return ensemble_dir

def ensemble_notrajectrories(
        ensemble_dir,N_infection,emission_type='talk',excel_filename='../input/inputs.xlsx',
        overwrite_gridding=True,):
    scenario_dirs = utils.get_scenario_dirs(ensemble_dir)
    for ii,scenario_dir in enumerate(scenario_dirs):
        X_i_dispersion = np.array([])
        varnames_dispersion = []
        idx, = np.where([e == '/' for e in ensemble_dir])
        idx_, = np.where([e == '_' for e in ensemble_dir])
        ensemble_name = ensemble_dir[(idx[-2]+1):(idx_[-1]-1)]
        
        options = pickle.load(open(scenario_dir + '00_info.pkl','rb'))
        X_infection,varnames_infection = get_inputs__ensemble__infection(
                ensemble_name, N_infection, fixed_vars={'S_source':1.}, 
                emission_type=emission_type, excel_filename=excel_filename)
        QuaRAD.run_model(
                scenario_dir,X_i_dispersion,varnames_dispersion,X_infection,varnames_infection,
                user_options=options,
                trajectories_only=False,
                overwrite_trajectories=False,
                overwrite_gridding=overwrite_gridding)
        
def get_print_dir(scenario_name,dir_all,get_next_dir=True):
    all_dirs = os.listdir(dir_all)
    these_dirs = [dirname for dirname in all_dirs if dirname.startswith(scenario_name)]
    nums = []
    if len(these_dirs) == 0:
        num = 0
    else:
        for onedir in these_dirs:
            try:
                nums.append(int(onedir[-2:]))
            except ValueError:
                print('do not append num from dir:', onedir)
        if len(nums) == 0:
            num = 0
        else:
            if get_next_dir:
                num = int(max(nums) + 1)
            else:
                num = int(max(nums))
    print_dir = dir_all + scenario_name + '__' + str(num).zfill(2) + '/'
    return print_dir

# =============================================================================
# get inputs -- one scenario
# =============================================================================
def get_inputs__onescenario(scenario_name, excel_filename='../input/inputs.xlsx', fixed_vars={}, emission_type='talk'):
    # fixed_vars is a dictionary
    df = pd.read_excel(excel_filename, sheet_name=scenario_name)
    df = df.dropna(how='all')
    all_varnames = df['Variable'].to_numpy()
    X = df['value'].to_numpy()
    for ii,fixed_varname in enumerate(fixed_vars):
        if any(all_varnames == fixed_varname):
            idx_vv, = np.where(all_varnames==fixed_varname)
            X[idx_vv] = fixed_vars[fixed_varname]
    return X, all_varnames
    

def get_inputs__ensemble(ensemble_name, N_samples, excel_filename='../input/inputs.xlsx',fixed_vars={}, emission_type='talk'):
    # fixed_vars is a dictionary
    df = pd.read_excel(excel_filename, sheet_name='ensemble_' + ensemble_name)
    all_varnames = df['Variable'].to_numpy()
    all_dists = df['Distribution'].to_numpy()
    all_param1 = df['min or mu'].to_numpy()
    all_param2 = df['max or sigma'].to_numpy()    
    
    varnames = np.array(
            ['volfrac_aero','rho_aero','tkappa',
             'Tv_inf','S_inf','S_source','Tv_source',
             'D_mouth','u0_' + emission_type,
             'ug_inf','vg_inf',
             'N_b_' + emission_type,'N_l_' + emission_type,'N_o_' + emission_type,
             'mu_b_' + emission_type,'mu_l_' + emission_type,'mu_o_' + emission_type,
             'sig_b_' + emission_type,'sig_l_' + emission_type,'sig_o_' + emission_type,
             'acr','L','W','H','x_co2_inf','x_co2_source','K',
             'vol_inhale','inhalation_rate_bpm','z0_source',
             'dNdt_cov_' + emission_type,'COV_fraction_coarse',
             'FE_out_relative_std','FE_in_relative_std',
             'dep_coeff_a','dep_coeff_b','dep_coeff_c','dep_coeff_d',
             'E_a','A_eff','A_sol'])
    
    lhd = lhs(len(varnames), samples=N_samples)
    X = np.zeros(lhd.shape)
    
    vv = np.where(varnames == 'N_b_' + emission_type)[0][0]
    param1 = all_param1[all_varnames=='N_b_' + emission_type][0]
    param2 = all_param2[all_varnames=='N_b_' + emission_type][0]
    dist = all_dists[all_varnames=='N_b_' + emission_type][0]
    N_b = get_sample(lhd[:,vv],dist,param1,param2)
    
    vv = np.where(varnames == 'N_o_' + emission_type)[0][0]
    param1 = all_param1[all_varnames=='frac_o_' + emission_type][0]
    param2 = all_param2[all_varnames=='frac_o_' + emission_type][0]
    dist = all_dists[all_varnames=='frac_o_' + emission_type][0]
    frac_o = get_sample(lhd[:,vv],dist,param1,param2)
    
    vv = np.where(varnames == 'N_l_' + emission_type)[0][0]    
    param1 = all_param1[all_varnames=='N_l_' + emission_type][0]
    param2 = all_param2[all_varnames=='N_l_' + emission_type][0]
    dist = all_dists[all_varnames=='N_l_' + emission_type][0]
    N_l = get_sample(lhd[:,vv],dist,param1,param2)
    
    for vv,varname in enumerate(varnames):
        if varname == 'N_o_' + emission_type:
            X[:,vv] = frac_o/(1-frac_o)*(N_b+N_l)
        elif varname == 'N_b_' + emission_type:
            X[:,vv] = N_b
        elif varname == 'N_l_' + emission_type:
            X[:,vv] = N_l
        else:
            param1 = all_param1[all_varnames==varname][0]
            param2 = all_param2[all_varnames==varname][0]
            dist = all_dists[all_varnames==varname][0]
            X[:,vv] = get_sample(lhd[:,vv],dist,param1,param2)
            
    for ii,fixed_varname in enumerate(fixed_vars):
        if any(varnames == fixed_varname):
            idx_vv, = np.where(varnames==fixed_varname)
            X[:,idx_vv] = fixed_vars[fixed_varname]
    return X, varnames

def get_inputs__ensemble__infection(ensemble_name, N_samples, fixed_vars={}, emission_type='talk',excel_filename='../input/inputs.xlsx',u0_min=0.5):
    # fixed_vars is a dictionary
    df = pd.read_excel(excel_filename, sheet_name='ensemble_' + ensemble_name)
    all_varnames = df['Variable'].to_numpy()
    all_dists = df['Distribution'].to_numpy()
    all_param1 = df['min or mu'].to_numpy()
    all_param2 = df['max or sigma'].to_numpy()    
    
    sample_varnames = np.array(
            ['muc_free','K_m','p_pfu','p_cell','dNdt_cov_' + emission_type])
    
    lhd = lhs(len(sample_varnames), samples=N_samples)
    X = np.zeros(lhd.shape)
    for vv,varname in enumerate(sample_varnames):
        param1 = all_param1[all_varnames==varname][0]
        param2 = all_param2[all_varnames==varname][0]
        dist = all_dists[all_varnames==varname][0]            
        X[:,vv] = get_sample(lhd[:,vv],dist,param1,param2)
        if varname == 'u0':
            idx_too_small, = np.where(X[:,vv]<u0_min)
            for ii in idx_too_small:
                while X[ii,vv]<u0_min:
                    X[ii,vv] = get_sample(np.random.rand(1),dist,param1,param2)
    
    for ii,fixed_varname in enumerate(fixed_vars):
        if any(sample_varnames == fixed_varname):
            idx_vv, = np.where(sample_varnames==fixed_varname)
            X[:,idx_vv] = fixed_vars[fixed_varname]
            
    return X, sample_varnames

def get_sample(cdf_val,dist,param1,param2):
    if dist == 'uniform':
        x = cdf_val*(param2 - param1) + param1
    elif dist == 'loguniform':
        x = 10**(cdf_val*(np.log10(param2) - np.log10(param1)) + np.log10(param1))
    elif dist == 'normal':
        x = param1 + param2*np.sqrt(2)*erfinv(2*cdf_val-1)
    elif dist == 'lognormal':
        x = 10**(np.log10(param1) + np.log10(param2)*np.sqrt(2)*erfinv(2*cdf_val-1))
    return x

def get_varlims(varname,excel_filename):
    df = pd.read_excel(excel_filename, sheet_name='input_pdfs')
    all_varnames = df['Variable'].to_numpy()
    these = all_varnames == varname
    dist = df['Distribution'].to_numpy()[these][0]
    param1 = df['min or mu'].to_numpy()[these][0]
    param2 = df['max or sigma'].to_numpy()[these][0]
    varlims = np.array([get_sample(0.01,dist,param1,param2),get_sample(0.99,dist,param1,param2)])
    return varlims

    
    
    