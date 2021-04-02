#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Laura Fierce
"""
import numpy as np
import os, pickle
from SALib.analyze import delta
import pandas as pd
from scipy.interpolate import interp1d


def get_select_cases(normconcs_dict,case_options,return_idx=True):
    where_true = np.ones(len(normconcs_dict.keys()))
    for cc in normconcs_dict.keys():
        for varname in case_options.keys():
            if normconcs_dict[cc]['inputs'][varname] != case_options[varname]:
                where_true[cc] = 0
    if return_idx:
        case_idx, = np.where(where_true)
        return case_idx
    else:
        return where_true


def get_ensemble_dat(varname,ensemble_dir,case_options,dispersion_type='both',dxval='all',last_scenario='none',idx_t=0):
    # varname is p_infect, dNdt_deposit, dNdt_inhale, or C
    scenario_dirs = get_scenario_dirs(ensemble_dir)
    if last_scenario!='none':
        scenario_dirs = scenario_dirs[:int(last_scenario+1)]
    
    for ss,scenario_dir in enumerate(scenario_dirs):
        infection_filename = scenario_dir + '03_infection.pkl'
        if os.path.exists(infection_filename):
            infection_dict = pickle.load(open(infection_filename,'rb'))
            dxs = infection_dict[0][0]['vs_x']['dxs']
            for jj in infection_dict.keys():
                case_idx = get_select_cases(infection_dict[jj],case_options,return_idx=True)
                if len(case_idx)>1:
                    raise Exception('more than onecase. adjust case_options!')
                elif len(case_idx) == 0:
                    raise Exception('no cases match!')
                else:
                    case_idx = case_idx[0]
                if dxval == 'all':
                    if ss == 0 and jj == 0:
                        if dispersion_type == 'both':
                            vardat = infection_dict[jj][case_idx]['vs_x'][dispersion_type][varname][idx_t,:,0,0]
                        elif dispersion_type == 'jet':
                            vardat = infection_dict[jj][case_idx]['vs_x'][dispersion_type][varname][:,0,0]
                        elif dispersion_type == 'reflections':
                            vardat = np.ones(len(dxs))*infection_dict[jj][case_idx]['vs_x'][dispersion_type][varname][idx_t]
                        elif dispersion_type == 'well_mixed':
                            vardat = np.ones(len(dxs))*infection_dict[jj][case_idx]['vs_x'][dispersion_type][varname][idx_t]
                    else:
                        if dispersion_type == 'both':
                            vardat = np.vstack([vardat,infection_dict[jj][case_idx]['vs_x'][dispersion_type][varname][idx_t,:,0,0]])
                        elif dispersion_type == 'jet':
                            vardat = np.vstack([vardat,infection_dict[jj][case_idx]['vs_x'][dispersion_type][varname][:,0,0]])
                        elif dispersion_type == 'reflections':
                            vardat = np.vstack([vardat,np.ones(len(dxs))*infection_dict[jj][case_idx]['vs_x'][dispersion_type][varname][idx_t]])
                        elif dispersion_type == 'well_mixed':
                            vardat = np.vstack([vardat,np.ones(len(dxs))*infection_dict[jj][case_idx]['vs_x'][dispersion_type][varname][idx_t]])
                    dxs_out = dxs
                else:
                    idx, = np.where([dx == dxval for dx in dxs])
                    if ss == 0 and jj == 0:
                        if dispersion_type == 'both':
                            vardat = infection_dict[jj][case_idx]['vs_x'][dispersion_type][varname][idx_t,idx,0,0]
                        elif dispersion_type == 'jet':
                            vardat = infection_dict[jj][case_idx]['vs_x'][dispersion_type][varname][idx,0,0]
                        elif dispersion_type == 'reflections':
                            vardat = infection_dict[jj][case_idx]['vs_x'][dispersion_type][varname][idx_t]
                        elif dispersion_type == 'well_mixed':
                            vardat = infection_dict[jj][case_idx]['vs_x'][dispersion_type][varname][idx_t]
                    else:
                        if dispersion_type == 'both':
                            vardat = np.hstack([vardat,infection_dict[jj][case_idx]['vs_x'][dispersion_type][varname][idx_t,idx,0,0]])
                        elif dispersion_type == 'jet':
                            vardat = np.hstack([vardat,infection_dict[jj][case_idx]['vs_x'][dispersion_type][varname][idx,0,0]])
                        elif dispersion_type == 'reflections':
                            vardat = np.hstack([vardat,infection_dict[jj][case_idx]['vs_x'][dispersion_type][varname][idx_t]])
                        elif dispersion_type == 'well_mixed':
                            vardat = np.hstack([vardat,infection_dict[jj][case_idx]['vs_x'][dispersion_type][varname][idx_t]])
                    dxs_out = dxs[idx]
    return dxs_out, vardat

def get_ensemble_dat__vs_t(varname,ensemble_dir,case_options,dxval,dispersion_type='both',last_scenario='none'):
    # varname is p_infect, dNdt_deposit, dNdt_inhale, or C
    scenario_dirs = get_scenario_dirs(ensemble_dir)
    if last_scenario!='none':
        scenario_dirs = scenario_dirs[:int(last_scenario+1)]
    
    for ss,scenario_dir in enumerate(scenario_dirs):
        infection_filename = scenario_dir + '03_infection.pkl'
        if os.path.exists(infection_filename):
            infection_dict = pickle.load(open(infection_filename,'rb'))
            dxs = infection_dict[0][0]['vs_xt']['dxs']
            ts_durration = infection_dict[0][0]['vs_xt']['ts_durration']
            for jj in infection_dict.keys():
                case_idx = get_select_cases(infection_dict[jj],case_options,return_idx=True)
                if len(case_idx)>1:
                    raise Exception('more than onecase. adjust case_options!')
                elif len(case_idx) == 0:
                    raise Exception('no cases match!')
                else:
                    case_idx = case_idx[0]
                if dispersion_type == 'both':                    
                    onevardat = infection_dict[jj][case_idx]['vs_xt'][dispersion_type][varname][:,:,0,0]
                elif dispersion_type == 'jet':
                    onevardat = infection_dict[jj][case_idx]['vs_xt'][dispersion_type][varname][:,0,0]
                elif dispersion_type == 'reflections':
                    onevardat = np.ones(len(dxs))*infection_dict[jj][case_idx]['vs_xt'][dispersion_type][varname][:]
                elif dispersion_type == 'well_mixed':
                    onevardat = np.ones(len(dxs))*infection_dict[jj][case_idx]['vs_xt'][dispersion_type][varname][:]
                
                idx_x, = np.where(abs(dxs - dxval)<1e-10)
                idx_x = idx_x[0]
                if ss == 0 and jj == 0:
                    vardat = onevardat[:,idx_x]
                else:
                    vardat = np.vstack([vardat,onevardat[:,idx_x]])
                    
    return ts_durration, vardat


def get_scenario_dat(varname,scenario_dir,case_options,dispersion_type='both',dxval='all',last_scenario='none'):
    # varname is p_infect, dNdt_deposit, dNdt_inhale, or C
    infection_filename = scenario_dir + '03_infection.pkl'
    if os.path.exists(infection_filename):
        infection_dict = pickle.load(open(infection_filename,'rb'))
        dxs = infection_dict[0][0]['vs_x']['dxs']
        
        for jj in infection_dict.keys():
            case_idx = get_select_cases(infection_dict[jj],case_options,return_idx=True)
            if len(case_idx)>1:
                raise Exception('more than onecase. adjust case_options!')
            else:
                case_idx = case_idx[0]
            if dxval == 'all':
                if jj == 0:
                    if dispersion_type == 'both':
                        vardat = infection_dict[jj][case_idx]['vs_x'][dispersion_type][varname][0,:,0,0]
                    elif dispersion_type == 'jet':
                        vardat = infection_dict[jj][case_idx]['vs_x'][dispersion_type][varname][:,0,0]
                    elif dispersion_type == 'reflections':
                        vardat = np.ones(len(dxs))*infection_dict[jj][case_idx]['vs_x'][dispersion_type][varname][0]
                else:
                    if dispersion_type == 'both':
                        vardat = np.vstack([vardat,infection_dict[jj][case_idx]['vs_x'][dispersion_type][varname][0,:,0,0]])
                    elif dispersion_type == 'jet':
                        vardat = np.vstack([vardat,infection_dict[jj][case_idx]['vs_x'][dispersion_type][varname][:,0,0]])
                    elif dispersion_type == 'reflections':
                        vardat = np.vstack([vardat,np.ones(len(dxs))*infection_dict[jj][case_idx]['vs_x'][dispersion_type][varname][0]])
                dxs_out = dxs
            else:
                idx, = np.where(dxs == dxval)
                idx = idx[0]
                if jj == 0:
                    if dispersion_type == 'both':
                        vardat = infection_dict[jj][case_idx]['vs_x'][dispersion_type][varname][0,idx,0,0]
                    elif dispersion_type == 'jet':
                        vardat = infection_dict[jj][case_idx]['vs_x'][dispersion_type][varname][idx,0,0]
                    elif dispersion_type == 'reflections':
                        vardat = infection_dict[jj][case_idx]['vs_x'][dispersion_type][varname][0]
                else:
                    if dispersion_type == 'both':
                        vardat = np.hstack([vardat,infection_dict[jj][case_idx]['vs_x'][dispersion_type][varname][0,idx,0,0]])
                    elif dispersion_type == 'jet':
                        vardat = np.hstack([vardat,infection_dict[jj][case_idx]['vs_x'][dispersion_type][varname][idx,0,0]])
                    elif dispersion_type == 'reflections':
                        vardat = np.hstack([vardat,infection_dict[jj][case_idx]['vs_x'][dispersion_type][varname][0]])
                dxs_out = dxs[idx]
    return dxs_out, vardat

def get_ensemble_input_dat(input_varname,ensemble_dir,last_scenario='none'):
    scenario_dirs = get_scenario_dirs(ensemble_dir)
    if last_scenario!='none':
        scenario_dirs = scenario_dirs[:int(last_scenario+1)]
    first_thing = True
    for ss,scenario_dir in enumerate(scenario_dirs):
        infection_filename = scenario_dir + '03_infection.pkl'
        if os.path.exists(infection_filename):
            infection_dict = pickle.load(open(infection_filename,'rb'))
            for jj in infection_dict.keys():
                if input_varname == 'V_room':          
                    one_vardat = infection_dict[jj][0]['inputs']['L']*infection_dict[jj][0]['inputs']['W']*infection_dict[jj][0]['inputs']['H']
                else:
                    one_vardat = infection_dict[jj][0]['inputs'][input_varname]

                if first_thing:
                    vardat = one_vardat
                    first_thing = False
                else:
                    vardat = np.vstack([vardat,one_vardat])
    return vardat

def get_input_dist(input_varname,ensemble_dir,ensemble_name='orig',excel_filename='inputs.xlsx'):
    df = pd.read_excel(excel_filename, sheet_name='ensemble_' + ensemble_name)
    all_varnames = df['Variable'].to_numpy()
    all_dists = df['Distribution'].to_numpy()
    idx, = np.where([var==input_varname for var in all_varnames])
    return all_dists[idx[0]]
def get_scenario_dirs(ensemble_dir):
    all_dirs = os.listdir(ensemble_dir)
    return [ensemble_dir + one_dir + '/' for one_dir in all_dirs if one_dir.isnumeric()]
    
def x_near__SA(ensemble_dir,these_varnames,thresh=1.05,varname='p_infect',last_scenario='none'):
    X, problem, labs = get_X(ensemble_dir,these_varnames,last_scenario='none')
    default_case_options = {'acr_enhancement':1.}
    dxs, vardat_default = get_ensemble_dat(
            varname,ensemble_dir,default_case_options,dispersion_type='both',dxval='all',last_scenario=last_scenario)
    dxs, vardat_reflections = get_ensemble_dat(
            varname,ensemble_dir,default_case_options,dispersion_type='reflections',dxval='all')
    x_near = np.zeros(vardat_default.shape[0])
    for ii in range(vardat_default.shape[0]):
        x_near[ii] = interp1d((vardat_default/vardat_reflections)[ii,:],dxs,fill_value='extrapolate')(thresh)
    
    idx, = np.where(~np.isnan(x_near))
    output_delta = delta.analyze(problem,X[idx,:],x_near[idx])
    return output_delta
    
def get_X(ensemble_dir,varnames,last_scenario='none',emission_type='talk'):
    vardat = []
    bounds = []
    labs = []
    dists = []
    for varname in varnames:
        if varname == 'dNdt_cov':
            dNdt_cov = get_ensemble_input_dat('dNdt_cov',ensemble_dir,last_scenario=last_scenario)
            vardat.append(dNdt_cov[:,0])
            bounds.append([min(vardat[-1]),max(vardat[-1])])
            labs.append('virion expiration rate')
            dist = get_input_dist('dNdt_cov_' + emission_type,ensemble_dir,ensemble_name='orig',excel_filename='inputs.xlsx')
#            labs.append('virion expiration rate,\nfine particles')
#            labs.append(r'\frac{dN_{\mathrm{v}}{dt}')
        elif varname == 'frac_cov_fine':
            w_cov = get_ensemble_input_dat('w_cov',ensemble_dir,last_scenario=last_scenario) 
            vardat.append(np.sum(w_cov[:,:-2],axis=1))
#            COV_frac_coarse = get_ensemble_input_dat('COV_fraction_coarse',ensemble_dir,last_scenario=last_scenario)             
            bounds.append([min(vardat[-1]),max(vardat[-1])])
            dist = get_input_dist(varname,ensemble_dir,ensemble_name='orig',excel_filename='inputs.xlsx')
            labs.append(varname)
            
        elif varname == 'dNdt_cov_fine':
            w_cov = get_ensemble_input_dat('w_cov',ensemble_dir,last_scenario=last_scenario) 
            dNdt_cov = get_ensemble_input_dat('dNdt_cov',ensemble_dir,last_scenario=last_scenario)        
            vardat.append(dNdt_cov[:,0]*np.sum(w_cov[:,:-2],axis=1))
            bounds.append([min(vardat[-1]),max(vardat[-1])])
            labs.append('virion expiration rate,\nfine particles')            
            dist = get_input_dist('dNdt_cov_' + emission_type,ensemble_dir,ensemble_name='orig',excel_filename='inputs.xlsx')
        elif varname == 'dNdt_cov_coarse':
            w_cov = get_ensemble_input_dat('w_cov',ensemble_dir,last_scenario=last_scenario) 
            dNdt_cov = get_ensemble_input_dat('dNdt_cov',ensemble_dir,last_scenario=last_scenario)        
            vardat.append(dNdt_cov[:,0]*np.sum(w_cov[:,-2:],axis=1))           
            bounds.append([min(vardat[-1]),max(vardat[-1])])            
            dist = get_input_dist('dNdt_cov_' + emission_type,ensemble_dir,ensemble_name='orig',excel_filename='inputs.xlsx')
            labs.append('virion expiration rate,\ncoarse particles')            
        elif varname == 'p_1':
            p_pfu = get_ensemble_input_dat('p_pfu',ensemble_dir,last_scenario=last_scenario) 
            p_cell = get_ensemble_input_dat('p_cell',ensemble_dir,last_scenario=last_scenario)
            vardat.append(p_pfu[:,0]*p_cell[:,0])
            bounds.append([min(vardat[-1]),max(vardat[-1])])            
            dist = 'lognormal'
            labs.append('$p_1$')
        elif varname == 'p_pfu':
            p_pfu = get_ensemble_input_dat('p_pfu',ensemble_dir,last_scenario=last_scenario) 
            vardat.append(p_pfu[:,0])
            bounds.append([min(vardat[-1]),max(vardat[-1])])            
            labs.append('$p_{\mathrm{pfu}}$')
            dist = get_input_dist(varname,ensemble_dir,ensemble_name='orig',excel_filename='inputs.xlsx')
        elif varname == 'p_cell':
            p_cell = get_ensemble_input_dat('p_cell',ensemble_dir,last_scenario=last_scenario) 
            vardat.append(p_cell[:,0])
            bounds.append([min(vardat[-1]),max(vardat[-1])])            
            labs.append('$p_{\mathrm{cell}}$')
            dist = get_input_dist(varname,ensemble_dir,ensemble_name='orig',excel_filename='inputs.xlsx')            
        elif varname == 'muc_free':
            muc_free = get_ensemble_input_dat('muc_free',ensemble_dir,last_scenario=last_scenario)
            vardat.append(muc_free[:,0])
            bounds.append([min(vardat[-1]),max(vardat[-1])])            
            labs.append('$[\mathrm{Muc}_{\mathrm{free}}]$')
            dist = get_input_dist(varname,ensemble_dir,ensemble_name='orig',excel_filename='inputs.xlsx')
        elif varname == 'u0':
            u0 = get_ensemble_input_dat('u0',ensemble_dir,last_scenario=last_scenario) 
            vardat.append(u0[:,0])
            bounds.append([min(vardat[-1]),max(vardat[-1])])            
            labs.append('$u_{0}$')
            dist = get_input_dist('u0_' + emission_type,ensemble_dir,ensemble_name='orig',excel_filename='inputs.xlsx')
        elif varname == 'ug_inf':
            ug_inf = get_ensemble_input_dat('ug_inf',ensemble_dir,last_scenario=last_scenario) 
            vardat.append(ug_inf[:,0]) 
            bounds.append([min(vardat[-1]),max(vardat[-1])])
            labs.append('$u_{{\mathrm{g},\infty}}$')
            dist = get_input_dist(varname,ensemble_dir,ensemble_name='orig',excel_filename='inputs.xlsx')
        elif varname == 'acr':
            acr = get_ensemble_input_dat('acr',ensemble_dir,last_scenario=last_scenario) 
            vardat.append(acr[:,0])
            bounds.append([min(vardat[-1]),max(vardat[-1])])            
            labs.append('air change rate')
            dist = get_input_dist(varname,ensemble_dir,ensemble_name='orig',excel_filename='inputs.xlsx')            
        elif varname == 'V_room':
            L = get_ensemble_input_dat('L',ensemble_dir,last_scenario=last_scenario)
            W = get_ensemble_input_dat('W',ensemble_dir,last_scenario=last_scenario)
            H = get_ensemble_input_dat('H',ensemble_dir,last_scenario=last_scenario)            
            vardat.append(L[:,0]*W[:,0]*H[:,0])
            bounds.append([min(vardat[-1]),max(vardat[-1])])
            labs.append('room volume')
            dist = get_input_dist('L',ensemble_dir,ensemble_name='orig',excel_filename='inputs.xlsx')
        elif varname == 'V_breathe':
            V_breathe = get_ensemble_input_dat('V_breathe',ensemble_dir,last_scenario=last_scenario)
            vardat.append(V_breathe[:,0])
            bounds.append([min(vardat[-1]),max(vardat[-1])])
            labs.append('breathing rate')
            dist = get_input_dist('vol_inhale',ensemble_dir,ensemble_name='orig',excel_filename='inputs.xlsx')
        elif varname == 'mu_cov':
            w_cov = get_ensemble_input_dat('w_cov',ensemble_dir,last_scenario=last_scenario)
            D = get_ensemble_input_dat('D',ensemble_dir,last_scenario=last_scenario)
            vardat.append(np.sum(np.log10(D)*w_cov,axis=1))
            bounds.append([min(vardat[-1]),max(vardat[-1])])
            labs.append('$\mu_{\text{v}}$')
            dist = get_input_dist('mu_l_' + emission_type,ensemble_dir,ensemble_name='orig',excel_filename='inputs.xlsx')            
        elif varname == 'mu_particle':
            w = get_ensemble_input_dat('w',ensemble_dir,last_scenario=last_scenario)             
            D = get_ensemble_input_dat('D',ensemble_dir,last_scenario=last_scenario)
            vardat.append(np.sum(np.log10(D)*w,axis=1))
            bounds.append([min(vardat[-1]),max(vardat[-1])])
            labs.append('geom. mean. diam.\nof expired particles')
            dist = get_input_dist('mu_l_' + emission_type,ensemble_dir,ensemble_name='orig',excel_filename='inputs.xlsx')
        elif varname == 'tkappa':
            tkappa = get_ensemble_input_dat('tkappa',ensemble_dir,last_scenario=last_scenario)
            vardat.append(tkappa[:,0])
            bounds.append([min(vardat[-1]),max(vardat[-1])])
            labs.append('aerosol $\kappa$')
            dist = get_input_dist(varname,ensemble_dir,ensemble_name='orig',excel_filename='inputs.xlsx')            
        elif varname == 'volfrac_aero':
            volfrac_aero = get_ensemble_input_dat('volfrac_aero',ensemble_dir,last_scenario=last_scenario)
            vardat.append(volfrac_aero[:,0])
            bounds.append([min(vardat[-1]),max(vardat[-1])])
            labs.append('initial solute fraction')
            dist = get_input_dist('volfrac_aero',ensemble_dir,ensemble_name='orig',excel_filename='inputs.xlsx')
        elif varname == 'FE_in_relative_std':
            FE_in_relative_std = get_ensemble_input_dat('FE_in_relative_std',ensemble_dir,last_scenario=last_scenario)
            vardat.append(FE_in_relative_std[:,0])
            bounds.append([min(vardat[-1]),max(vardat[-1])])
            labs.append('filtration effficiency, in')
            dist = get_input_dist(varname,ensemble_dir,ensemble_name='orig',excel_filename='inputs.xlsx')
        elif varname == 'FE_out_relative_std':
            FE_in_relative_std = get_ensemble_input_dat('FE_out_relative_std',ensemble_dir,last_scenario=last_scenario)
            vardat.append(FE_in_relative_std[:,0])
            bounds.append([min(vardat[-1]),max(vardat[-1])])
            labs.append('filtration effficiency, out')
            dist = get_input_dist(varname,ensemble_dir,ensemble_name='orig',excel_filename='inputs.xlsx')
        else:
            print('\"' + varname + '\" not defined')
            return [],{},[]
        if dist == 'normal':
            dists.append('norm')
        elif dist == 'lognormal':
            dists.append('lognorm')
        elif dist == 'uniform':
            dists.append('unif')
            
        problem = {'num_vars':len(varnames),'names':varnames,'bounds':bounds,'dists':dists}
    return np.array(vardat).transpose(), problem, labs