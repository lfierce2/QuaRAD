#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Laura Fierce
"""
# =============================================================================
# import dependencies
# =============================================================================
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import fmin, fsolve
from scipy.integrate import solve_ivp
from scipy.special import erfinv
import time
from scipy.constants import g, R, k
import os
import pickle

# =============================================================================
# define constants
# =============================================================================
rho_h2o = 1000.

# molecular weight of dry air, H2O, and CO2:
M_dry_air = 28.9647/1000. # kg/mol
M_h2o = 18./1000.
M_co2 = 44./1000.


Cp_h2o_vapor = 1000.
Cp_h2o = 4184. # J

Kg = 25.95/1e3 # W/(m*K)
Lv = 2450.9*1e3 # J/kg
D_inf = 0.242/100**2

# surface tenstion of water
sig_h2o = 71.97/1000.; # mN/m  J - Nm --- mN/m = mJ/m^2 = 1000 J/m^2


# =============================================================================
# RUN SCENARIO with the Quadrature-based Respiratory Aerosol Model (QuaRAM)
# =============================================================================
def run_model(
        scenario_dir,X_i_dispersion,varnames_dispersion,X_infection,varnames_infection,
        griddings=['vs_x'],emission_type='talk',excel_filename='../input/inputs.xlsx',
        user_options={},trajectories_only=False,overwrite_trajectories=False,overwrite_gridding=False):
    
    if scenario_dir[-1]!='/':
        scenario_dir = scenario_dir + '/'
        
    # check to see if print_dir exists, if not create
    if not os.path.exists(scenario_dir):
        os.mkdir(scenario_dir)
    
    store__00_info(scenario_dir,user_options)
    
    
    # STORE TRAJECTORIES DICTONARY
    print('starting 01_trajectories ...')    
    store__01_trajectories(
            scenario_dir,X_i_dispersion,varnames_dispersion,
            emission_type=emission_type,excel_filename=excel_filename,
            overwrite_trajectories=overwrite_trajectories)
    
    if not trajectories_only: # STORE NORMCONCS DICTONARY
        for gridding in griddings:
            print('starting 02_normconcs ...')
            store__02_normconcs(
                    scenario_dir,gridding,
                    emission_type=emission_type,excel_filename=excel_filename,
                    overwrite_gridding=overwrite_gridding)
        print('starting 03_infection ...')
        store__03_infection(
                X_infection,varnames_infection,scenario_dir,
                emission_type='talk',excel_filename='../input/inputs.xlsx')


# =============================================================================
# PART 0: SET OPTIONS, STORE INPUTS, CONSTRUCT QUADRATURE
# =============================================================================

def store__00_info(scenario_dir,user_options):
    # could add X, varnames, and problem here
    info_filename = scenario_dir + '00_info.pkl'
    options = set_options(user_options)
    pickle.dump(options,open(info_filename, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    return options

# =============================================================================
# set options
# =============================================================================
def set_options(user_options):
    options = set_default_options()
    for varname in user_options.keys():
        options[varname] = user_options[varname]
    return options
    
def set_default_options():
    options = dict()
    
    options['evaporates'] = [True]
    options['buoyancys'] = [True]
    options['vary_Tv_and_Ss'] = [False]    
    options['N_quads'] = [[1,3,2]]
    options['equil_assumptions'] = ['uniform'] # assumptions used to compute particle dry diameter, uniform: no equilibrium conditions assumed (assume uniform fraction of aerosol components at emission, prescribed by aero_frac, to get dry_diameter)
    options['vl_distributions'] = ['uniform_fine'] # assume virions in "fine" mode are distributed uniformly between b- and l-mode, assume "coarse" virions are in the o-mode 
    options['inactivates'] = [False]
#    options['acr_enhancements'] = [1.]
    options['acr_increases'] = [0.]
    options['other_param_dicts'] = [{}]
    
    options['p'] = 101325. # atmospheric pressure
    
    options['smask_types'] = ['none']
    options['rmask_types'] = ['none']
    
    options['t_max'] = 60. # end simulation for a quadrature point if simulation exceeds t_max
    options['x_max'] = 5. # end simulation for a quadratuer point if x exceeds x_max
    options['z_min'] = -4. # end simulation for a quadratuer point if z goes below z_min
    options['max_cpu_time'] = 120. # maximum time allowed for simulation (if it get's stuck on a slow particle, end the simualtion early -- and record that it ended early)
    options['tau_thresh'] = 1e-4 # relaxation time at which particles are assumed to "go with the flow"
    
    options['Nts_jet'] = int(1e4) # number of "puffs" used to resolve the steady-state jet
    options['tlim_thresh'] = 1e-5 # to compute the steady state concentration, we sum over puffs emitted over a certain time range. This threshold defines the portion of the plume that must be encompassed by that time span. If tlim_thresh = 0.01, 98% of the plume is within 
    options['tlims_log'] = [-10,3] # logarithm time range to span
    options['N_loops'] = 3 # number of loops used to refine the time interval
    options['Nts_reflections'] = int(1e3) # number of "puffs" used to resolve the reflections
    options['N_reflections'] = 1000 # number of reflections off walls (sum from -infty to infty approxiamted by sum from -N_reflections to N_reflections)
    
    options['Dp_noreflect'] = 30e-6 # particles larger than this are assumed not to reflect (we assume they settle to the ground and stay there)
    
    options['vaccine_efficacys'] = [0.]
    options['ts_durration'] = np.array([3600.])
    return options

def get_default_grid(gridding,z0_source=1.5):
    print('CHANGE XZ GRID BACK!')
    if gridding == 'vs_x':
#        dxs = np.linspace(0.,4.,21)
        dxs = np.linspace(0.,4.,11)        
        dxs[0] = 1./100.
        dys = np.array([0.])
        dzs = np.array([0.])
        ts_durration = np.array([3600.])
    elif gridding == 'vs_xz':
#        dxs = np.linspace(0.,4.,41)
#        dzs = np.linspace(-z0_source,0.5,21)        
        dxs = np.linspace(0.,4.,121)
        dzs = np.linspace(-z0_source,0.5,61)
        dxs[0] = 1./100.
        dys = np.array([0.])
        ts_durration = np.array([3600.])
    elif gridding == 'vs_xt':
        dxs = np.array([0.5,1.,2.,4.])
        dys = np.array([0.])
        dzs = np.array([0.])
        ts_durration = np.logspace(np.log10(3600),np.log10(86400),7)
        
    return dxs, dys, dzs, ts_durration
# =============================================================================
# 
#    FUNCTIONS TO CONSTRUCT QUADRATURE AND RETRIEVE INPUTS
#    
# =============================================================================

# =============================================================================
# construct quadrature approximation
# =============================================================================
def construct_quadrature(Ns,mus,sigs,volfrac_aero,modes=['b','l','o'],tkappa=0.65,rho_aero=1000.,N_quad=np.array([3,3,3]),equil_assumption='uniform'):

    nums_q = np.array([])
    wetdias_q = np.array([])
    modes_q = []
    for jj,(N,mu,sig,mode) in enumerate(zip(Ns,mus,sigs,modes)):
        x_q,w_q = np.polynomial.hermite.hermgauss(N_quad[jj])
        for q,(x,w) in enumerate(zip(x_q,w_q)):
            nums_q = np.append(nums_q,N*w/np.sqrt(np.pi))
            wetdias_q = np.append(wetdias_q,10**(mu+np.sqrt(2)*sig*x))
            modes_q += mode
    weights_q = nums_q/sum(nums_q)
    drydias_q = get_drydias(wetdias_q,weights_q,modes_q,volfrac_aero,equil_assumption=equil_assumption,rho_aero=rho_aero)
    tkappa_q = np.ones(weights_q.shape)*tkappa
    return wetdias_q, drydias_q, weights_q, modes_q, tkappa_q

def get_drydias(wetdias_q,weights_q,modes_q,volfrac_aero,equil_assumption='uniform',rho_aero=1000.,tkappa=0.65,Tv_source=310.15,S_source=1.):
    if equil_assumption == 'uniform':
        massfrac_q = np.ones(len(wetdias_q))*volfrac_aero
        drydias_q = (massfrac_q*rho_h2o)**(1/3)*wetdias_q/((massfrac_q*rho_h2o)**(1/3) + ((1-massfrac_q)*rho_aero)**(1/3))
    elif equil_assumption == 'equil':
        drydias_q = np.zeros(wetdias_q.shape)
        for qq in range(len(drydias_q)):
            drydias_q[qq] = get_dry_diameter__equil(wetdias_q[qq],tkappa,rho_aero,Tv_source,S_source)
    elif equil_assumption == 'lb_equil':
        drydias_q = np.zeros(wetdias_q.shape)
        idx_small, = np.where([this_mode=='l' or this_mode=='b' for this_mode in modes_q])
        dryvol_total = sum(np.pi/6*wetdias_q**3*volfrac_aero)
        for ii in idx_small:
            drydias_q[ii] = get_dry_diameter__equil(wetdias_q[ii],tkappa,rho_aero,Tv_source,S_source)
        dryvol_in_small = sum(np.pi/6*drydias_q[idx_small]**3)
        idx_big, = np.where([this_mode=='o' for this_mode in modes_q])
        if dryvol_in_small<=dryvol_total:
            drydias_q[idx_big] = (wetdias_q[idx_big]**3*(dryvol_total - dryvol_in_small)/(np.pi/6*wetdias_q[idx_big]**3))**(1/3)
    else:
        print('only coded for \'assumption == uniform\', \'assumption == equil\', \'assumption == lb_equil\' right now')
    return drydias_q

def get_dry_diameter__equil(D_wet,tkappa_aero,density_aero,Tv,S_env_initial):
    zero_this = lambda Dd: get_equilibrium_diameter(Dd,tkappa_aero,Tv,S_env_initial)[0] - D_wet
    D_dry = fsolve(zero_this,D_wet/2.)[0]
    return D_dry

def get_equilibrium_diameter(Dd,tkappa,Tv,S):
    # computes the equilibrium diameter for a droplet having dry diameter (Dd), hygroscopicity parameter (tkappa), and at an environmental supersaturation ration (S)
    zero_this = lambda D: get_droplet_saturation_ratio(D,Dd,tkappa,Tv) - S
    D = fsolve(zero_this,Dd)
    D = fsolve(zero_this,D)
    return D

def get_ws_cov(inputs_dict, vl_distribution='base'):
    ws = inputs_dict['w']
    Dps = inputs_dict['D']
    modes_q = inputs_dict['mode']
    COV_fraction_coarse = inputs_dict['COV_fraction_coarse']
    dVdts = ws*np.pi/6*Dps**3
    VL_cov_q = np.zeros(len(ws)) # not actual virions
    dNdt_cov_coarse = COV_fraction_coarse
    dNdt_cov_fine = 1. - COV_fraction_coarse
    
    dVdt_b = 0.
    dVdt_l = 0.
    dVdt_o = 0.
    for (dVdt,mode) in zip(dVdts,modes_q):
        if mode == 'b':
            dVdt_b += dVdt
        elif mode == 'l':
            dVdt_l += dVdt
        if mode == 'o':
            dVdt_o += dVdt
    
    for qq,mode in enumerate(modes_q):
        if vl_distribution=='base':
            if mode == 'b':
                VL_cov_q[qq] = 0.
            elif mode == 'l':
                VL_cov_q[qq] = dNdt_cov_fine/dVdt_l
            elif mode == 'o':
                VL_cov_q[qq] = dNdt_cov_coarse/dVdt_o
        elif vl_distribution=='uniform_fine':
            if mode == 'b':
                VL_cov_q[qq] = dNdt_cov_fine/(dVdt_b + dVdt_l)
            elif mode == 'l':
                VL_cov_q[qq] = dNdt_cov_fine/(dVdt_b + dVdt_l)
            elif mode == 'o':
                VL_cov_q[qq] = dNdt_cov_coarse/dVdt_o
        elif vl_distribution=='uniform':
            VL_cov_q[qq] = (dNdt_cov_fine+dNdt_cov_coarse)/sum(dVdts)
    ws_cov = VL_cov_q*dVdts/(sum(VL_cov_q*dVdts))
    return ws_cov

# =============================================================================
#   get inputs all cases
# =============================================================================
def get_inputs___trajectories__allcases(X_i,all_varnames,options,
        emission_type='talk',excel_filename='../input/inputs.xlsx'):
    
    inputs__allcases = dict()
    cc = 0
    
    other_option_vars = ['max_cpu_time','tau_thresh','t_max','x_max','z_min','p','ts_durration']
    
    for evaporate in options['evaporates']:
        for buoyancy in options['buoyancys']:
            for vary_Tv_and_S in options['vary_Tv_and_Ss']:
                for N_quad in options['N_quads']:
                    for equil_assumption in options['equil_assumptions']:
                        for smask_type in options['smask_types']:
    #                        for acr_enhancement in options['acr_enhancements']:
                            for acr_increase in options['acr_increases']:                        
                                if len(options['other_param_dicts'])>0:
                                    for other_param_dict in options['other_param_dicts']:
                                        options_onecase = {
                                                'evaporate':evaporate,
                                                'buoyancy':buoyancy,
                                                'vary_Tv_and_S':vary_Tv_and_S,
                                                'N_quad':N_quad,
                                                'equil_assumption':equil_assumption,
                                                'smask_type':smask_type,
                                                'acr_increase':acr_increase,                                            
    #                                            'acr_enhancement':acr_enhancement,
                                                'other_param_dict':other_param_dict}
                                        for var in other_option_vars:
                                            options_onecase[var] = options[var]
                                        
                                        inputs__allcases[cc] = get_inputs_dict__trajectories__onecase(
                                                X_i,all_varnames,options_onecase,
                                                excel_filename=excel_filename,emission_type=emission_type)
#                                        print(cc,inputs__allcases[cc]['acr'])
                                        cc += 1
                                else:
                                    options_onecase = {
                                            'evaporate':evaporate,
                                            'buoyancy':buoyancy,
                                            'vary_Tv_and_S':vary_Tv_and_S,
                                            'N_quad':N_quad,
                                            'equil_assumption':equil_assumption,
                                            'smask_type':smask_type,
                                            'acr_increase':acr_increase,                                        
    #                                        'acr_enhancement':acr_enhancement,
                                            'other_param_dict':{}}             
                                    for var in other_option_vars:
                                            options_onecase[var] = options[var]
                                            
                                    inputs__allcases[cc] = get_inputs_dict__trajectories__onecase(
                                            X_i,all_varnames, options_onecase,
                                            excel_filename=excel_filename,emission_type=emission_type)
                                    cc += 1
    return inputs__allcases

def get_inputs___deposition__allcases(trajectories_dict,options,excel_filename='../input/inputs.xlsx'):
    
    inputs__allcases = dict()
    
    other_option_vars = [
            'max_cpu_time','tau_thresh','t_max','x_max','z_min','p',
            'Nts_jet','Nts_reflections','Dp_noreflect','tlim_thresh',
            'N_reflections','tlims_log','N_loops']
    
    ccc = int(0)
    for cc in trajectories_dict.keys():
        trajectory_inputs_dict = trajectories_dict[cc]['inputs']
        smask_type = trajectory_inputs_dict['smask_type']
        for rmask_type in options['rmask_types']:
            if rmask_type == 'none' or rmask_type == smask_type or smask_type == 'none':
                for inactivate in options['inactivates']:
                    for vl_distribution in options['vl_distributions']:
                        options_onecase = dict()
                        options_onecase['rmask_type'] = rmask_type
                        options_onecase['inactivate'] = inactivate
                        options_onecase['vl_distribution'] = vl_distribution
                        for var in other_option_vars:
                            options_onecase[var] = options[var]
                        
                        inputs__onecase = get_inputs_dict__deposition__onecase(
                                trajectory_inputs_dict,options_onecase,excel_filename=excel_filename)
                        inputs__allcases[ccc] = inputs__onecase.copy()
                        inputs__allcases[ccc]['trajectories_case'] = cc
                        ccc = int(ccc + 1)
    return inputs__allcases

def get_these_cases(dict_of_inputs,case_params):
    is_one = []
    for cc in dict_of_inputs.keys():
        yep = True
        for param_name in case_params.keys():
            if dict_of_inputs[cc]['inputs'][param_name] != case_params[param_name]:
                yep = False
        is_one.append(yep)
    idx_cases, = np.where(is_one)
    return idx_cases

# =============================================================================
# functions to get inputs for trajetories
# =============================================================================
def get_inputs_dict__trajectories__onecase(
        X_i,all_varnames,options_onecase, # the optional arguments should all be stored in inputs_dict
        excel_filename='../input/inputs.xlsx',emission_type='talk'):
    modes = ['b','l','o']
    Ns = np.array([])
    mus = np.array([])
    sigs = np.array([])
    
    for jj,mode in enumerate(modes):
        Ns = np.append(Ns, X_i[[var=='N_' + mode + '_' + emission_type for var in all_varnames]][0])
        mus = np.append(mus, np.log10(X_i[[var=='mu_' + mode + '_' + emission_type for var in all_varnames]][0]))
        sigs = np.append(sigs, np.log10(X_i[[var=='sig_' + mode + '_' + emission_type for var in all_varnames]][0]))    
        
    other_varnames = ['D_mouth','ug_inf','volfrac_aero','rho_aero','tkappa',
                      'Tv_source','S_source','x_co2_source','Tv_inf','S_inf','x_co2_inf',
                      'acr','L','W','H','x_co2_inf','x_co2_source','vol_inhale',#'K',
                      'COV_fraction_coarse','FE_out_relative_std','FE_in_relative_std',
                      'E_a','A_eff','A_sol','z0_source',
                      'dep_coeff_a','dep_coeff_b','dep_coeff_c','dep_coeff_d']
    inputs_dict = dict()
    inputs_dict['emission_type'] = emission_type
    inputs_dict['u0'] = X_i[[var == 'u0_' + emission_type for var in all_varnames]][0]  
    inputs_dict['dNdt_cov'] = X_i[[var == 'dNdt_cov_' + emission_type for var in all_varnames]][0]    
    inputs_dict['Ns'] = Ns
    inputs_dict['mus'] = mus
    inputs_dict['sigs'] = sigs
    inputs_dict['ts_durration'] = options_onecase['ts_durration']
    
    other_option_vars = [
            'evaporate','buoyancy','vary_Tv_and_S','equil_assumption','N_quad','smask_type',
            'max_cpu_time','tau_thresh','t_max','x_max','z_min','p','acr_increase']#'acr_enhancement']
    
    for varname in other_option_vars:
        inputs_dict[varname] = options_onecase[varname]
    for varname in other_varnames:
        inputs_dict[varname] = X_i[[var == varname for var in all_varnames]][0]

    if inputs_dict['dep_coeff_a']>-0.00128:
        inputs_dict['dep_coeff_a'] = -0.00128
    if inputs_dict['dep_coeff_a']<-0.00839:
        inputs_dict['dep_coeff_a'] = -0.00839
    if inputs_dict['dep_coeff_b']>-11.9:
        inputs_dict['dep_coeff_b'] = -11.9
    if inputs_dict['dep_coeff_b']<-20.9:
        inputs_dict['dep_coeff_b'] = -20.9
    if inputs_dict['dep_coeff_c']<0.44:
        inputs_dict['dep_coeff_c'] = 0.44
    if inputs_dict['dep_coeff_c']>0.56:
        inputs_dict['dep_coeff_c'] = 0.56
    if inputs_dict['dep_coeff_d']>-0.19:
        inputs_dict['dep_coeff_d'] = -0.19
    if inputs_dict['dep_coeff_d']<-0.37:
        inputs_dict['dep_coeff_d'] = -0.37
    
#    u0__masked = inputs_dict['u0']#
    u0__masked = get_u0_mask(inputs_dict['u0'],mask_type=options_onecase['smask_type'],excel_filename=excel_filename,u0_reduction=None)
    D_mouth__masked = inputs_dict['D_mouth']*np.sqrt(inputs_dict['u0']/u0__masked)
    inputs_dict['u0'] = u0__masked
    inputs_dict['D_mouth'] = D_mouth__masked
    inputs_dict['dNdt_particles'] = sum(Ns)
    volfrac_aero, tkappa, rho_aero = get_select_inputs(inputs_dict,['volfrac_aero', 'tkappa', 'rho_aero'])
    wetdias_q, drydias_q, weights_q, modes_q, tkappa_q = construct_quadrature(
            Ns,mus,sigs,volfrac_aero,modes=modes,tkappa=tkappa,rho_aero=rho_aero,
            N_quad=options_onecase['N_quad'],equil_assumption=options_onecase['equil_assumption'])
    equildias_q = get_equilibrium_diameter(drydias_q,tkappa_q,inputs_dict['Tv_inf'],inputs_dict['S_inf'])
    part_input_varnames = [
            'D', 'Dd', 'w', 'mode', 'tkappa','De']
    part_input_varvals = [
            wetdias_q, drydias_q, weights_q, modes_q, tkappa_q,equildias_q]
    for varname,varval in zip(part_input_varnames,part_input_varvals):
        inputs_dict[varname] = varval
    
    acr_orig = inputs_dict['acr']
#    inputs_dict['acr'] = acr_orig*inputs_dict['acr_enhancement']
    inputs_dict['acr'] = acr_orig + inputs_dict['acr_increase']
    
    other_param_dict = options_onecase['other_param_dict']
    for varname in other_param_dict.keys():
        if varname == 'tkappa':
            inputs_dict['tkappa'] = other_param_dict[varname]*np.ones(len(tkappa_q))
        else:
            inputs_dict[varname] = other_param_dict[varname]
    inputs_dict['V_breathe'] = X_i[[var=='vol_inhale' for var in all_varnames]][0]*X_i[[var=='inhalation_rate_bpm' for var in all_varnames]][0]/60.
    
    return inputs_dict

def get_select_inputs(inputs_dict,some_varnames):
    select_inputs = ()
    for varname in some_varnames:
        select_inputs += (inputs_dict[varname],)
        
    return select_inputs

# =============================================================================
# functions to compute inputs for deposition
# =============================================================================
def get_inputs_dict__deposition__onecase(
    trajectory_inputs_dict,options,excel_filename='../input/inputs.xlsx'):
    
    deposition_inputs_dict = trajectory_inputs_dict
    
    for var in options.keys():
        deposition_inputs_dict[var] = options[var]
    
    ws_cov = get_ws_cov(deposition_inputs_dict, vl_distribution=options['vl_distribution'])
    deposition_inputs_dict['w_cov'] = ws_cov
    
    return deposition_inputs_dict


# =============================================================================
# PART 1: FUNCTIONS TO COMPUTE TRAJECTORIES
#   propogate the evoltuion of quadarature points
#   output: data frames with
#       mp(t), Tp(t) due to evaporation
#       x0(t), y0(t), z0(t) (as well as dx/dt, dy/dt, dz/dt) of "center" due to transport and settling
# =============================================================================

# =============================================================================
# compute trajectories, store in dfs, sstore those in dictionary
# =============================================================================
    
def store__01_trajectories(
        scenario_dir,X_i,all_varnames,
        emission_type='talk',excel_filename='../input/inputs.xlsx',
        overwrite_trajectories=False):
    
    start_time = time.process_time()
    options = pickle.load(open(scenario_dir + '00_info.pkl','rb'))
    trajectories_filename = scenario_dir + '01_trajectories.pkl'
    
    # check to see if trajectories dict has already been stored
    if overwrite_trajectories or (not os.path.exists(trajectories_filename)):
        # compute trajectories for this scenario, all cases
        trajectories_dict = get_trajectories(
            X_i,all_varnames,options,
            excel_filename=excel_filename, emission_type=emission_type)

        # store trajectories
        pickle.dump(trajectories_dict, open(trajectories_filename, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    else:
        trajectories_dict = pickle.load(open(trajectories_filename, 'rb'))
    print('another trajectory finished!', scenario_dir, 'process_time:', time.process_time() - start_time)
    return trajectories_dict

def get_trajectories(
        X_i,all_varnames,options,
        emission_type='talk',excel_filename='../input/inputs.xlsx'):
    
    inputs__allcases = get_inputs___trajectories__allcases(
            X_i,all_varnames,options,
            emission_type=emission_type,excel_filename = excel_filename)
    
    trajectories__onescenario = dict()
    for cc in inputs__allcases.keys():
        inputs = inputs__allcases[cc]
        trajectories__onescenario[cc] = get_trajectories__onecase(inputs)
        
    return trajectories__onescenario

# =============================================================================
# functions to compute trajectories for single case
# =============================================================================
def get_trajectories__onecase(inputs):
    trajectories__onecase = {'inputs': inputs}
    for qq in range(len(inputs['w'])):
        df_pos, df_evap, any_stopped_early = get_trajectory_dfs(qq,inputs)
        trajectories__onecase[qq] = {'pos':df_pos,'evap':df_evap,'stopped_early':any_stopped_early}
    return trajectories__onecase

def get_trajectory_dfs(qq,inputs):
    wetdias_q,drydias_q,tkappa_q,equildias_q = get_select_inputs(inputs,['D','Dd','tkappa','De']) 
    p = inputs['p']
    Dp0 = wetdias_q[qq]
    Dd = drydias_q[qq]
    tkappa = tkappa_q[qq]
    De = equildias_q[qq]
    u0,ug_inf,Tv_inf,S_inf,x_co2_inf,Tv_source,S_source,x_co2_source,rho_aero,D_mouth = get_select_inputs(inputs,['u0','ug_inf','Tv_inf','S_inf','x_co2_inf','Tv_source','S_source','x_co2_source','rho_aero','D_mouth'])    
    K = 0.
    max_cpu_time,t_max,tau_thresh,x_max,z_min = get_select_inputs(inputs,['max_cpu_time','t_max','tau_thresh','x_max','z_min'])
    evaporate,buoyancy,vary_Tv_and_S = get_select_inputs(inputs,['evaporate','buoyancy','vary_Tv_and_S'])
    
    any_stopped_early = False
    
    # compute centerline fun, with or without buoyancy
    if buoyancy: 
        z0_fun = lambda x0: get_z_centerline__nosedi(x0,u0,D_mouth,Tv_source,S_source,x_co2_source,Tv_inf,S_inf,x_co2_inf,p=p)
        s_fun = lambda x0: get_s(x0, z0_fun)
        dz0dx_fun = lambda x0: get_dz0dx(x0,u0,D_mouth,Tv_source,S_source,x_co2_source,Tv_inf,S_inf,x_co2_inf,p=p)
    else:
        z0_fun = lambda x0: 0.*x0
        s_fun = lambda x0: x0
        dz0dx_fun = lambda x0: 0.*x0
    
    #  store evaporation, without convection
    udiff_fun = lambda t: 0.
    
    if evaporate:
        ts_evap, mps_evap, Tps_evap, stopped_early = get_evaporation_sol(Dp0,Dd,tkappa,rho_aero,Tv_inf,S_inf,max_cpu_time=max_cpu_time,udiff_fun=udiff_fun,Tv_source=Tv_source,S_source=S_source,p=p,t_max=t_max,evaporate=True)
#        print('evap',qq,stopped_early)
        if stopped_early:
            any_stopped_early = True
        Dps = get_diameter_from_mass(Dd,mps_evap,rho_aero=rho_aero)
        Dp_fun = get_Dp_fun(ts_evap,Dps,fill_value=De,evaporate=True)
    else:
        Dp_fun = get_Dp_fun(np.array([0.,t_max]),np.array([Dp0,Dp0]),evaporate=False)
    
    #  store position without conctive feedback on evaporation
    ts_pos,x_pos,dxdt_pos,y_pos,dydt_pos,z_pos,dzdt_pos,udiff,stopped_early = get_position_sol(
            Dp_fun,z0_fun,dz0dx_fun,s_fun,u0,Dd,rho_aero,D_mouth,ug_inf,Tv_inf,S_inf,
            max_cpu_time=max_cpu_time,tau_thresh=tau_thresh,x_co2_inf=x_co2_inf,
            Tv_source=Tv_source,S_source=S_source,p=p,buoyancy=buoyancy,
            t_max=t_max,K=K,x_max=x_max,z_min=z_min)
#    print('pos',qq,stopped_early)
    if stopped_early:
        any_stopped_early = True
    

    x0_pos = np.array([get_x0(x,z,z0_fun,u0,Tv_inf,S_inf,Tv_source,S_source,D_mouth=D_mouth,p=p) for (x,z) in zip(x_pos,z_pos)])
    z0_pos = z0_fun(x0_pos)
    udiff_fun = lambda t: interp1d(ts_pos,udiff,fill_value='extrapolate')(t)
    
    if evaporate:
        # @AR: updated to spatially-varying Tv and S here xxx
        if vary_Tv_and_S:
            Tv_vs_t, S_vs_t = get_Tv_and_S(Tv_source,S_source,Tv_inf,S_inf,ts_pos,x_pos,y_pos,z_pos,z0_fun,u0,D_mouth=D_mouth,p=p)
            ts_evap, mps_evap, Tps_evap, stopped_early = get_evaporation_sol(Dp0,Dd,tkappa,rho_aero,Tv_vs_t,S_vs_t,max_cpu_time=max_cpu_time,udiff_fun=udiff_fun,Tv_source=Tv_source,S_source=S_source,p=p,t_max=t_max,evaporate=True)        
        else:
            ts_evap, mps_evap, Tps_evap, stopped_early = get_evaporation_sol(Dp0,Dd,tkappa,rho_aero,Tv_inf,S_inf,max_cpu_time=max_cpu_time,udiff_fun=udiff_fun,Tv_source=Tv_source,S_source=S_source,p=p,t_max=t_max,evaporate=True)
        if stopped_early:
            any_stopped_early = True
    else:
        ts_evap = np.array([0.,t_max])
        mp0 = get_mass_from_diameter(Dd,Dp0,rho_aero)
        mps_evap = np.array([mp0,mp0])
        Tps_evap = np.array([Tv_inf,Tv_inf])        
    
    Dps = get_diameter_from_mass(Dd,mps_evap,rho_aero=rho_aero)                
    Dp_fun = get_Dp_fun(ts_evap,Dps,fill_value=De,evaporate=evaporate)
    the_vars = [mps_evap,Tps_evap]
    the_varnames = ['mp','Tp']
    for ii,(data,varname) in enumerate(zip(the_vars,the_varnames)):
        if ii == 0:
            df_evap = pd.Series(data=data, index=ts_evap,name=varname)
        else:
            df_evap = pd.concat([df_evap,pd.Series(data=data, index=ts_evap,name=varname)],axis=1)
    
    ts_pos,x_pos,dxdt_pos,y_pos,dydt_pos,z_pos,dzdt_pos,udiff,stopped_early = get_position_sol(Dp_fun,z0_fun,dz0dx_fun,s_fun,u0,Dd,rho_aero,D_mouth,ug_inf,Tv_inf,S_inf,max_cpu_time=max_cpu_time,tau_thresh=tau_thresh,x_co2_inf=x_co2_inf,Tv_source=Tv_source,S_source=S_source,p=p,buoyancy=buoyancy,t_max=t_max,K=K)
    if stopped_early:
        any_stopped_early = True    
    

    x0_pos = np.array([get_x0(x,z,z0_fun,u0,Tv_inf,S_inf,Tv_source,S_source,D_mouth=D_mouth,p=p) for (x,z) in zip(x_pos,z_pos)])
    z0_pos = z0_fun(x0_pos)
    
    the_vars = [x_pos,y_pos,z_pos,dxdt_pos,dydt_pos,dzdt_pos,x0_pos,z0_pos]
    the_varnames = ['x','y','z','dxdt','dydt','dzdt','x0','z0']
    for ii,(data,varname) in enumerate(zip(the_vars,the_varnames)):
        if ii == 0:
            df_pos = pd.Series(data=data, index=ts_pos,name=varname)
        else:
            df_pos = pd.concat([df_pos,pd.Series(data=data, index=ts_pos,name=varname)],axis=1)
    return df_pos, df_evap, any_stopped_early


# =============================================================================
# solve for evaporation trajectories
# =============================================================================
def get_evaporation_sol(Dp0,Dd,tkappa,rho_aero,Tv_in,S_in,max_cpu_time=5.,udiff_fun=lambda t: 0.,Tv_source=310.15,S_source=1.,p=101325.,t_max=10.,evaporate=True):
    mp0 = get_mass_from_diameter(Dd,Dp0,rho_aero)
    q0 = np.array([mp0,Tv_source])
    stopped_early = False
    if evaporate:
        # replace S_inf, Tv_inf with varying...
        if callable(Tv_in):
#            Tv_inf = Tv_in(1e10)
#            S_inf = S_in(1e10)
            Tv_inf = Tv_in(100.)
            S_inf = S_in(100.)
            odesystem_evaporate = lambda t,q: get_odesystem_evaporate__callable(t,q,Dd,tkappa,rho_aero,Tv_in,S_in,udiff_fun=lambda t: 0., p=101325.)
        else:
            Tv_inf = Tv_in
            S_inf = S_in
            odesystem_evaporate = lambda t,q: get_odesystem_evaporate(t,q,Dd,tkappa,rho_aero,Tv_in,S_in,udiff_fun=lambda t: 0., p=101325.)            
            
        Dp_equil = get_equilibrium_diameter(Dd,tkappa,Tv_inf,S_inf)[0]
        mp_equil = get_mass_from_diameter(Dd,Dp_equil,rho_aero)
        
        stop_time = time.time() + max_cpu_time
        stop_condition = lambda t, q: 1.0*((q[0]-mp_equil)>(0e-3*mp_equil) and time.time()<stop_time)
        stop_condition.terminal = True

        ivp_output = solve_ivp(odesystem_evaporate,(0.,t_max),q0,events=stop_condition,method='RK23')
#        ivp_output = solve_ivp(odesystem_evaporate,(0.,t_max),q0,events=stop_condition,method='LSODA')
        evap_output = np.vstack([ivp_output['t'],ivp_output['y']]).transpose()
        
        if (evap_output[-1,1] == mp_equil):
            evap_output = np.vstack([evap_output,np.hstack([t_max,evap_output[-1,1:]])])
        elif (np.isnan(evap_output[-1,1])) and mp_equil == 0.:
            idx, = np.where(np.isnan(evap_output[:,1]))
            evap_output = evap_output[:(idx[0]-1),:]
            evap_output = np.vstack([evap_output,np.hstack([t_max,np.array([mp_equil,Tv_inf])])])
        elif (np.isnan(evap_output[-1,1])):
            idx, = np.where(np.isnan(evap_output[:,1]))
            evap_output = evap_output[:(idx[0]-1),:]
        ts = evap_output[:,0]
        mps = evap_output[:,1]
        Tps = evap_output[:,2]
        if len(mps) == 0:
            ts = np.array([0.,t_max])
            mps = mp0*np.ones(len(ts))
            Tps = Tv_inf*np.ones(len(ts))
            stopped_early == True
        elif abs(mps[-1] - mp_equil)>1e-5:
            stopped_early == True
    else:
        ts = np.array([0.,t_max])
        mps = mp0*np.ones(len(ts))
        Tps = Tv_inf*np.ones(len(ts))
        
    return ts, mps, Tps, stopped_early

def get_odesystem_evaporate(t,q,Dd,tkappa,rho_aero,Tv,Sv,udiff_fun=lambda t: 0., p=101325.):
    mp = q[0]
    Tp = q[1]
    Dp = get_diameter_from_mass(Dd,mp,rho_aero=rho_aero)
    
    lamb = 1.6 # constant between 1.6 and 2
    if abs(Tv**(2-lamb)-Tp**(2-lamb)) > 0:
        C_T = (Tv - Tp)/Tv**(lamb-1.)*(2-lamb)/(Tv**(2-lamb)-Tp**(2-lamb))
    else:
        C_T = 1.
    
    p_d = get_droplet_vapor_pressure(Dp,Dd,tkappa,Tp)
    p_v = get_vapor_pressure(Sv,Tv)
    udiff = udiff_fun(t)
    Re = get_particle_Re(udiff,Dp,Tv,Sv)
    Sc = get_particle_Sc(Dp,Tv,Sv)
    Pr = get_particle_Pr(Tv,Sv)
    Sh = 1+0.38*Re**(1/2)*Sc**(1/3) # chen et al.
#    Sh = 2+0.6*Re**(1/2)*Sc**(1/3) #  Ranz, W. E. and Marshall, W. R. Evaporation from Drops. Chemical Engineering Progress, 48:141-146, 173-180, 1952.; via https://en.wikipedia.org/wiki/Sherwood_number
    dmdt = 2*np.pi*p*Dp*M_h2o*D_inf*C_T*Sh/(R*Tv)*np.log((p-p_d)/(p-p_v))
#    Nu = 2 + 0.4*Re**(1/2)*Pr**(1/3) # McAllister, S., Chen, J-Y. and Carlos Fernandez-Pello, A. Fundamentals of Combustion Processes. Springer, 2011. ch. 8 p. 159; via https://en.wikipedia.org/wiki/Nusselt_number#cite_note-9
    Nu = 1 + 0.3*Re**(1/2)*Pr**(1/3) # Chen et al. 
    dTdt = (np.pi*Dp**2*Kg*(Tv-Tp)/(Dp/2)*Nu+Lv*(dmdt))/(mp*Cp_h2o)
    
    return np.array([dmdt, dTdt])

def get_odesystem_evaporate__callable(t,q,Dd,tkappa,rho_aero,Tv_in,Sv_in,udiff_fun=lambda t: 0., p=101325.):
    mp = q[0]
    Tp = q[1]
    Dp = get_diameter_from_mass(Dd,mp,rho_aero=rho_aero)
    
    Tv = Tv_in(t)
    Sv = Sv_in(t)
    if Sv>1.:
        Sv = 1.
    
    lamb = 1.6 # constant between 1.6 and 2
    if abs(Tv**(2-lamb)-Tp**(2-lamb)) > 0:
        C_T = (Tv - Tp)/Tv**(lamb-1.)*(2-lamb)/(Tv**(2-lamb)-Tp**(2-lamb))
    else:
        C_T = 1.
    
    p_d = get_droplet_vapor_pressure(Dp,Dd,tkappa,Tp)
    p_v = get_vapor_pressure(Sv,Tv)
    udiff = udiff_fun(t)
    Re = get_particle_Re(udiff,Dp,Tv,Sv)
    Sc = get_particle_Sc(Dp,Tv,Sv)
    Pr = get_particle_Pr(Tv,Sv)
    Sh = 1+0.38*Re**(1/2)*Sc**(1/3) # chen et al.
#    Sh = 2+0.6*Re**(1/2)*Sc**(1/3) #  Ranz, W. E. and Marshall, W. R. Evaporation from Drops. Chemical Engineering Progress, 48:141-146, 173-180, 1952.; via https://en.wikipedia.org/wiki/Sherwood_number
    dmdt = 2*np.pi*p*Dp*M_h2o*D_inf*C_T*Sh/(R*Tv)*np.log((p-p_d)/(p-p_v))
#    Nu = 2 + 0.4*Re**(1/2)*Pr**(1/3) # McAllister, S., Chen, J-Y. and Carlos Fernandez-Pello, A. Fundamentals of Combustion Processes. Springer, 2011. ch. 8 p. 159; via https://en.wikipedia.org/wiki/Nusselt_number#cite_note-9
    Nu = 1 + 0.3*Re**(1/2)*Pr**(1/3) # Chen et al. 
    dTdt = (np.pi*Dp**2*Kg*(Tv-Tp)/(Dp/2)*Nu+Lv*(dmdt))/(mp*Cp_h2o)
    
    return np.array([dmdt, dTdt])

def get_Dp_fun(ts,Dps,fill_value='none',evaporate=True,error_thresh=1e-5,N_samples_min=1000):
    these = ~np.isnan(Dps)
    Dps = Dps[these]
    ts = ts[these]
    
    if fill_value == 'none':
        fill_value = Dps[-1]
    ts,Dps = reduce_data(ts,Dps,error_thresh=error_thresh,N_samples_min=N_samples_min)
    if len(Dps) <= 2. or evaporate==False:
        Dp_fun = lambda t: Dps[0] + 0.*t
    else:
        Dp_fun = lambda t: interp1d(ts,Dps,fill_value=fill_value,bounds_error=False)(t)
    return Dp_fun

def get_wp_fun(Dp_vs_t,Dd,Tv_inf,S_inf,x_co2,ts = np.hstack([0.,np.logspace(-5,1,100)]),p=101325.,rho_aero=1000.):
    wp_vs_t = lambda t: get_terminal_velocity(Dp_vs_t(1.),get_particle_density(Dp_vs_t(1.),Dd,rho_aero=rho_aero),Tv_inf,S_inf,x_co2=x_co2,p=p)
    return wp_vs_t
def reduce_data(xs_orig,ys_orig,error_thresh=1e-5,N_samples_min=1000):
    N_samples = len(xs_orig)
    if N_samples > N_samples_min:
        keep_going = True    
    else:
        keep_going = False
    xs = xs_orig
    ys = ys_orig
    while keep_going:
        xs_new = np.linspace(min(xs),max(xs),int(N_samples/2))
        ys_new = interp1d(xs,ys)(xs_new)
        
        ys_test = interp1d(xs_new,ys_new)(xs_orig)
        
        error = np.mean(abs(ys_test-ys_orig)/ys_orig)
        
        N_samples = len(xs_new)
        
        if N_samples < N_samples_min:
            keep_going = False
            
        if error > error_thresh:
            keep_going = False
        
        if keep_going:
            xs = xs_new.copy()
            ys = ys_new.copy()
        
    return xs, ys

# @AR: new function to compute Tv and S dispersion: xxx
def get_Tv_and_S(Tv_0,S0,Tv_inf,S_inf,ts,xs,ys,zs,z0_fun,u0,D_mouth=2/100,p=101325.,K=0.,beta=0.114):
    x0s = np.asarray([get_x0(x,z,z0_fun,u0,Tv_inf,S_inf,Tv_0,S0,D_mouth=D_mouth,p=p) for (x,z) in zip(xs,zs)])
    ss = np.asarray([get_s(x0,z0_fun) for x0 in x0s])
    rs = np.asarray([get_r(x,0.,z,x0,0.,z0_fun(x0)) for (x0,x,z) in zip(x0s,xs,zs)])
    
    rho_v0 = M_h2o*S0*get_saturation_vapor_pressure(Tv_0)/(R*Tv_0)
    rho_vinf = M_h2o*S_inf*get_saturation_vapor_pressure(Tv_inf)/(R*Tv_inf)
    Tvs = np.zeros(ss.shape)
    Ss = np.zeros(ss.shape)
    Sv = S0
    for ii,(r,s) in enumerate(zip(rs,ss)):
        if s < 6.2*D_mouth:
            R_uni = D_mouth/2 - s/12.4
            if r<=R_uni:
                Tv = Tv_0
                rho_v = rho_v0
            else:
                Tv = Tv_inf
                rho_v = rho_vinf
        else:
            if Sv>S_inf or Tv>Tv_inf:
                Tv_m = Tv_inf + (Tv_0 - Tv_inf)*(5./(s/D_mouth))*np.sqrt(Tv_0/Tv_inf)
                rho_vm = rho_vinf + (rho_v0 - rho_vinf)*(5./(s/D_mouth))*np.sqrt(Tv_0/Tv_inf)
                Tv = Tv_inf + (Tv_m - Tv_inf)*np.exp(-r**2*np.log(2)/(0.11*s)**2)
                rho_v = rho_vinf + (rho_vm - rho_vinf)*np.exp(-r**2*np.log(2)/(0.11*s)**2)
                
        Tvs[ii] = Tv
        Sv = rho_v*R*Tv/(M_h2o*get_saturation_vapor_pressure(Tv))
        if Sv>S0:
            Ss[ii] = S0
        else:
            Ss[ii] = Sv
    ts0,Tvs0 = reduce_data(ts,Tvs,N_samples_min=1000)
    ts1,Ss1 = reduce_data(ts,Ss,N_samples_min=1000)  
    Tv_vs_t = lambda t: interp1d(ts0,Tvs0,fill_value=Tv_inf,bounds_error=False)(t)
    S_vs_t = lambda t: interp1d(ts1,Ss1,fill_value=S_inf,bounds_error=False)(t)
    
#    Tv_vs_t = lambda t: interp1d(ts,Tvs,fill_value=Tv_inf,bounds_error=False)(t)
#    S_vs_t = lambda t: interp1d(ts,Ss,fill_value=S_inf,bounds_error=False)(t)
    return Tv_vs_t, S_vs_t
    
# =============================================================================
# solve for position trajectories
# =============================================================================
def get_position_sol(Dp_fun,z0_fun,dz0dx_fun,s_fun,u0,Dd,rho_aero,D_mouth,ug_inf,Tv_inf,S_inf,max_cpu_time=60.,tau_thresh=0.,x_co2_inf=410e-6,Tv_source=310.15,S_source=1.,p=101325.,buoyancy=True,t_max=10.,K=0.,x_max=5.,z_min=-4.,beta=0.114,wp_thresh=1e-3):

    mp0 = get_mass_from_diameter(Dd,Dp_fun(0.),rho_aero)
    tau_relax = get_relaxation_time(mp0,Dp_fun(0.),Tv_inf)
    y = 0.
    if tau_relax >= tau_thresh:
        stopped_early = False
        z0 = 0.
        dzdt0 = 0.
        x0 = 0.
        dxdt0 = u0
        
        q0 = np.asarray([x0,dxdt0,z0,dzdt0])
        odesystem = lambda t,q: get_odesystem_position__no_y(
                q,t,Dp_fun,z0_fun,dz0dx_fun,s_fun,u0,Dd,rho_aero,D_mouth,ug_inf,Tv_inf,S_inf,
                tau_thresh=tau_thresh,x_co2_inf=x_co2_inf,Tv_source=Tv_source,S_source=S_source,p=p,beta=beta)
        stop_time = time.time() + max_cpu_time
        stop_condition = lambda t, q: 1.0*((q[0]<x_max) and (q[2]>z_min) and (Dp_fun(t)>0.) and (time.time()<stop_time))
        stop_condition.terminal = True
        
        ivp_output =solve_ivp(odesystem,(0.,t_max),q0,events=stop_condition,method='LSODA')
        q = ivp_output['y']
        
        x0s = np.asarray([get_x0(x,z,z0_fun,u0,Tv_inf,S_inf,Tv_source,S_source,D_mouth=D_mouth,p=p) for (x,z) in zip(q[0,:],q[2,:])])
        rs = np.asarray([get_r(x,y,z,x0,0.,z0_fun(x0)) for (x,z,x0) in zip(q[0,:],q[2,:],x0s)])
        ss = np.asarray([s_fun(x) for x in q[0,:]])
        bs = np.array([get_b(s,D_mouth,K=K,beta=beta) for s in ss])
        u0_excess = u0 - ug_inf
        ts = ivp_output['t']
        ug = ug_inf + np.asarray([jet_dispersion__velocity(s,r,D_mouth,u0_excess,K=K) for (r,s) in zip(rs,ss)])
        wg = q[1,:]*(beta*(q[2,:]-z0_fun(x0s))/bs + dz0dx_fun(x0s))
        
        udiff = np.sqrt((ug - q[1,:])**2 + (wg - q[3,:])**2)
        
        if not ((q[0,-1]>=x_max) or (q[2,-1]<=z_min) or (Dp_fun(ts[-1])==0.)):
            stopped_early = True
        
        x_pos = q[0,:]
        dxdt_pos = q[1,:]
        y_pos = 0.
        dydt_pos = 0.
        z_pos = q[2,:]
        dzdt_pos = q[3,:]
    else:
        stopped_early = False
        z0 = 0.
        dzdt0 = 0.
        x0 = 0.
        
        q0 = np.array([x0,z0])
        odesystem = lambda t,q: get_odesystem_position__with_flow__no_y2(
                q,t,Dp_fun,z0_fun,dz0dx_fun,s_fun,u0,Dd,rho_aero,D_mouth,ug_inf,Tv_inf,S_inf,
                tau_thresh=tau_thresh,x_co2_inf=x_co2_inf,Tv_source=Tv_source,S_source=S_source,p=p,beta=beta)
        stop_time = time.time() + max_cpu_time
        stop_condition = lambda t, q: 1.0*((q[0]<x_max) and (q[1]>z_min) and (Dp_fun(t)>0.) and (time.time()<stop_time))
        stop_condition.terminal = True
        
        ivp_output =solve_ivp(odesystem,(0.,t_max),q0,events=stop_condition,method='LSODA')
        q = ivp_output['y']
        
        x0s = np.asarray([get_x0(x,z,z0_fun,u0,Tv_inf,S_inf,Tv_source,S_source,D_mouth=D_mouth,p=p) for (x,z) in zip(q[0,:],q[1,:])])
        rs = np.asarray([get_r(x,y,z,x0,0.,z0_fun(x0)) for (x,z,x0) in zip(q[0,:],q[1,:],x0s)])
        ss = np.asarray([s_fun(x) for x in q[0,:]])
        bs = np.array([get_b(s,D_mouth,K=K,beta=beta) for s in ss])
        u0_excess = u0 - ug_inf
        ts = ivp_output['t']
        ug = ug_inf + np.asarray([jet_dispersion__velocity(s,r,D_mouth,u0_excess,K=K) for (r,s) in zip(rs,ss)])
        wg = ug*(beta*(q[1,:]-z0_fun(x0s))/bs + dz0dx_fun(x0s))
        
        wps = np.array([get_terminal_velocity(Dp_fun(t),get_particle_density(Dp_fun(t),Dd,rho_aero=rho_aero),Tv_inf,S_inf,x_co2=x_co2_inf) for t in ts])
        udiff = np.zeros(q[0,:].shape)
        
        if not ((q[0,-1]>=x_max) or (q[1,-1]<=z_min) or (Dp_fun(ts[-1])==0.)):
            stopped_early = True
        
        x_pos = q[0,:]
        dxdt_pos = ug
        y_pos = 0.*np.ones(q[0,:].shape)
        dydt_pos = 0.*np.ones(q[0,:].shape)
        z_pos = q[1,:]
        dzdt_pos = (wg - wps.transpose())[0,:]
            
    return ts,x_pos,dxdt_pos,y_pos,dydt_pos,z_pos,dzdt_pos,udiff,stopped_early

def get_odesystem_position__no_y(q,t,Dp_fun,z0_fun,dz0dx_fun,s_fun,u0,Dd,rho_aero,D_mouth,
                           ug_inf,Tv_inf,S_inf,tau_thresh=1e-5,
                           x_co2_inf=410e-6,Tv_source=310.15,S_source=1.,p=101325.,beta=0.114,alpha=0.057):
    
    x = q[0]
    dxdt = q[1]
    z = q[2]
    dzdt = q[3]
    y = 0.
    
    Dp = Dp_fun(t)
    
    rho_p = get_particle_density(Dp,Dd,rho_aero=rho_aero)
    
    x0 = get_x0(x,z,z0_fun,u0,Tv_inf,S_inf,Tv_source,S_source,D_mouth=2/100,p=101325.)    
    s = s_fun(x0)
    r = get_r(x,y,z,x0,0.,z0_fun(x0))
    
    u0_excess = u0 - ug_inf
    ug = ug_inf + jet_dispersion__velocity(s,r,D_mouth,u0_excess)
    
    b = get_b(s,D_mouth,K=0.,beta=beta)
    
    wg = dxdt*(beta*(z-z0_fun(x))/b + dz0dx_fun(x))
    
    Tv = Tv_inf
    Sv = S_inf
    
    rho_g = get_air_density(Tv,Sv,p=p)
    d2xdt2 = (ug - dxdt)*abs(ug - dxdt)*3*rho_g*get_drag_coefficient(dxdt,ug,Dp,Tv,Sv)/(4*Dp*rho_p)
    d2zdt2 = -g + (wg - dzdt)*abs(wg - dzdt)*3*rho_g*get_drag_coefficient(dzdt,wg,Dp,Tv,Sv)/(4*Dp*rho_p)
    
    return np.array([dxdt, d2xdt2, dzdt, d2zdt2])


def get_odesystem_position__with_flow__no_y(q,t,Dp_fun,z0_fun,dz0dx_fun,s_fun,u0,Dd,rho_aero,D_mouth,
                           ug_inf,Tv_inf,S_inf,tau_thresh=1e-5,
                           x_co2_inf=410e-6,Tv_source=310.15,S_source=1.,p=101325.,beta=0.114,alpha=0.057):    
    
    x = q[0]
    z = q[1]
    dzdt = q[2]
    
    y = 0.
    
    
    Dp = Dp_fun(t)
    rho_p = get_particle_density(Dp,Dd,rho_aero=rho_aero)
    
    x0 = get_x0(x,z,z0_fun,u0,Tv_inf,S_inf,Tv_source,S_source,D_mouth=2/100,p=101325.)    
    s = s_fun(x0)
    r = get_r(x,y,z,x0,0.,z0_fun(x0))
    
    u0_excess = u0 - ug_inf
    ug = ug_inf + jet_dispersion__velocity(s,r,D_mouth,u0_excess)
    dxdt = ug
    
    b = get_b(s,D_mouth,K=0.,beta=beta)
#    wg = dxdt*(beta*(z-z0_fun(x))/b + dz0dx_fun(x))
    if z<z0_fun(x):
        wg = -alpha*(6.2*u0_excess*(D_mouth/x))*(1-np.exp(-r**2/b**2) - (beta/alpha)*(r**2/b**2)*np.exp(-r**2/b**2))/(r/b)
    else:
        wg = alpha*(6.2*u0_excess*(D_mouth/x))*(1-np.exp(-r**2/b**2) - (beta/alpha)*(r**2/b**2)*np.exp(-r**2/b**2))/(r/b)
    
    Tv = Tv_inf
    Sv = S_inf
    
    rho_g = get_air_density(Tv,Sv,p=p)    
    d2zdt2 = -g + (wg - dzdt)*abs(wg - dzdt)*3*rho_g*get_drag_coefficient(dzdt,wg,Dp,Tv,Sv)/(4*Dp*rho_p)
    
    return np.array([dxdt, dzdt, d2zdt2])

def get_odesystem_position__with_flow__no_y2(q,t,Dp_fun,z0_fun,dz0dx_fun,s_fun,u0,Dd,rho_aero,D_mouth,
                           ug_inf,Tv_inf,S_inf,tau_thresh=1e-5,
                           x_co2_inf=410e-6,Tv_source=310.15,S_source=1.,p=101325.,beta=0.114,):    
    
    x = q[0]
    z = q[1]
    y = 0.
    
    x0 = get_x0(x,z,z0_fun,u0,Tv_inf,S_inf,Tv_source,S_source,D_mouth=2/100,p=101325.)    
    s = s_fun(x0)
    r = get_r(x,y,z,x0,0.,z0_fun(x0))
    
    u0_excess = u0 - ug_inf
    ug = ug_inf + jet_dispersion__velocity(s,r,D_mouth,u0_excess)
    dxdt = ug
    
    b = get_b(s,D_mouth,K=0.,beta=beta)
    wg = dxdt*(beta*(z-z0_fun(x))/b + dz0dx_fun(x))
    
    wp = get_terminal_velocity(Dp_fun(t),get_particle_density(Dp_fun(t),Dd,rho_aero=rho_aero),Tv_inf,S_inf,x_co2=x_co2_inf)    
    dzdt = wg - wp
    
    return np.array([dxdt, dzdt])

def get_odesystem_position(q,t,Dp_fun,z0_fun,dz0dx_fun,s_fun,u0,Dd,rho_aero,D_mouth,
                           ug_inf,Tv_inf,S_inf,tau_thresh=1e-5,
                           x_co2_inf=410e-6,Tv_source=310.15,S_source=1.,p=101325.,beta=0.114,alpha=0.057):    
    
    x = q[0]
    dxdt = q[1]
    y = q[2]
    dydt = q[3]
    z = q[4]
    dzdt = q[5]
    
    Dp = Dp_fun(t)
    rho_p = get_particle_density(Dp,Dd,rho_aero=rho_aero)
    
    x0 = get_x0(x,z,z0_fun,u0,Tv_inf,S_inf,Tv_source,S_source,D_mouth=2/100,p=101325.)    
    s = s_fun(x0)
    r = get_r(x,y,z,x0,0.,z0_fun(x0))
    
    u0_excess = u0 - ug_inf
    ug = ug_inf + jet_dispersion__velocity(s,r,D_mouth,u0_excess)
    vg = 0.
    
    b = get_b(s,D_mouth,K=0.,beta=beta)
    wg = dxdt*(beta*(z-z0_fun(x))/b + dz0dx_fun(x))
    Tv = Tv_inf
    Sv = S_inf
    
    rho_g = get_air_density(Tv,Sv,p=p)    
    d2xdt2 = (ug - dxdt)*abs(ug - dxdt)*3*rho_g*get_drag_coefficient(dxdt,ug,Dp,Tv,Sv)/(4*Dp*rho_p)
    d2ydt2 = (vg - dydt)*abs(vg - dydt)*3*rho_g*get_drag_coefficient(dydt,vg,Dp,Tv,Sv)/(4*Dp*rho_p)
    d2zdt2 = -g + (wg - dzdt)*abs(wg - dzdt)*3*rho_g*get_drag_coefficient(dzdt,wg,Dp,Tv,Sv)/(4*Dp*rho_p)
    
    return np.array([dxdt, d2xdt2, dydt, d2ydt2, dzdt, d2zdt2])


# =============================================================================
# PART 2: FUNCTIONS TO COMPUTE VIRION CONCENTRATION AND DEPOSITION PER VIRIONS EMITTED
#   at each location x, y, z and exposure time t, predicts  
#        concentration per rate at which virions are expelled (s/m^3),
#        virions inhaled per virons expelled (unitless), and
#        virions deposited to the nasal cavity per virons expelled (unitless)
# =============================================================================

def store__02_normconcs(
        scenario_dir,gridding,
        emission_type='talk',excel_filename='../input/inputs.xlsx',
        overwrite_gridding=False):
    
    start_time = time.process_time()
    trajectories_filename = scenario_dir + '01_trajectories.pkl'
    trajectories_dict = pickle.load(open(trajectories_filename, 'rb'))
    
    normconcs_filename = scenario_dir + '02_normconcs.pkl'
    
    # check to see if normconc_dict dict has already been stored
    options = pickle.load(open(scenario_dir + '00_info.pkl','rb'))
    inputs__allcases = get_inputs___deposition__allcases(
            trajectories_dict,options,excel_filename='../input/inputs.xlsx')
    if os.path.exists(normconcs_filename):
        normconcs_dict = pickle.load(open(normconcs_filename, 'rb'))        
    else:
        normconcs_dict = dict()
    
    if len(normconcs_dict) == 0:
        store = True
    elif overwrite_gridding or int(len(inputs__allcases)-1) not in normconcs_dict.keys() or gridding not in normconcs_dict[int(len(normconcs_dict)-1)].keys():
        store = True
    else:
        store = False
        
    if store:
        if not any(normconcs_dict.keys()):
            for cc in inputs__allcases.keys():
                inputs = inputs__allcases[cc]
                normconcs_dict[cc] = {'inputs':inputs}
        normconcs_dict = populate_normconcs_dict(
            normconcs_dict, trajectories_dict, gridding,
            excel_filename='../input/inputs.xlsx')
        pickle.dump(normconcs_dict,open(normconcs_filename, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        print('another normconc finished!', scenario_dir, gridding, 'process_time:', time.process_time() - start_time)
    else:
        print('normconc gridding already stored!', scenario_dir, gridding, 'process_time:', time.process_time() - start_time)
    print('normconcs_dict[0].keys():',normconcs_dict[0].keys())
    return normconcs_dict

def populate_normconcs_dict(
        normconcs_dict, trajectories_dict, gridding,
        excel_filename='../input/inputs.xlsx'):
    
    dxs, dys, dzs, ts_durration = get_default_grid(gridding,z0_source=normconcs_dict[0]['inputs']['z0_source'])
    
    # additional cases not included in trajectories
    
    
#    if len(trajectories_dict)>1:
#        chk = True
#        for cc in trajectories_dict.keys():
#            if len(trajectories_dict[0]['inputs']['D']) != len(trajectories_dict[cc]['inputs']['D']): 
#                chk = False
#                break;
#    else:
#        chk = False
#        
#    if chk: 
    # need to re-work the looping
    tmin = np.ones([len(dxs),len(dys),len(dzs),len(trajectories_dict[0]['inputs']['D'])])*1000.
    tmax = np.ones([len(dxs),len(dys),len(dzs),len(trajectories_dict[0]['inputs']['D'])])*0.
    
    for cc in trajectories_dict.keys(): # loop through cases
        inputs_dict = trajectories_dict[cc]['inputs']
        
        xs = dxs + inputs_dict['L']/2.
        ys = dys + inputs_dict['W']/2.
        zs = dzs + inputs_dict['z0_source']
#        print(inputs_dict['D'])
        for qq in range(len(inputs_dict['D'])): # loop through particles:
            df_pos = trajectories_dict[cc][qq]['pos']
            df_evap = trajectories_dict[cc][qq]['evap']
            Dd = inputs_dict['Dd'][qq]
            rho_aero = inputs_dict['rho_aero']
            evaporate = inputs_dict['evaporate']
            x0_vs_t, y0_vs_t, z0_vs_t, b_vs_t = get_puff_funs(df_pos,inputs_dict)
            Dp_vs_t = get_Dp_fun(df_evap['mp'].index, get_diameter_from_mass(Dd,df_evap['mp'].values,rho_aero), evaporate=evaporate)
            for ii,x in enumerate(xs):
                for jj,y in enumerate(ys):
                    for kk,z in enumerate(zs):
                        conc_fun = lambda t: get_conc(t,x,y,z,x0_vs_t, y0_vs_t, z0_vs_t, b_vs_t, Dp_vs_t, 
                                 inputs_dict, just_reflections=False,
                                 dzdt_vs_t=lambda t: 0.*t, wp=0.,
                                 N_reflections=0,Dp_noreflect=0.,exponent_BL = 0.26,prefactor_BL = 0.06)
                        ts,dt = get_ts(conc_fun,fixed_ts=False,tlims_log=normconcs_dict[cc]['inputs']['tlims_log'],N_loops=normconcs_dict[0]['inputs']['N_loops'],thresh=normconcs_dict[0]['inputs']['tlim_thresh'],Nts=normconcs_dict[0]['inputs']['Nts_jet'])
                        tmin[ii,jj,kk,qq] = min([tmin[ii,jj,kk,qq],min(ts)])
                        tmax[ii,jj,kk,qq] = max([tmax[ii,jj,kk,qq],max(ts)])
        
    for cc in normconcs_dict.keys():
        inputs_dict = normconcs_dict[cc]['inputs']
        ts_durration = inputs_dict['ts_durration']
        
        cc_trajectories = inputs_dict['trajectories_case']
        
        xs = dxs + inputs_dict['L']/2.
        ys = dys + inputs_dict['W']/2.
        zs = dzs + inputs_dict['z0_source']
        
        C_per_emit__jet = np.zeros([len(dxs),len(dys),len(dzs),len(trajectories_dict[cc_trajectories]['inputs']['D'])])
        dNdt_inhale_per_emit__jet = np.zeros(C_per_emit__jet.shape)
        dNdt_deposit_per_emit__jet = np.zeros(C_per_emit__jet.shape)        
        
        C_per_emit__reflections = np.zeros([len(ts_durration),len(trajectories_dict[cc_trajectories]['inputs']['D'])])
        dNdt_inhale_per_emit__reflections = np.zeros(C_per_emit__reflections.shape)
        dNdt_deposit_per_emit__reflections = np.zeros(C_per_emit__reflections.shape)
        
        C_per_emit__well_mixed = np.zeros([len(ts_durration),len(trajectories_dict[cc_trajectories]['inputs']['D'])])
        dNdt_inhale_per_emit__well_mixed = np.zeros(C_per_emit__reflections.shape)
        dNdt_deposit_per_emit__well_mixed = np.zeros(C_per_emit__reflections.shape)
        
        ts_inactivate = np.logspace(0,np.log10(3600*24*2),1000)
        k_inactivate_q = np.zeros([len(ts_inactivate),len(trajectories_dict[cc_trajectories]['inputs']['D'])])
        f_alive_q = np.zeros([len(ts_inactivate),len(trajectories_dict[cc_trajectories]['inputs']['D'])])
        
        V_breathe = inputs_dict['V_breathe']
        
        for qq,w_cov in enumerate(inputs_dict['w_cov']):
            start_time = time.process_time()
            if w_cov>0: 
                df_pos = trajectories_dict[cc_trajectories][qq]['pos']
                df_evap = trajectories_dict[cc_trajectories][qq]['evap']
                Dd = inputs_dict['Dd'][qq]
                evaporate = inputs_dict['evaporate']
                x0_vs_t, y0_vs_t, z0_vs_t, b_vs_t = get_puff_funs(df_pos,inputs_dict)
                Dp_vs_t = get_Dp_fun(df_evap['mp'].index, get_diameter_from_mass(Dd,df_evap['mp'].values,inputs_dict['rho_aero']), evaporate=evaporate)
                wp_vs_t = get_wp_fun(Dp_vs_t,Dd,inputs_dict['Tv_inf'],inputs_dict['S_inf'],inputs_dict['x_co2_inf'],ts = np.hstack([0.,np.logspace(-5,3,1000)]),p=inputs_dict['p'],rho_aero=inputs_dict['rho_aero'])
                
                FE_out,rel_std = get_mask_efficiency(Dp_vs_t(0.),direction='out',mask_type=inputs_dict['smask_type'],excel_filename='../input/inputs.xlsx', relative_std_dev=inputs_dict['FE_out_relative_std'], fill_val='conservative')
                
                if inputs_dict['inactivate']:
#                    k_inactivate = lambda t: get_inactivation_rate(
#                            inputs_dict['Dd'][qq], inputs_dict['D'][qq], Dp_vs_t(t), inputs_dict['rho_aero'], inputs_dict['tkappa'][qq], 
#                            inputs_dict['Tv_inf'], inputs_dict['S_inf'], 
#                            inputs_dict['E_a'], inputs_dict['A_eff'], inputs_dict['A_sol'])
#                    k_inactivate = lambda t: get_inactivation_rate__step(inputs_dict['S_inf'])
                    k_inactivate = lambda t: get_inactivation_rate__step(inputs_dict['S_inf'],inputs_dict['Tv_inf'])                    
#                    k_inactivate_morris = lambda t: get_inactivation_rate(
#                            inputs_dict['Dd'][qq], inputs_dict['D'][qq], Dp_vs_t(t), inputs_dict['rho_aero'], inputs_dict['tkappa'][qq], 
#                            inputs_dict['Tv_inf'], inputs_dict['S_inf'], 
#                            inputs_dict['E_a'], inputs_dict['A_eff'], inputs_dict['A_sol'])
#                    f_alive_vs_time_moris= lambda t: np.exp(-k_inactivate(t)*t)
#                    ts = np.linspace(0.,3600*4,5)
#                    for one_t in ts:
#                        print(one_t,k_inactivate(one_t),get_inactivation_rate__step(inputs_dict['S_inf']))
                else:
                    k_inactivate = lambda t: 0.*t
                
                f_alive_vs_time = lambda t: np.exp(-k_inactivate(t)*t)
                
                conc_fun__jet = lambda t, x, y, z: get_conc(t,x,y,z,x0_vs_t, y0_vs_t, z0_vs_t, b_vs_t, Dp_vs_t, 
                         inputs_dict, just_reflections=False,
                         dzdt_vs_t=wp_vs_t, wp=wp_vs_t(1.),
                         N_reflections=0,Dp_noreflect=inputs_dict['Dp_noreflect'],exponent_BL = 0.26,prefactor_BL = 0.06)
                
                conc_fun__reflections = lambda t, x, y, z: get_conc(
                        t,x,y,z,
                        x0_vs_t, y0_vs_t, z0_vs_t, b_vs_t, Dp_vs_t, 
                        inputs_dict, 
                        just_reflections=True, dzdt_vs_t=wp_vs_t, wp=wp_vs_t(1.),
                        N_reflections=inputs_dict['N_reflections'],Dp_noreflect=inputs_dict['Dp_noreflect'],exponent_BL = 0.26,prefactor_BL = 0.06)
                
                conc_fun__well_mixed = lambda t, x, y, z: get_conc(
                        t,x,y,z,
                        x0_vs_t, y0_vs_t, z0_vs_t, b_vs_t, Dp_vs_t, 
                        inputs_dict, well_mixed=True,
                        just_reflections=True, dzdt_vs_t=wp_vs_t, wp=wp_vs_t(1.),
                        N_reflections=inputs_dict['N_reflections'],Dp_noreflect=inputs_dict['Dp_noreflect'],exponent_BL = 0.26,prefactor_BL = 0.06)
                
                for ii,x in enumerate(xs):
                    for jj,y in enumerate(ys):
                        for kk,z in enumerate(zs):
                            conc_fun = lambda t: conc_fun__jet(t,x,y,z)
#                            ts,dt = get_ts(conc_fun,fixed_ts=False,tlims_log=normconcs_dict[cc]['inputs']['tlims_log'],N_loops=normconcs_dict[cc]['inputs']['N_loops'],thresh=normconcs_dict[0]['inputs']['tlim_thresh'],Nts=normconcs_dict[cc]['inputs']['Nts_jet'])  
                            ts = np.linspace(tmin[ii,jj,kk,qq],tmax[ii,jj,kk,qq],normconcs_dict[cc]['inputs']['Nts_jet'])
                            dt = ts[1] - ts[0]
                            dep_coeffs = [inputs_dict['dep_coeff_a'],inputs_dict['dep_coeff_b'],inputs_dict['dep_coeff_c'],inputs_dict['dep_coeff_d']]
                            DEs = get_nasal_deposition_efficiency(Dp_vs_t(ts),inputs_dict['Dd'][qq],inputs_dict['rho_aero'],inputs_dict['V_breathe'],inputs_dict['Tv_source'],dep_coeffs=dep_coeffs)
                            FEs_in,rel_std = get_mask_efficiency(Dp_vs_t(ts),direction='in',mask_type=inputs_dict['rmask_type'],excel_filename='../input/inputs.xlsx',relative_std_dev=inputs_dict['FE_in_relative_std'],fill_val='conservative')
                            cs = conc_fun__jet(ts,x,y,z)*f_alive_vs_time(ts)*dt
                            these = ~np.isnan(cs)
                            C_per_emit__jet[ii,jj,kk,qq] = w_cov*(1.-FE_out)*sum(cs[these])
                            dNdt_inhale_per_emit__jet[ii,jj,kk,qq] = w_cov*(1.-FE_out)*V_breathe*sum((1.-FEs_in[these])*cs[these])
                            dNdt_deposit_per_emit__jet[ii,jj,kk,qq] = w_cov*(1.-FE_out)*V_breathe*sum(DEs[these]*(1.-FEs_in[these])*cs[these])
                
                for tt,t in enumerate(ts_durration):
#                    print(t)
                    ts = np.linspace(0,t,inputs_dict['Nts_reflections'])
                    dt = ts[1] - ts[0]
                    dep_coeffs = [inputs_dict['dep_coeff_a'],inputs_dict['dep_coeff_b'],inputs_dict['dep_coeff_c'],inputs_dict['dep_coeff_d']]
                    DEs = get_nasal_deposition_efficiency(Dp_vs_t(ts),Dd,inputs_dict['rho_aero'],inputs_dict['V_breathe'],inputs_dict['Tv_source'],dep_coeffs=dep_coeffs)
                    FEs_in,rel_std = get_mask_efficiency(Dp_vs_t(ts),direction='in',mask_type=inputs_dict['rmask_type'],excel_filename='../input/inputs.xlsx',relative_std_dev=inputs_dict['FE_in_relative_std'],fill_val='conservative')
                    cs = conc_fun__reflections(ts,inputs_dict['L']/2.,inputs_dict['W']/2.,inputs_dict['z0_source'])*f_alive_vs_time(ts)*dt
                    these = ~np.isnan(cs)
                    C_per_emit__reflections[tt,qq] = w_cov*(1.-FE_out)*sum(cs[these])
                    dNdt_inhale_per_emit__reflections[tt,qq] = w_cov*(1.-FE_out)*inputs_dict['V_breathe']*sum((1.-FEs_in[these])*cs[these])
                    dNdt_deposit_per_emit__reflections[tt,qq] = w_cov*(1.-FE_out)*inputs_dict['V_breathe']*sum(DEs[these]*(1.-FEs_in[these])*cs[these])
                    
                    cs = conc_fun__well_mixed(ts,inputs_dict['L']/2.,inputs_dict['W']/2.,inputs_dict['z0_source'])*f_alive_vs_time(ts)*dt
                    these = ~np.isnan(cs)
                    C_per_emit__well_mixed[tt,qq] = w_cov*(1.-FE_out)*sum(cs[these])
                    dNdt_inhale_per_emit__well_mixed[tt,qq] = w_cov*(1.-FE_out)*inputs_dict['V_breathe']*sum((1.-FEs_in[these])*cs[these])
                    dNdt_deposit_per_emit__well_mixed[tt,qq] = w_cov*(1.-FE_out)*inputs_dict['V_breathe']*sum(DEs[these]*(1.-FEs_in[these])*cs[these])
                    
        f_alive_q[:,qq] = f_alive_vs_time(ts_inactivate)
        k_inactivate_q[:,qq] = k_inactivate(ts_inactivate)
                
        normconcs_dict[cc][gridding] = {
                'ts_durration':ts_durration,
                'dxs':dxs,'dys':dys,'dzs':dzs,
                'k_inactivate_q':k_inactivate_q,
                'f_alive_q':f_alive_q,
                'C_per_emit':{'jet':np.sum(C_per_emit__jet,axis=3),'reflections':np.sum(C_per_emit__reflections,axis=1),'well_mixed':np.sum(C_per_emit__well_mixed,axis=1)},
                'Cq_per_emit':{'jet':C_per_emit__jet,'reflections':C_per_emit__reflections,'well_mixed':C_per_emit__well_mixed},
                'dNqdt_inhale_per_emit':{'jet':dNdt_inhale_per_emit__jet,'reflections':dNdt_inhale_per_emit__reflections,'well_mixed':dNdt_inhale_per_emit__well_mixed},
                'dNqdt_deposit_per_emit':{'jet':dNdt_deposit_per_emit__jet,'reflections':dNdt_deposit_per_emit__reflections,'well_mixed':dNdt_deposit_per_emit__well_mixed},
                'dNdt_inhale_per_emit':{'jet':np.sum(dNdt_inhale_per_emit__jet,axis=3),'reflections':np.sum(dNdt_inhale_per_emit__reflections,axis=1),'well_mixed':np.sum(dNdt_inhale_per_emit__well_mixed,axis=1)},
                'dNdt_deposit_per_emit':{'jet':np.sum(dNdt_deposit_per_emit__jet,axis=3),'reflections':np.sum(dNdt_deposit_per_emit__reflections,axis=1),'well_mixed':np.sum(dNdt_deposit_per_emit__well_mixed,axis=1)}}
        
    return normconcs_dict

# =============================================================================
# STORE INFECTION
# =============================================================================

def store__03_infection(
        X_infection,all_varnames,scenario_dir,
        emission_type='talk',excel_filename='../input/inputs.xlsx'):
    # note: this X is not the same as the one used to generate the ensemble
    start_time = time.process_time()
    options = pickle.load(open(scenario_dir + '00_info.pkl','rb'))
    
    normconcs_filename = scenario_dir + '02_normconcs.pkl'
    normconcs_dict = pickle.load(open(normconcs_filename, 'rb'))

    infection_filename = scenario_dir + '03_infection.pkl'
    
    infection_dict = dict()
    for ii,X_i_infection in enumerate(X_infection):
        inputs__allcases = get_inputs___infection__allcases(
                X_i_infection,all_varnames,normconcs_dict,options,excel_filename='../input/inputs.xlsx')    
        
        infection_dict__onescenario = dict()
        for cc in inputs__allcases.keys():
            inputs = inputs__allcases[cc]
            infection_dict__onescenario[cc] = {'inputs':inputs}
            
        infection_dict__onescenario = populate_infection_dict__onescenario(
                infection_dict__onescenario, normconcs_dict,excel_filename='../input/inputs.xlsx')
        infection_dict[ii] = infection_dict__onescenario
    pickle.dump(infection_dict,open(infection_filename, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    print('another infection_dict finished!', scenario_dir, 'process_time:', time.process_time() - start_time)
    
    return infection_dict

def store__03_infection2(
        X_i_infection,all_varnames,scenario_dir,
        emission_type='talk',excel_filename='../input/inputs.xlsx'):
    
    # note: this X is not the same as the one used to generate the ensemble
    start_time = time.process_time()
    options = pickle.load(open(scenario_dir + '00_info.pkl','rb'))
    
    normconcs_filename = scenario_dir + '02_normconcs.pkl'
    normconcs_dict = pickle.load(open(normconcs_filename, 'rb'))

    infection_filename = scenario_dir + '03_infection.pkl'
    
    infection_dict = dict()
    inputs__allcases = get_inputs___infection__allcases(
            X_i_infection,all_varnames,normconcs_dict,options,excel_filename='../input/inputs.xlsx')    
    
    infection_dict__onescenario = dict()
    for cc in inputs__allcases.keys():
        inputs = inputs__allcases[cc]
        infection_dict__onescenario[cc] = {'inputs':inputs}
        
    infection_dict__onescenario = populate_infection_dict__onescenario(
            infection_dict__onescenario, normconcs_dict,excel_filename='../input/inputs.xlsx')
    infection_dict[0] = infection_dict__onescenario
    pickle.dump(infection_dict,open(infection_filename, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    print('another infection_dict finished!', scenario_dir, 'process_time:', time.process_time() - start_time)
    
    return infection_dict

def get_inputs___infection__allcases(
        X_i_infection,all_varnames,normconcs_dict,options,excel_filename='../input/inputs.xlsx'):    

    inputs__allcases = dict()
    ccc = 0
    for cc in range(len(normconcs_dict)):
        for vaccine_efficacy in options['vaccine_efficacys']:
            emission_type = normconcs_dict[cc]['inputs']['emission_type']
            infection_inputs_dict = normconcs_dict[cc]['inputs']            
            
            other_varnames = ['muc_free','K_m','p_pfu','p_cell']
            for varname in other_varnames:
                idx,=np.where([var == varname for var in all_varnames])[0]
                if type(X_i_infection[idx]) == np.float64:
                    infection_inputs_dict[varname] = X_i_infection[idx]
                else:
                    infection_inputs_dict[varname] = X_i_infection[idx][0]
                
            idx,=np.where([var == 'dNdt_cov_' + emission_type for var in all_varnames])
            if type(X_i_infection[idx]) == np.float64:
                infection_inputs_dict['dNdt_cov'] = X_i_infection[idx]
            else:
                infection_inputs_dict['dNdt_cov'] = X_i_infection[idx][0]
            infection_inputs_dict['vaccine_efficacy'] = vaccine_efficacy
            infection_inputs_dict['normconcs_case'] = cc
        inputs__allcases[ccc] = infection_inputs_dict
        ccc += 1
    return inputs__allcases


def populate_infection_dict__onescenario(
        infection_dict__onescenario, normconcs_dict,excel_filename='../input/inputs.xlsx',dispersion_types=['jet','reflections','both','well_mixed']):
    
    # Fraction of challenge dose reaching (LLF) nasal epiphelium?
    F_trans = 1.
    # Fraction of virus dose bound to (lung) nasal? cells
    F_c = 1.
    
    for cc in infection_dict__onescenario.keys():
        inputs_dict = infection_dict__onescenario[cc]['inputs']
        cc_normconcs = inputs_dict['normconcs_case']
        
        griddings = [one_key for one_key in normconcs_dict[cc_normconcs].keys() if one_key != 'inputs']
        for gridding in griddings:
            infection_dict__onescenario[cc][gridding] = dict()
            ts_durration = normconcs_dict[cc_normconcs][gridding]['ts_durration']
            dxs = normconcs_dict[cc_normconcs][gridding]['dxs']
            dys = normconcs_dict[cc_normconcs][gridding]['dys']
            dzs = normconcs_dict[cc_normconcs][gridding]['dzs']
            
            infection_dict__onescenario[cc][gridding]['dxs'] = dxs
            infection_dict__onescenario[cc][gridding]['dys'] = dys
            infection_dict__onescenario[cc][gridding]['dzs'] = dzs
            infection_dict__onescenario[cc][gridding]['ts_durration'] = ts_durration
            
            dNdt_cov = inputs_dict['dNdt_cov']
            
            for dispersion_type in dispersion_types:
                if dispersion_type == 'jet':
                    C_per_emit = normconcs_dict[cc_normconcs][gridding]['C_per_emit']['jet']
                    dNdt_inhale_per_emit = normconcs_dict[cc_normconcs][gridding]['dNdt_inhale_per_emit']['jet']
                    dNdt_deposit_per_emit = normconcs_dict[cc_normconcs][gridding]['dNdt_deposit_per_emit']['jet']
                    dose_per_emit = dNdt_deposit_per_emit*ts_durration[0]
                elif dispersion_type == 'reflections':
                    C_per_emit = normconcs_dict[cc_normconcs][gridding]['C_per_emit']['reflections']
                    dNdt_inhale_per_emit = normconcs_dict[cc_normconcs][gridding]['dNdt_inhale_per_emit']['reflections']
                    dNdt_deposit_per_emit = normconcs_dict[cc_normconcs][gridding]['dNdt_deposit_per_emit']['reflections']
                    dose_per_emit = normconcs_dict[cc_normconcs][gridding]['dNdt_deposit_per_emit']['reflections']          
                    dose_per_emit = dNdt_deposit_per_emit*ts_durration
                elif dispersion_type == 'both':
                    C_per_emit = np.zeros([len(ts_durration),len(dxs),len(dys),len(dzs)])
                    dNdt_inhale_per_emit = np.zeros([len(ts_durration),len(dxs),len(dys),len(dzs)])
                    dNdt_deposit_per_emit = np.zeros([len(ts_durration),len(dxs),len(dys),len(dzs)])
                    dose_per_emit = np.zeros([len(ts_durration),len(dxs),len(dys),len(dzs)])                    
                    for tt in range(len(ts_durration)):
                        C_per_emit[tt,:,:,:] = normconcs_dict[cc_normconcs][gridding]['C_per_emit']['reflections'][tt] + normconcs_dict[cc_normconcs][gridding]['C_per_emit']['jet']
                        dNdt_inhale_per_emit[tt,:,:,:] = normconcs_dict[cc_normconcs][gridding]['dNdt_inhale_per_emit']['reflections'][tt] + normconcs_dict[cc_normconcs][gridding]['dNdt_inhale_per_emit']['jet']
                        dNdt_deposit_per_emit[tt,:,:,:] = normconcs_dict[cc_normconcs][gridding]['dNdt_deposit_per_emit']['reflections'][tt] + normconcs_dict[cc_normconcs][gridding]['dNdt_deposit_per_emit']['jet']
                        dose_per_emit[tt,:,:,:] = dNdt_deposit_per_emit[tt,:,:,:]*ts_durration[tt]
                elif dispersion_type == 'well_mixed':
                    C_per_emit = normconcs_dict[cc_normconcs][gridding]['C_per_emit']['well_mixed']
                    dNdt_inhale_per_emit = normconcs_dict[cc_normconcs][gridding]['dNdt_inhale_per_emit']['well_mixed']
                    dNdt_deposit_per_emit = normconcs_dict[cc_normconcs][gridding]['dNdt_deposit_per_emit']['well_mixed']
                    dose_per_emit = normconcs_dict[cc_normconcs][gridding]['dNdt_deposit_per_emit']['well_mixed']          
                    dose_per_emit = dNdt_deposit_per_emit*ts_durration
                    
                infection_dict__onescenario[cc][gridding][dispersion_type] = dict()
                infection_dict__onescenario[cc][gridding][dispersion_type]['C'] = dNdt_cov*C_per_emit
                infection_dict__onescenario[cc][gridding][dispersion_type]['dNdt_inhale'] = dNdt_cov*dNdt_inhale_per_emit
                infection_dict__onescenario[cc][gridding][dispersion_type]['dNdt_deposit'] = dNdt_cov*dNdt_deposit_per_emit
                infection_dict__onescenario[cc][gridding][dispersion_type]['dose'] = dNdt_cov*dose_per_emit
                muc_free = inputs_dict['muc_free']
                K_m = inputs_dict['K_m']
                F_v = 1 / (1 + K_m*muc_free)                
                p_pfu = inputs_dict['p_pfu']
                p_cell = inputs_dict['p_cell']
                vaccine_efficacy = inputs_dict['vaccine_efficacy']    
                
                infection_dict__onescenario[cc][gridding][dispersion_type]['p_infect'] = 1 - (1 - (F_trans * F_v * p_pfu * F_c * p_cell*(1-vaccine_efficacy)))**(dNdt_cov*dose_per_emit)
                
    return infection_dict__onescenario



# =============================================================================
# evolution of a single gaussian puff
# =============================================================================
def get_puff_funs(df_pos,inputs_dict,error_thresh=1e-8,N_samples_min=1000):
#    D_mouth, K = get_select_inputs(inputs_dict,['D_mouth','K'])
    D_mouth, = get_select_inputs(inputs_dict,['D_mouth'])    
    K = 0.
    y0_source = inputs_dict['W']/2.
    x0_source = inputs_dict['L']/2.
    z0_source = inputs_dict['z0_source']
    
    t_part = df_pos['x'].index
    x_part = df_pos['x'].values + x0_source
    y_part = df_pos['y'].values + y0_source
    z_part = df_pos['z'].values + z0_source
    
    val,idx = np.unique(x_part,return_index=True)
    z0_fun = lambda x: interp1d(x_part[idx],z_part[idx],fill_value='extrapolate')(x)
    s0_source = get_s(x0_source,z0_fun)
    s_part = np.array([get_s(onex,z0_fun) for onex in x_part])
    b_part = np.array([get_b(s - s0_source, D_mouth, beta=0.114, K=K,t=t) for (s,t) in zip(s_part,t_part)])
    
    vals,idx = np.unique(t_part,return_index=True)
    t_part = t_part[idx]
    b_part = b_part[idx]
    x_part = x_part[idx]
    y_part = y_part[idx]
    z_part = z_part[idx]
    
    ts1,bs = reduce_data(t_part,b_part,error_thresh=error_thresh,N_samples_min=N_samples_min)
    b_vs_t = lambda t: interp1d(ts1,bs,fill_value='extrapolate')(t)
    
    ts2,xs = reduce_data(t_part,x_part,error_thresh=error_thresh,N_samples_min=N_samples_min)    
    x0_vs_t = lambda t: interp1d(ts2,xs,fill_value='extrapolate')(t)
    
    ts3,ys = reduce_data(t_part,y_part,error_thresh=error_thresh,N_samples_min=N_samples_min)    
    y0_vs_t = lambda t: interp1d(ts3,ys,fill_value='extrapolate')(t)
    
    ts4,zs = reduce_data(t_part,z_part,error_thresh=error_thresh,N_samples_min=N_samples_min)    
    z0_vs_t = lambda t: interp1d(ts4,zs,fill_value='extrapolate')(t)
    
    return x0_vs_t, y0_vs_t, z0_vs_t, b_vs_t

def get_conc(t,x,y,z,x0_vs_t, y0_vs_t, z0_vs_t, b_vs_t, Dp_vs_t, inputs_dict, just_reflections=False,N_reflections=0,wp=0.,dzdt_vs_t=lambda t: 0.,Dp_noreflect=3e-5,exponent_BL = 0.26,prefactor_BL = 0.06,well_mixed=False):
    L = inputs_dict['L']
    W = inputs_dict['W']
    H = inputs_dict['H']    
    
    V_room = L*W*H
    A_sedi = L*W
    A_wall = L*W*2 + L*H*2 + W*H*2
    
    if N_reflections > 0 or well_mixed:
        diffusion_coeff = get_diffusion_coefficient(Dp_vs_t(t),inputs_dict['Tv_inf'])
        # xxx Fuchs (1964) = (brown diffusion coeff * wall area)/(BL thickness * V_room) + w_sedi/H
#        k_sedi = dzdt_vs_t(t)*A_sedi/V_room
        k_sedi = wp*A_sedi/V_room        
        chamber_diff_BL_thick = (prefactor_BL*diffusion_coeff**exponent_BL)
        k_wall = (diffusion_coeff * A_wall) / (chamber_diff_BL_thick * V_room)
        k_acr = inputs_dict['acr']/3600.
    else:
        k_sedi = 0.
        k_wall = 0.    
        k_acr = 0.
        
    if well_mixed:
        c_at_t = np.exp(-(k_acr + k_wall + k_sedi)*t)/V_room
    else:
        Rx,Ry,Rz = get_reflection_terms(
                t,x,y,z,x0_vs_t, y0_vs_t, z0_vs_t, b_vs_t, Dp_vs_t,
                inputs_dict, N_reflections=N_reflections, Dp_noreflect=Dp_noreflect)
        if just_reflections:
            
            if Dp_vs_t(1.)>Dp_noreflect:
                c_at_t = 0.*t
            else:
                c_at_t = (Rx*Ry*Rz - np.exp(-(x-x0_vs_t(t))**2/b_vs_t(t)**2)*np.exp(-(y-y0_vs_t(t))**2/b_vs_t(t)**2)*np.exp(-(z-z0_vs_t(t))**2/b_vs_t(t)**2))*(np.exp(-(k_acr + k_wall + k_sedi)*t)/((np.pi*b_vs_t(t)**2)**(3./2.)))
        else:
            c_at_t = Rx*Ry*Rz*(np.exp(-(k_acr + k_wall + k_sedi)*t)/((np.pi*b_vs_t(t)**2)**(3./2.)))

    return c_at_t


def get_reflection_terms(t,x,y,z,x0_vs_t, y0_vs_t, z0_vs_t, b_vs_t, Dp_vs_t, inputs_dict, N_reflections=0,Dp_noreflect=30e-6):
    L = inputs_dict['L']
    W = inputs_dict['W']
    H = inputs_dict['H']
    
    if N_reflections>0:
        reflect = True
    else:
        reflect = False
        
    if reflect:
        if Dp_vs_t(1.)<Dp_noreflect:
            Rx = sum([np.exp(-(x+2*i*L-x0_vs_t(t))**2/b_vs_t(t)**2) + np.exp(-(x+2*i*L+x0_vs_t(t))**2/b_vs_t(t)**2) for i in np.arange(-N_reflections,N_reflections)])
            Ry = sum([np.exp(-(y+2*i*W-y0_vs_t(t))**2/b_vs_t(t)**2) + np.exp(-(y+2*i*W+y0_vs_t(t))**2/b_vs_t(t)**2) for i in np.arange(-N_reflections,N_reflections)])
            Rz = sum([np.exp(-(z+2*i*H-z0_vs_t(t))**2/b_vs_t(t)**2) + np.exp(-(z+2*i*H+z0_vs_t(t))**2/b_vs_t(t)**2) for i in np.arange(-N_reflections,N_reflections)])
        else:
            Rx = np.exp(-(x-x0_vs_t(t))**2/b_vs_t(t)**2)
            Ry = np.exp(-(y-y0_vs_t(t))**2/b_vs_t(t)**2)
            Rz = np.exp(-(z-z0_vs_t(t))**2/b_vs_t(t)**2)
    else:
        Rx = np.exp(-(x-x0_vs_t(t))**2/b_vs_t(t)**2)
        Ry = np.exp(-(y-y0_vs_t(t))**2/b_vs_t(t)**2)
        Rz = np.exp(-(z-z0_vs_t(t))**2/b_vs_t(t)**2)
    
    return Rx,Ry,Rz

def get_ts(conc_fun,fixed_ts=False,tlims_log=[-10,3],thresh=1e-5,N_loops = 2,Nts=int(1e4)):
    ts = np.hstack([0.,np.logspace(tlims_log[0],tlims_log[1],int(Nts))])
    for ii in range(N_loops):
        cumsum_vals = np.cumsum(conc_fun(ts))/sum(conc_fun(ts))
        idx_min = np.where(cumsum_vals<=thresh)[0]
        if len(idx_min)>0:
            idx_min = idx_min[-1]
        else:
            idx_min = 0
        
        idx_max = np.where(cumsum_vals>=(1.-thresh))[0]
        if len(idx_max)>0:
            idx_max = idx_max[0]
        else:
            idx_max = int(len(ts)-1)
        dlog10t = np.log10(ts[2]) - np.log10(ts[1])
        if ts[idx_min] == 0.:
            ts = np.hstack([0.,np.logspace(np.log10(ts[idx_min+1])-2.*dlog10t,np.log10(ts[idx_max])+dlog10t,int(Nts))])
        else:
            ts = np.logspace(np.log10(ts[idx_min])-dlog10t,np.log10(ts[idx_max])+dlog10t,int(Nts))
    ts = np.linspace(min(ts),max(ts),int(Nts))
    
    dt = ts[1] - ts[0]
    
    return ts,dt

# =============================================================================
# nasal depsition efficieincy
# =============================================================================

def get_nasal_deposition_efficiency(Dp,Dd,rho_aero,V_breathe,Tv,dep_coeffs=np.array([-0.00309,-16.6,.5,-.28])):
    a = dep_coeffs[0]
    b = dep_coeffs[1]
    c = dep_coeffs[2]
    d = dep_coeffs[3]    
    Q_lpm = V_breathe*1e3*60 # inhalation rate, lpm
    da = get_aerodynamic_diam(Dp,np.zeros(Dp.shape),rho_aero)*1e6 # aerodynamic diameter in um
    diffusion_coeff = get_diffusion_coefficient(Dp,Tv)*100**2    # diffusion coeff in cm^2
    DE_nasal = 1 - np.exp(a*da**2*Q_lpm + b*diffusion_coeff**c*Q_lpm**d)
    if len(Dp)>1:
        DE_nasal[DE_nasal<0.] = 0.
    else:
        if DE_nasal<0.:
            DE_nasal = 0.
        elif DE_nasal>1:
            DE_nasal = 1.
    return DE_nasal


# =============================================================================
# mask efficiency
# =============================================================================

def get_mask_efficiency(D,direction='out',mask_type='surgical',excel_filename='../input/inputs.xlsx',relative_std_dev=None,fill_val='conservative',return_std_dev=True):
    df_avg = pd.read_excel(excel_filename, sheet_name='FE_' + direction + '_average')
    df_ub = pd.read_excel(excel_filename, sheet_name='FE_' + direction + '_upper')
    df_lb = pd.read_excel(excel_filename, sheet_name='FE_' + direction + '_lower')
    
    dias = df_avg['diameter (um)'].values*1e-6
    FEs_avg = df_avg[mask_type].values
    FEs_lb = df_lb[mask_type].values
    FEs_ub = df_ub[mask_type].values
    
    if relative_std_dev == None:
        relative_std_dev = np.random.normal()
    
    FE_std_devs = FEs_ub - FEs_lb
    FEs = FEs_avg + FE_std_devs*relative_std_dev
    if fill_val == 'conservative':
        mask_FE = interp1d(np.log10(dias),FEs,fill_value='extrapolate')(np.log10(D))
        mask_FE[D>max(dias)] = FEs[-1]
    elif fill_val == 'extrapolate':
        mask_FE = interp1d(np.log10(dias),FEs,fill_value='extrapolate')(np.log10(D))    
    mask_FE[mask_FE<0.] = 0.
    mask_FE[mask_FE>1.] = 1.
    
    if mask_type == 'none':
        mask_FE = 0.*D
        relative_std_dev = 0.
    if return_std_dev:
        return mask_FE,relative_std_dev
    else:
        return mask_FE

def get_inactivation_rate(Dd, Dp0, Dp, rho_aero, tkappa, Tv_inf, S_inf, E_a, A_eff, A_sol):
    mp0 = get_mass_from_diameter(Dd,Dp0,rho_aero) #initial droplet mass
#    mp = get_mass_from_diameter(Dd,Dp,rho_aero) # current droplet mass
    Dp_equil = get_equilibrium_diameter(Dd,tkappa,Tv_inf,S_inf)[0] #equilibrium droplet diameter
    mp_equil = get_mass_from_diameter(Dd,Dp_equil,rho_aero) #equilibrium droplet masss
    m_s = get_mass_from_diameter(Dd,Dd,rho_aero)
    
    conc_fac = (mp0 - m_s) / (mp_equil - m_s) #concentration factor (= [Seq]/[S0])    
#    conc_fac = (mp0 - m_s)/(mp - m_s)
    
    if S_inf < .45: #effloresced region, retrieve k_eff [/h]
        k_h = A_eff * np.exp(-E_a / (R*Tv_inf))
    else: #solution region, retrieve k_sol [/h]
        k_h = conc_fac * A_sol * np.exp(-E_a / (R*Tv_inf))
    k = k_h/3600 #inactivation rate time conversion [/sec]
    return k

#def get_inactivation_rate__step(S_inf):
#    if S_inf < 0.45:
#        k_h = -np.log(0.5)/4./3600.
#    elif S_inf < 0.75:
#        k_h = -np.log(0.5)/1./3600.
#    else:
#        k_h = -np.log(0.5)/4./3600.
#    return k_h

def get_inactivation_rate__step(S_inf,Tv_inf,represent_uncertainty=True):
    T_low = 16+273.15 #289.15
    T_high = 25+273.15 #298.15
    S_low = 0.45
    S_high = 0.75
    if Tv_inf<T_low:
        if S_inf<S_low:
            tau_median = 26.55
            tau__2_5 = 20.28
            tau__97_5 = 38.75
        elif S_inf<S_high:
            tau_median = 14.22
            tau__2_5 = 12.17
            tau__97_5 = 17.16           
        else:
            tau_median = 13.78           
            tau__2_5 = 10.67
            tau__97_5 = 19.70           
    elif Tv_inf<T_high:
        if S_inf<S_low:
            tau_median = 6.43           
            tau__2_5 = 5.52
            tau__97_5 = 7.56
        elif S_inf<S_high:
            tau_median = 2.41 
            tau__2_5 = 2.03
            tau__97_5 = 2.88
        else:        
            tau_median = 7.50
            tau__2_5 = 6.22
            tau__97_5 = 9.24
    else:
        if S_inf<S_low:
            tau_median = 3.43
            tau__2_5 = 2.91
            tau__97_5 = 4.12
        elif S_inf<S_high:
            tau_median = 1.52
            tau__2_5 = 1.05
            tau__97_5 = 2.14
        else:        
            tau_median = 2.79
            tau__2_5 = 2.12
            tau__97_5 = 2.78
    loggsd__97_5 = (np.log10(tau__97_5) - np.log10(tau_median))/(np.sqrt(2)*erfinv(2*0.975 - 1.))
    loggsd__2_5 = (np.log10(tau__2_5) - np.log10(tau_median))/(np.sqrt(2)*erfinv(2*0.025 - 1.))
    log_gsd = (loggsd__97_5 + loggsd__2_5)/2.
#    print(log_gsd,loggsd__97_5,loggsd__2_5)
    if represent_uncertainty:
        tau_half_hours = 10**np.random.normal(np.log10(tau_median),log_gsd)
    else:
        tau_half_hours = tau_median
    k = -np.log(0.5)/(tau_half_hours*3600.)
    return k
# =============================================================================
#   ADDITIONAL FUNCTIONS
#    miscelaneous functions go here
#    
# =============================================================================

# =============================================================================
#  functions for turbulent jet
# =============================================================================
def get_z_centerline__nosedi(x,u0,D_mouth,Tv_source,S_source,x_co2_source,Tv_inf,S_inf,x_co2_inf,p=101325.):
    A0 = np.pi/4.*D_mouth**2
    rho_source = get_air_density(Tv_source,S_source,x_co2=x_co2_source,p=p)
    rho_inf = get_air_density(Tv_inf,S_inf,x_co2=x_co2_inf,p=p)
    Ar0 = (g*np.sqrt(A0)/u0**2)*((rho_inf - rho_source)/rho_source)
    z = 0.0354*Ar0*np.sqrt(A0)*(x/np.sqrt(A0))**3*np.sqrt(Tv_source/Tv_inf)
    return z

def get_dz0dx(x,u0,D_mouth,Tv_source,S_source,x_co2_source,Tv_inf,S_inf,x_co2_inf,p=101325.):
    A0 = np.pi/4.*D_mouth**2
    rho_source = get_air_density(Tv_source,S_source,x_co2=x_co2_source,p=p)
    rho_inf = get_air_density(Tv_inf,S_inf,x_co2=x_co2_inf,p=p)
    Ar0 = (g*np.sqrt(A0)/u0**2)*((rho_inf - rho_source)/rho_source)
    dz0dx = 0.0354*Ar0*np.sqrt(A0)*(3*x**2/np.sqrt(A0)**3)*np.sqrt(Tv_source/Tv_inf)
    return dz0dx

def jet_dispersion__velocity(s,r,D_mouth,u0,K=0,beta=0.114):
    # disperse Qv, where Qv is either temperature or mixing ratio    
    # model by Lee and Chu, https://play.google.com/books/reader?id=PUztBwAAQBAJ&hl=en&pg=GBS.PA35
    b = get_b(s,D_mouth,K=K,beta=beta) # Lee and Chu 
    if s < 6.2*D_mouth:
        # flow establishment zone
        R_uni = D_mouth/2 - s/12.4
        if r<=R_uni:
            u = u0
        else:
            u = u0*np.exp(-(r-R_uni)**2/b**2)
    else:
        # established flow zone
        um = u0*6.2*(s/D_mouth)**(-1)
        u = um*np.exp(-(r/b)**2)
    return u

    
def jet_dispersion__concentration(x,y,z,z0_fun,s_fun,u0,c0,Tv_inf,S_inf,Tv_source,S_source,D_mouth=2/100.,p=101325.,K=0.):
    # model by Lee and Chu, https://play.google.com/books/reader?id=PUztBwAAQBAJ&hl=en&pg=GBS.PA35
    # 
    x0 = get_x0(x,z,z0_fun,u0,Tv_inf,S_inf,Tv_source,S_source,D_mouth=D_mouth,p=p)
    s = s_fun(x0)
    r = get_r(x,y,z,x0,0.,z0_fun(x0))
    
    b = get_b(s,D_mouth,K=K) # Lee and Chu 
    lamb = 1.2 # Papanicolaou, Panos N., and E. John List. "Investigations of round vertical turbulent buoyant jets." Journal of Fluid Mechanics 195 (1988): 341-391.
    
    if s < 6.2*D_mouth:
        # flow establishment zone
        R_uni = D_mouth/2 - s/12.4
        if r<=R_uni:
            c = c0
        else:
            c = c0*np.exp(-(r-R_uni)**2/(lamb**2*b**2))
    else:
        # established flow zone
        cm = 5.26*c0*D_mouth/s#c0*(1 - lamb**2)/(2*np.sqrt(2)*lamb**2*beta)*(D_mouth/s)#get_cm(s,c0,lamb,beta,D_mouth) #c0*(1 - lamb**2)/(2*np.sqrt(2)*lamb**2*beta)*(D_mouth/s)
        c = cm*np.exp(-(r/(lamb*b))**2)

    return c

def get_b(s,D_mouth,beta = 0.114, K=0.,t=0.):
    if s>6.2*D_mouth:
        return np.sqrt(beta**2*s**2 + 4*K*t)
    else:
        return 0.5*D_mouth + np.sqrt(0.033**2*s**2 + 4*K*t)
def get_s(x0,z0_fun):
    Nxs = int(np.round(max([100,x0*100])))
    x0s = np.linspace(0.,x0,Nxs)
    z0s = z0_fun(x0s)
    s = np.sum(np.sqrt(np.diff(x0s)**2 + np.diff(z0s)**2))
    return s

def get_x0(x,z,z0_fun,u0,Tv_inf,S_inf,Tv_source,S_source,D_mouth=2/100,p=101325.):
    distance_fun = lambda x0: ((x - x0)**2 + (z - z0_fun(x0))**2)    
    
    x0 = fmin(distance_fun,x,disp=False)
    if len(x0) == 1:
        x0 = x0[0]
    return x0

def get_r(x,y,z,x0,y0,z0):
    r = np.sqrt((x0-x)**2 + (y0-y)**2 + (z0-z)**2)
    return r

def get_u0_reduction__best(mask_type='surgical',excel_filename='../input/inputs.xlsx'):
    df_u0_mask = pd.read_excel(excel_filename, sheet_name='u0_mask')
    idx, = np.where([mask_type == 'surgical' for mask_type in df_u0_mask['mask_type']])
    u0_reduction = df_u0_mask['u0_reduction'][idx].values[0]
    return u0_reduction    
    
def get_u0_mask(u0,mask_type='surgical',excel_filename='../input/inputs.xlsx',u0_reduction=None):
    if mask_type == 'none':
        u0_mask = u0
    else:
        if u0_reduction == None:
            df_u0_mask = pd.read_excel(excel_filename, sheet_name='u0_mask')
            idx, = np.where([mask_type == 'surgical' for mask_type in df_u0_mask['mask_type']])
            u0_mean = df_u0_mask['u0_reduction'][idx].values[0]
            u0_std = df_u0_mask['std_dev'][idx].values[0]
            u0_reduction = np.random.normal(loc=u0_mean,scale=u0_std)
        if u0_reduction>1.:
            u0_reduction=1.
        u0_mask = u0*u0_reduction
    return u0_mask


# =============================================================================
# compute air properties
# =============================================================================
def get_mean_free_path():
    # assume mean free path is constant, i.e. doesn't vary with temp or pressure (within simulation range)
    return 68e-9
    

def get_dynamic_viscosity_of_air(Tv):
    # http://www-mdp.eng.cam.ac.uk/web/library/enginfo/aerothermal_dvd_only/aero/fprops/propsoffluids/node5.html
    
    mu0 = 17.15e-6
    T0 = 273.15
    mu = mu0*(Tv/T0)**0.7
    return mu

def get_air_density(Tv,S,x_co2=410e-6,p=101325.):
    x_h2o = get_h2o_mixing_ratio_from_Sv(S,Tv,p=p)
    rho_air = (M_dry_air*(1-x_h2o-x_co2) + x_h2o*M_h2o + x_co2*M_co2)*p/(R*Tv)#*(1+x_h2o+x_co2))
    
    return rho_air 

def get_saturation_vapor_pressure(Tv_K):

    Tv = Tv_K-273.15
    p_sat = 611.21*np.exp((18.678 - Tv/234.5)*(Tv/(257.14+Tv)))
    return p_sat # Pa

def get_vapor_pressure(S,Tv):
    p_sat = get_saturation_vapor_pressure(Tv)
    p_v = S*p_sat
    return p_v # Pa

def get_h2o_mixing_ratio_from_Sv(S,Tv,p=101325.):
    p_v = get_vapor_pressure(S,Tv)
    mixing_ratio = p_v/p#(p - p_v)
    return mixing_ratio
    
# =============================================================================
# compute particle microphysical and water uptake properties
# =============================================================================
def get_diameter_from_mass(Dd,mp,rho_aero):
    return ((mp*6/np.pi - Dd**3*(rho_aero - rho_h2o))/rho_h2o)**(1./3.)

def get_mass_from_diameter(Dd,D,rho_aero):
    return np.pi/6*Dd**3*rho_aero + np.pi/6*(D**3 - Dd**3)*rho_h2o

def get_drymass_from_drydiameter(Dd,rho_aero=1000.):
    return np.pi/6*Dd**3*rho_aero #(6/np.pipi*mp/rho_aero)**(1/3)

def get_particle_density(D,Dd,rho_aero=1000.):
    rho_p = (rho_aero*Dd**3 + (D**3 - Dd**3)*rho_h2o)/D**3
    return rho_p

def get_droplet_saturation_ratio(D,Dd,tkappa,Td):
    S_d = ((D**3 - Dd**3)/(D**3 - Dd**3*(1-tkappa)))*np.exp(4*sig_h2o*M_h2o/(R*Td*rho_h2o*D))
    return S_d

def get_droplet_vapor_pressure(D,Dd,tkappa,Td):
    p_sat = get_saturation_vapor_pressure(Td)
    S_d = get_droplet_saturation_ratio(D,Dd,tkappa,Td)
    p_d = S_d*p_sat
    return p_d
        
def get_particle_Sc(Dp,Tv,S,x_co2=3.65e-6,p=101325.):
    rho_air = get_air_density(Tv,S,x_co2=x_co2,p=p)
    mu = get_dynamic_viscosity_of_air(Tv)
    D_AB = get_water_vapor_diffusion_coefficient(Tv) # D_inf
    Sc = mu/(rho_air*D_AB)
    return Sc

    
def get_heat_capacity_of_air(S,Tv):
    Cp_dry_air = 1006.#1030.5 - 0.11975*Tv + 3.9734e-4*Tv**2 # http://www.mhtlab.uwaterloo.ca/pdf_reports/mhtl_G01.pdf
    H = get_specific_humidity(S,Tv,p=101325.)
    Cp = Cp_dry_air + 1.82*H # https://en.wiktionary.org/wiki/humid_heat
    return Cp
    
def get_specific_humidity(S,Tv,p=101325.):
    p_sat = get_saturation_vapor_pressure(Tv)
    p_h2o = p_sat*S
    specific_humidity = p_h2o*M_h2o/(p_h2o*M_h2o + (p-p_h2o)*M_dry_air)
    return specific_humidity
    
def get_particle_heat_capacity(m,m_aero,Cp_aero=853):
    return (Cp_aero*m_aero + (m-m_aero)*Cp_h2o)/m

def get_aerodynamic_diam(Dp,Dd,rho_aero):
    #computes the aerodymanic diameter
    rho_p = get_particle_density(Dp,Dd,rho_aero=rho_aero)    
    rho_0 = 1000.#
    Cc = get_cunningham_correction_factor(Dp)
    X = 1/Cc
    Da = Dp*((rho_p/(rho_0*X))**(0.5))
    return Da #[um]
# =============================================================================
# compute particle fluid dynamic properties
# =============================================================================
def get_cunningham_correction_factor(Dp):
    # eqn. 9.34 from Seinfeld and Pandis
    mean_free_path = get_mean_free_path();
    Cc = 1 + (2*mean_free_path/Dp)*(1.257 + 0.4*np.exp(-1.1*Dp/(2*mean_free_path)))    
    return Cc

def get_drag_coefficient(up,ug,Dp,Tv,S):
    Re = get_particle_Re(up - ug,Dp,Tv,S)
    if Re == 0:
        Cd = 0.
    elif Re<=1.:
        Cd = 24/Re
    else:
        Cd = (1 + 0.15*Re**0.687)*24/Re
    return Cd

def get_particle_Re(u_diff,Dp,Tv,S,x_co2=410e-6,p=101325.):
    rho_air = get_air_density(Tv,S,x_co2=x_co2,p=p)
    mu = get_dynamic_viscosity_of_air(Tv)
    nu = mu/rho_air
    
    Re = abs(u_diff)*Dp/nu
    return Re

def get_particle_Pr(Tv,S):
    Cp_v = get_heat_capacity_of_air(S,Tv)
    mu = get_dynamic_viscosity_of_air(Tv)
    Pr = Cp_v*mu/Kg # note: Kg = constant right now; should vary with S and T
    return Pr

def get_relaxation_time(mp,Dp,Tv):
    mu_air = get_dynamic_viscosity_of_air(Tv)
    Cc = get_cunningham_correction_factor(Dp)
    tau_relax = mp*Cc/(3*np.pi*mu_air*Dp)
    return tau_relax
    
def get_terminal_velocity(Dp,rho_p,Tv_inf,S_inf,x_co2=410e-6,p=101325.):
    # eqn. 9.49 from Seinfeld and Pandis    
    rho_g = get_air_density(Tv_inf,S_inf,x_co2=x_co2,p=p)
    wp_terminal = fsolve(lambda wp: wp - np.sqrt(g*(4*Dp*rho_p)/(3*rho_g*get_drag_coefficient(wp,0.,Dp,Tv_inf,S_inf))),1.)
    return wp_terminal


def get_water_vapor_diffusion_coefficient(Tv):
    # https://www.researchgate.net/post/Binary_diffusion_coefficients_for_water_vapour_in_air_at_normal_pressure
    diffusion_coeff = 0.211/100**2 * (Tv/273.15)**1.94 # Pruppacher and Klett
    return diffusion_coeff
    
def get_particle_diffusion_coefficient(D,Tv):
    # eqn. 9.73 from Seinfeld and Pandis
    mu_air = get_dynamic_viscosity_of_air(Tv)
    Cc = get_cunningham_correction_factor(D)
    diffusion_coeff = k*Tv*Cc/(3*np.pi*mu_air*D)
    return diffusion_coeff

def get_effective_heat_capacity(m,m_aero,Cp_aero=853):
    Cp_h2o = 4.184*1000. #specific heat of water [J/kg]
    return (Cp_aero*m_aero + (m-m_aero)*Cp_h2o)/m


def get_diffusion_coefficient(Dp,Tv):
    # eqn. 9.73 from Seinfeld and Pandis
    mu_air = get_dynamic_viscosity_of_air(Tv)
    Cc = get_cunningham_correction_factor(Dp)
    diffusion_coeff = k*Tv*Cc/(3*np.pi*mu_air*Dp)
    return diffusion_coeff