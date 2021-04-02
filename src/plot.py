#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Laura Fierce
"""
import QuaRAD
from scipy.interpolate import interp1d, interp2d
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
from scipy.stats import norm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import utils


def fig0(scenario_dir,qq_vals='all',dispersion_type='both',colorscale='log',normalize=False,fontsize=11):
# =============================================================================
# average concenration associated with quadrature point q per emission rate of these particles
# units: m^{-3}/(#/s)
# =============================================================================
    
    fig_dir = get_fig_dir()
    fig,ax = plt.subplots(1,constrained_layout=True)
    fig.set_size_inches(10.,3.)
    normconcs_filename = scenario_dir + '02_normconcs.pkl'
    normconcs_dict = pickle.load(open(normconcs_filename, 'rb'))
    dxs = normconcs_dict[0]['vs_xz']['dxs']
    dzs = normconcs_dict[0]['vs_xz']['dzs']
    N_tot = 5000
    ws_cov = normconcs_dict[0]['inputs']['w_cov']
    ws = normconcs_dict[0]['inputs']['w']    
    D0= normconcs_dict[0]['inputs']['D']    
    for qqq,qq in enumerate(np.argsort(normconcs_dict[0]['inputs']['Dd'])):#range(N_part):
        if dispersion_type == 'both':
            C =1./ws_cov[qqq]*normconcs_dict[0]['vs_xz']['Cq_per_emit']['jet'][:,0,:,qqq] + normconcs_dict[0]['vs_xz']['Cq_per_emit']['reflections'][0,qqq]
        elif dispersion_type == 'jet':
            C = 1./ws_cov[qqq]*normconcs_dict[0]['vs_xz']['Cq_per_emit']['jet'][:,0,:,qqq]
        elif dispersion_type == 'reflections':
            C = 1./ws_cov[qqq]*np.ones(dxs.shape)*normconcs_dict[0]['vs_xz']['Cq_per_emit']['reflections'][0,qqq]
        
        N_parts = int(N_tot*ws[qqq])
        C_cumsum_x = np.cumsum(np.sum(C,axis=1),axis=0)/sum(sum(C))
        xs = interp1d(C_cumsum_x,dxs,fill_value='extrapolate')(np.random.rand(N_parts))
        zs = np.zeros(xs.shape)
        for ii,x_i in enumerate(xs):
            C_i = interp2d(dzs,dxs,C)(dzs,x_i)
            C_cumsum_z = np.cumsum(C_i)/sum(C_i)
            z_i = interp1d(C_cumsum_z,dzs,fill_value='extrapolate')(np.random.rand(1))
            zs[ii] = z_i
        ax.scatter(xs,zs,D0[qqq]*1.5e5,'C3',alpha=0.2)
    ax.set_frame_on(False)
    ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    print('fig 0 saved: ' + fig_dir + 'scenario__particle_scatter.png')
    fig.savefig(fig_dir+'scenario__particle_scatter.png',dpi=1000)
    
def fig1(scenario_dir,fontsize=11):
    # =============================================================================
    # quadrature approximation of expelleds particle
    # =============================================================================

    fig_dir = get_fig_dir()    
    normconcs_filename = scenario_dir + '02_normconcs.pkl'
    normconcs_dict = pickle.load(open(normconcs_filename, 'rb'))
    
    ii = 0
    cols = ['C0','C1','C2']
    three_modes = ['b','l','o']
    inputs = normconcs_dict[ii]['inputs']
    Ns = inputs['Ns']
    mus = inputs['mus']
    sigs = inputs['sigs']    
    
    
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches([3.,8.])
    
    gs = fig.add_gridspec(22,1)
    ax1 = fig.add_subplot(gs[0:7,0])
    ax2 = fig.add_subplot(gs[7:14,0])
    ax3 = fig.add_subplot(gs[15:21,0])
    
    modes = inputs['mode']
#    x_vals = [0.,0.1,0.2]
    dias = np.logspace(-8,-2,1000)
    hln = ['']*len(Ns)
    for mm,(one_mode,one_N,mu,sig) in enumerate(zip(three_modes,Ns,mus,sigs)):
        idx, = np.where([a_mode==one_mode for a_mode in modes])
        dNdD = lambda D: one_N/sum(Ns)*norm(loc=mu,scale=sig).pdf(np.log10(dias))        
        hln[mm], = ax1.plot(dias,dNdD(dias),color=cols[mm]); 
        ax2.stem(inputs['D'][idx],inputs['w'][idx],linefmt=cols[mm],markerfmt=cols[mm]+'o',basefmt='');
        ax3.stem(inputs['D'][idx],inputs['w_cov'][idx],linefmt=cols[mm],markerfmt=cols[mm]+'o',basefmt='');
        
        if mm == 0:
            axin1 = ax1.inset_axes([0.65,0.65,0.3,0.3])
            axin2 = ax2.inset_axes([0.65,0.65,0.3,0.3])
            axin3 = ax3.inset_axes([0.65,0.65,0.3,0.3])
            
        axin1.plot(dias,dNdD(dias),color=cols[mm])
        axin2.stem(inputs['D'][idx],inputs['w'][idx],linefmt=cols[mm],markerfmt=cols[mm]+'o',basefmt='');
        axin3.stem(inputs['D'][idx],inputs['w_cov'][idx],linefmt=cols[mm],markerfmt=cols[mm]+'o',basefmt='');        

        if mm == 2:
            x1, x2, y1, y2 = 2.2e-5,9e-4,0.,ax1.get_ylim()[1]/50
            axin1.set_xlim(x1, x2)
            axin1.set_ylim(y1, y2)
            
            y2 = ax2.get_ylim()[1]/50
            axin2.set_xlim(x1, x2)
            axin2.set_ylim(y1, y2)
            
            y2 = ax3.get_ylim()[1]/15
            axin3.set_xlim(x1, x2)
            axin3.set_ylim(y1, y2)
    
    ax1.set_ylim([0.,ax1.get_ylim()[1]])
    ax2.set_ylim([0.,ax2.get_ylim()[1]])
    ax3.set_ylim([0.,ax3.get_ylim()[1]])    
    
    ax1.set_xscale('log')
    ax2.set_xscale('log')
    ax3.set_xscale('log')    
    ax1.set_xticklabels('')
    ax2.set_xticklabels('')    
    ax1.set_xlim([5e-7,1e-3])
    ax2.set_xlim(ax1.get_xlim())
    ax3.set_xlim(ax1.get_xlim())

    axin1.set_xscale('log')
    axin2.set_xscale('log')
    axin3.set_xscale('log')    

    ax1.set_ylabel('particle size distribution',fontsize=fontsize)
    ax2.set_ylabel('particle weights',fontsize=fontsize)
    ax3.set_ylabel('virion weights',fontsize=fontsize)   
    
    ax3.set_xlabel('diameter [m]',fontsize=fontsize)
    
    
    ax1.indicate_inset_zoom(axin1)
    ax2.indicate_inset_zoom(axin2)
    ax3.indicate_inset_zoom(axin3)    
    ax1.legend(hln,['b-mode','l-mode','o-mode'],
                     bbox_to_anchor=(-0.1,1.1,1.2,0.2), loc="lower left",handlelength=1.,
                    mode="expand", borderaxespad=0., ncol=3,handletextpad=0.2,frameon=False)
    shrinkx = 0.03
    shrinky = 0.9    
    ax1.annotate('A', xy=get_axlab_locs(ax1, shrinkx=shrinkx,shrinky=shrinky),fontsize=fontsize)
    ax2.annotate('B', xy=get_axlab_locs(ax2, shrinkx=shrinkx,shrinky=shrinky),fontsize=fontsize)
    ax3.annotate('C', xy=get_axlab_locs(ax3, shrinkx=shrinkx,shrinky=shrinky),fontsize=fontsize)
    
    fig.savefig(fig_dir + 'expelled_aerosol.png',dpi=500)
    print('fig 1 saved: ' + fig_dir + 'expelled_aerosol.png')
    
def fig2(scenario_dir,fontsize=11,print_fig=False):
    # =============================================================================
    # diameter vs. time for quadrature approximation
    # =============================================================================
    
    fig_dir = get_fig_dir()
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(3.5,4.)
    
    trajectories_filename = scenario_dir + '01_trajectories.pkl'
    trajectories_dict = pickle.load(open(trajectories_filename, 'rb'))
    
    default_case_options = {'smask_type':'none','vary_Tv_and_S':False}#,'acr_enhancement':1.}
    case_idx = utils.get_select_cases(trajectories_dict,default_case_options,return_idx=True)[0]
    gs = fig.add_gridspec(2,1)
    ax = fig.add_subplot(gs[0,:])
    ax_false = fig.add_subplot(gs[:,:])
    Dd = trajectories_dict[case_idx]['inputs']['Dd']
    t_vals = np.logspace(-4,2,5000)
    hln = ['']*len(Dd)
    for qq in range(len(Dd)):
        ts = trajectories_dict[case_idx][qq]['evap']['mp'].index
        Dps = QuaRAD.get_diameter_from_mass(Dd[qq],trajectories_dict[case_idx][qq]['evap']['mp'].values,trajectories_dict[case_idx]['inputs']['rho_aero'])
        Dp_vs_t = QuaRAD.get_Dp_fun(ts,Dps)
        if trajectories_dict[case_idx]['inputs']['mode'][qq] == 'b':
            col = 'C0'
        elif trajectories_dict[case_idx]['inputs']['mode'][qq] == 'l':
            col = 'C1'            
        elif trajectories_dict[case_idx]['inputs']['mode'][qq] == 'o':
            col = 'C2'            
            
        hln[qq], = ax.plot(t_vals,Dp_vs_t(t_vals),col)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim([min(t_vals),max(t_vals)])
    plt.legend(
            [hln[0],hln[1],hln[-1]],['b-mode','l-mode','o-mode'],loc='upper left')
        
    default_case_options = {'smask_type':'none','vary_Tv_and_S':True}#,'acr_enhancement':1.}
    case_idx = utils.get_select_cases(trajectories_dict,default_case_options,return_idx=True)[0]
    ax = fig.add_subplot(gs[1,:])
    Dd = trajectories_dict[case_idx]['inputs']['Dd']
    t_vals = np.logspace(-4,2,5000)
    hln = ['']*len(Dd)
    for qq in range(len(Dd)):
        ts = trajectories_dict[case_idx][qq]['evap']['mp'].index
        Dps = QuaRAD.get_diameter_from_mass(Dd[qq],trajectories_dict[case_idx][qq]['evap']['mp'].values,trajectories_dict[case_idx]['inputs']['rho_aero'])
        Dp_vs_t = QuaRAD.get_Dp_fun(ts,Dps)
        if trajectories_dict[case_idx]['inputs']['mode'][qq] == 'b':
            col = 'C0'
        elif trajectories_dict[case_idx]['inputs']['mode'][qq] == 'l':
            col = 'C1'            
        elif trajectories_dict[case_idx]['inputs']['mode'][qq] == 'o':
            col = 'C2'            
            
        hln[qq], = ax.plot(t_vals,Dp_vs_t(t_vals),col)        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim([min(t_vals),max(t_vals)])
    ax.set_xlabel('time [s]',fontsize=fontsize)
    ax_false.set_ylabel('diameter [m]',fontsize=fontsize,labelpad=15)
    ax_false.set_frame_on(False)
    ax_false.tick_params(labelcolor='none',left=None,right=None,top=None,bottom=None)
    if print_fig:
        fig.savefig(fig_dir+'Dp_vs_t.png',dpi=500)
        print('fig 2 saved: ' + fig_dir + 'Dp_vs_t.png')

def fig2b(scenario_dir,fontsize=11,print_fig=False):
    # =============================================================================
    # diameter vs. time for quadrature approximation
    # =============================================================================
    
    fig_dir = get_fig_dir()
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(3.5,2.3)
    
    trajectories_filename = scenario_dir + '01_trajectories.pkl'
    trajectories_dict = pickle.load(open(trajectories_filename, 'rb'))
    
    default_case_options = {'smask_type':'none','vary_Tv_and_S':False}#,'acr_enhancement':1.}
    case_idx = utils.get_select_cases(trajectories_dict,default_case_options,return_idx=True)[0]
    gs = fig.add_gridspec(1,1)
    ax = fig.add_subplot(gs[:,:])
    Dd = trajectories_dict[case_idx]['inputs']['Dd']
    t_vals = np.logspace(-4,2,5000)
    hln = ['']*len(Dd)
    for qq in range(len(Dd)):
        ts = trajectories_dict[case_idx][qq]['evap']['mp'].index
        Dps = QuaRAD.get_diameter_from_mass(Dd[qq],trajectories_dict[case_idx][qq]['evap']['mp'].values,trajectories_dict[case_idx]['inputs']['rho_aero'])
        Dp_vs_t = QuaRAD.get_Dp_fun(ts,Dps)
        if trajectories_dict[case_idx]['inputs']['mode'][qq] == 'b':
            col = 'C0'
        elif trajectories_dict[case_idx]['inputs']['mode'][qq] == 'l':
            col = 'C1'            
        elif trajectories_dict[case_idx]['inputs']['mode'][qq] == 'o':
            col = 'C2'            
            
        hln[qq], = ax.plot(t_vals,Dp_vs_t(t_vals),col)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim([min(t_vals),max(t_vals)])
    plt.legend(
            [hln[0],hln[1],hln[-1]],['b-mode','l-mode','o-mode'],loc='upper left')
        
    default_case_options = {'smask_type':'none','vary_Tv_and_S':True}#,'acr_enhancement':1.}
    case_idx = utils.get_select_cases(trajectories_dict,default_case_options,return_idx=True)[0]
    Dd = trajectories_dict[case_idx]['inputs']['Dd']
    t_vals = np.logspace(-4,2,5000)
    hln = ['']*len(Dd)
    for qq in range(len(Dd)):
        ts = trajectories_dict[case_idx][qq]['evap']['mp'].index
        Dps = QuaRAD.get_diameter_from_mass(Dd[qq],trajectories_dict[case_idx][qq]['evap']['mp'].values,trajectories_dict[case_idx]['inputs']['rho_aero'])
        Dp_vs_t = QuaRAD.get_Dp_fun(ts,Dps)
        if trajectories_dict[case_idx]['inputs']['mode'][qq] == 'b':
            col = 'C0'
        elif trajectories_dict[case_idx]['inputs']['mode'][qq] == 'l':
            col = 'C1'            
        elif trajectories_dict[case_idx]['inputs']['mode'][qq] == 'o':
            col = 'C2'            
            
        hln[qq], = ax.plot(t_vals,Dp_vs_t(t_vals),col,linestyle=':',linewidth=3.)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim([min(t_vals),max(t_vals)])
    ax.set_xlabel('time [s]',fontsize=fontsize)
    ax.set_ylabel('diameter [m]',fontsize=fontsize)
    if print_fig:
        fig.savefig(fig_dir+'Dp_vs_t__b.png',dpi=500)
        print('fig 2 saved: ' + fig_dir + 'Dp_vs_t__b.png')
    
def fig3(scenario_dir,qq_vals='all',dispersion_type='both',colorscale='log',normalize=False,fontsize=11):
# =============================================================================
# average concenration associated with quadrature point q per emission rate of these particles
# units: m^{-3}/(#/s)
# =============================================================================

    fig_dir = get_fig_dir()
    normconcs_filename = scenario_dir + '02_normconcs.pkl'
    normconcs_dict = pickle.load(open(normconcs_filename, 'rb'))
    trajectories_filename = scenario_dir + '01_trajectories.pkl'
    trajectories_dict = pickle.load(open(trajectories_filename, 'rb'))

    dxs = normconcs_dict[0]['vs_xz']['dxs']
    dzs = normconcs_dict[0]['vs_xz']['dzs']
    if dispersion_type == 'both':
        Cq = normconcs_dict[0]['vs_xz']['Cq_per_emit']['jet'][:,0,:,:] + normconcs_dict[0]['vs_xz']['Cq_per_emit']['reflections'][0,:]
    elif dispersion_type == 'jet':
        Cq = normconcs_dict[0]['vs_xz']['Cq_per_emit']['jet'][:,0,:,:]
    elif dispersion_type == 'reflections':
        Cq = np.ones(dxs.shape)*normconcs_dict[0]['vs_xz']['Cq_per_emit']['reflections'][0,:]
    ws_cov = normconcs_dict[0]['inputs']['w_cov']
    if qq_vals == 'all':
        qq_vals = range(Cq.shape[2])
        
    N_part = len(qq_vals)
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(3.5,7.5)
    gs = fig.add_gridspec(N_part,1)
    ax_false = fig.add_subplot(gs[:,:])

    _max = 0.    
    _min = 1e12
    if normalize:
        for qq in qq_vals:
            _max = 10**np.floor(np.log10(max(np.append(Cq[:,:,qq].ravel(),_max))))
            _min = 10**np.ceil(np.log10(min(np.append(Cq[:,:,qq].ravel(),_max))))
    else:
        _max = 10**np.floor(np.log10(max(Cq.ravel())))
        _min = 10**np.ceil(np.log10(min(Cq.ravel())))   
    if colorscale != 'log':
        _max = _max
    
    _min = _max/1e4
    
    axs = []
    pcm = []
    ts = np.linspace(0.,60,1000)
    for qqq,qq in enumerate(np.argsort(normconcs_dict[0]['inputs']['Dd'])):#range(N_part):    
        if normconcs_dict[0]['inputs']['mode'][qqq] == 'b':
            col = 'C0'
        if normconcs_dict[0]['inputs']['mode'][qqq] == 'l':
            col = 'C1'
        if normconcs_dict[0]['inputs']['mode'][qqq] == 'o':
            col = 'C2'            
        ax_one = fig.add_subplot(gs[qq,0])
        if normalize:
            if colorscale == 'log':
                pcm_one = ax_one.pcolor(dxs,dzs,Cq[:,:,qqq].transpose()/ws_cov[qqq], norm=colors.LogNorm(vmin=_min, vmax=_max))
            else:
                pcm_one = ax_one.pcolor(dxs,dzs,Cq[:,:,qqq].transpose()/ws_cov[qqq], vmin=0.,vmax=_max)
        else:
            if colorscale == 'log':
                pcm_one = ax_one.pcolor(dxs,dzs,Cq[:,:,qqq].transpose(), norm=colors.LogNorm(vmin=_min, vmax=_max),zorder=1)
            else:
                pcm_one = ax_one.pcolor(dxs,dzs,Cq[:,:,qqq].transpose(), vmin=0.,vmax=_max)
        x0_vs_t, y0_vs_t, z0_vs_t, b_vs_t = QuaRAD.get_puff_funs(trajectories_dict[0][qqq]['pos'],trajectories_dict[0]['inputs'])
        plt.plot(x0_vs_t(ts)-trajectories_dict[0]['inputs']['L']/2,z0_vs_t(ts)-trajectories_dict[0]['inputs']['z0_source'],color='k',linewidth=1.,zorder=3,linestyle='--')
        ax_one.set_xticks(np.linspace(0.,4.,5))
        ax_one.set_xlim([min(dxs),max(dxs)])
        ax_one.set_ylim([min(dzs),max(dzs)])        
        h_txt = ax_one.text(
                ax_one.get_xlim()[1]+0.1, ax_one.get_ylim()[1], 
#                '$D_{0}$='+str(np.round(normconcs_dict[0]['inputs']['D'][qqq]*1e6,1)) + ' $\mu$m',                
                '$D_{0}$='+str(np.round(normconcs_dict[0]['inputs']['D'][qqq]*1e6,1)) + ' $\mu$m\n' + normconcs_dict[0]['inputs']['mode'][qqq] + '-mode',
                color=col,horizontalalignment='left',verticalalignment='top',fontsize=fontsize)
        
        if qq < len(qq_vals)-1:
            ax_one.set_xticklabels('')
        
        if qq == 0:
            cb = plt.colorbar(pcm_one,ax=ax_one,location='top',shrink=0.7);
            if normalize:
                cb.set_label('conc. per number emitted [m$^{-3}$ s]')
            else:
                cb.set_label('virion concentration associated \n with each quadrature point [m$^{-3}}$]',labelpad=8.)
        axs.append(ax_one)            
        pcm.append(pcm_one)
        
    ax_false.set_frame_on(False)
    ax_false.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    ax_false.set_xlabel('distance downwind of\ninfectious person [m]',labelpad=5.,fontsize=fontsize)
    ax_false.set_ylabel('vertical displacement from infectious person\'s mouth [m]',labelpad=4.,fontsize=fontsize)
    
    fig.savefig(fig_dir+'scenario__Cq_vs_xz.png',dpi=500)
    print('fig 3 saved: ' + fig_dir + 'scenario__Cq_vs_xz.png')
    
def fig4(ensemble_name='orig',emission_type='talk',excel_filename='../input/inputs.xlsx',N_cases=10000,D = np.logspace(-8.5,-4.5,25),fontsize=11,output_dir = '../output/',whis=[5,95]):
    # =============================================================================
    # median, quartiles, and 95% confidence interval in deposition efficiency vs. diameter
    # =============================================================================
    
    fig_dir = get_fig_dir()
    X,sample_varnames = utils.get_inputs__ensemble(
            ensemble_name, N_cases, fixed_vars={'S_source':1.}, 
            emission_type=emission_type, excel_filename=excel_filename)
    
    DE = np.zeros([N_cases,len(D)])
    for ii in range(N_cases):
        dep_coeff_a = X[ii,[varname == 'dep_coeff_a' for varname in sample_varnames]][0]
        dep_coeff_b = X[ii,[varname == 'dep_coeff_b' for varname in sample_varnames]][0]
        dep_coeff_c = X[ii,[varname == 'dep_coeff_c' for varname in sample_varnames]][0]    
        dep_coeff_d = X[ii,[varname == 'dep_coeff_d' for varname in sample_varnames]][0]
        DE[ii,:] = QuaRAD.get_nasal_deposition_efficiency(D,D,1000.,X[ii,[varname=='vol_inhale' for varname in sample_varnames]],310.,dep_coeffs=np.array([dep_coeff_a,dep_coeff_b,dep_coeff_c,dep_coeff_d]))

    fig = plt.figure();
    gs = fig.add_gridspec(1,1)
    gs_false = fig.add_gridspec(1,1)
    ax= fig.add_subplot(gs[:,:])
    
    fig.set_size_inches(3.5,2.5)
    
    ax.boxplot(DE,positions=np.log10(D),widths=0.8*(np.log10(D)[1]-np.log10(D)[0]),showfliers=False,whis=whis)
    plt.text(-6.95,0.95,'particles\nsmaller than\nSARS-CoV-2',ha='right',va='top')
    
    ax_false = fig.add_subplot(gs_false[:,:])

    ax.set_xticks(np.linspace(-8,-5,4))
    ax.set_xticklabels(np.round(1e6*np.logspace(-8,-5,4),4))
    ax.set_xlim(np.log10([min(D),10e-6]))


    ax.set_xlim([-8,-5])
    ax.set_frame_on(True)
    
    D_sars = 123e-9
    ax.set_ylim([0.,1.])    
    ax_false.set_xscale('log') 
    ax_false.set_xlim([10**ax.get_xlim()[0],10**ax.get_xlim()[1]])
    ax_false.set_frame_on(False)
    ax_false.tick_params(labelcolor='k', top=False, bottom=True, left=True, right=False)
    ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    
    rect = patches.Rectangle((ax.get_xlim()[0], ax.get_ylim()[0]), np.log10(D_sars)-ax.get_xlim()[0], ax.get_ylim()[1]-ax.get_ylim()[0],facecolor='k',edgecolor='none',alpha=0.2)
    ax.add_patch(rect)
    
    ax_false.set_ylim([0.,1.])    
    ax.set_xlabel('diameter [m]',fontsize=fontsize)
    ax.set_ylabel('deposition efficiency',fontsize=fontsize)
    fig.subplots_adjust(bottom=0.2,left=0.15)
    fig.savefig(fig_dir+'deposition_efficiency.png',dpi=500)
    print('fig 4 saved: ' + fig_dir + 'deposition_efficiency.png')
    
def fig5(ensemble_name='orig',emission_type='talk',excel_filename='../input/inputs.xlsx',N_cases=10000,D = np.logspace(-8.5,-4.5,25),fontsize=11,whis=[5,95]):
    # =============================================================================
    # median, quartiles, and 95% confidence interval in infection risk vs. dose
    # =============================================================================
    
    fig_dir =get_fig_dir()
    X_infection,all_varnames = utils.get_inputs__ensemble__infection(
            ensemble_name, N_cases, fixed_vars={'S_source':1.}, 
            emission_type=emission_type, excel_filename=excel_filename)
    
    F_trans = 1.
    F_c = 1.
    
    dose_vals= np.logspace(0,4,25)
    p_infect_vals = np.zeros([N_cases,len(dose_vals)])
    for ii,X_i_infection in enumerate(X_infection):
        normconcs_dict = dict()
        normconcs_dict[0] = {'inputs':{'emission_type':emission_type}}
        options = QuaRAD.set_default_options()        
        inputs__allcases = QuaRAD.get_inputs___infection__allcases(
                X_i_infection,all_varnames,normconcs_dict,options,excel_filename='../input/inputs.xlsx')    
        inputs_dict = inputs__allcases[0]
        
        muc_free = inputs_dict['muc_free']
        K_m = inputs_dict['K_m']
        F_v = 1 / (1 + K_m*muc_free)                
        p_pfu = inputs_dict['p_pfu']
        p_cell = inputs_dict['p_cell']
        
        p_infect_vals[ii,:] = 1 - (1 - (F_trans * F_v * p_pfu * F_c * p_cell))**(dose_vals)
        
    fig = plt.figure();
    gs = fig.add_gridspec(1,1)
    gs_false = fig.add_gridspec(1,1)
    ax= fig.add_subplot(gs[:,:])
    
    fig.set_size_inches(3.5,2.5)
    ax.boxplot(p_infect_vals,positions=np.log10(dose_vals),widths=0.8*(np.log10(dose_vals)[1]-np.log10(dose_vals)[0]),showfliers=False,whis=whis)
    ax_false = fig.add_subplot(gs_false[:,:])
    
    ax.set_xlim(np.log10([min(dose_vals),max(dose_vals)]))
    ax.set_xlabel('virion dose',fontsize=fontsize)
    ax.set_ylabel('probability of initial infection',fontsize=fontsize)
    
    ax_false.set_xscale('log') 
    ax_false.set_xlim([1,1e4])    
    ax_false.set_frame_on(False)
    ax_false.tick_params(labelcolor='k', top=False, bottom=True, left=True, right=False)
    
    ax.set_xscale('linear') 
    ax.set_xlim([0,4])    
    ax.set_ylim([0.,1.])
    ax.set_frame_on(True)
    
    ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    fig.subplots_adjust(bottom=0.2,left=0.15)
    fig.savefig(fig_dir+'ensemble__dose_reseponse.png',dpi=500)
    print('fig 5 saved: ' + fig_dir + 'ensemble__dose_reseponse.png')
    
def fig6(scenario_dir,qq_vals='all',dispersion_type='both',colorscale='log',normalize=False,fontsize=11):
# =============================================================================
# average concenration associated with quadrature point q per emission rate of these particles
# units: m^{-3}/(#/s)
# =============================================================================
    
    fig_dir = get_fig_dir()
    normconcs_filename = scenario_dir + '02_normconcs.pkl'
    normconcs_dict = pickle.load(open(normconcs_filename, 'rb'))
    dxs = normconcs_dict[0]['vs_xz']['dxs']
    dzs = normconcs_dict[0]['vs_xz']['dzs']
    if dispersion_type == 'both':
        C = normconcs_dict[0]['vs_xz']['C_per_emit']['jet'][:,0,:] + normconcs_dict[0]['vs_xz']['C_per_emit']['reflections'][0]
    elif dispersion_type == 'jet':
        C = normconcs_dict[0]['vs_xz']['C_per_emit']['jet'][:,0,:]
    elif dispersion_type == 'reflections':
        C = np.ones(dxs.shape)*normconcs_dict[0]['vs_xz']['C_per_emit']['reflections'][0]
    
    fig,ax_one = plt.subplots(1,constrained_layout=True)
    fig.set_size_inches(7.,3.)
    
    _min = 1.
    _max = 100.
    if colorscale == 'log':
        pcm_one = ax_one.pcolor(dxs,dzs,C.transpose(), norm=colors.LogNorm(vmin=_min, vmax=_max))
    else:
        pcm_one = ax_one.pcolor(dxs,dzs,C.transpose(), vmin=0.,vmax=_max)
    
    ax_one.yaxis.tick_right()
    ax_one.set_xlabel('distance downwind of infectious person [m]',labelpad=5.,fontsize=fontsize)
    ax_one.set_ylabel('displacement from mouth [m]',labelpad=22.,fontsize=fontsize,rotation=270)
    ax_one.yaxis.set_label_position("right")
    divider = make_axes_locatable(ax_one)
    cax = divider.append_axes("right", size="4%", pad="30%")
    cb = fig.colorbar(pcm_one,cax=cax,shrink=1.);
    cb.set_label('virion concentration [m$^{-3}}$]',rotation=270,fontsize=fontsize,labelpad=18.)
    
    ax_one.set_yticks(np.linspace(-1.5,0.5,5))
    fig.subplots_adjust(bottom=0.2,right=0.9)
    fig.savefig(fig_dir+'scenario__C_vs_xz.png',dpi=500)
    print('fig 6 saved: ' + fig_dir + 'scenario__C_vs_xz.png')
    
def fig7(ensemble_dir,varname='p_infect',dxval='all',fontsize=11,whis=[5,95],thresh_vals = [1.1]):
    fig_dir = get_fig_dir()
    default_case_options = {'acr_enhancement':1.}
    dxs, vardat_default = utils.get_ensemble_dat(
            varname,ensemble_dir,default_case_options,dispersion_type='both',dxval=dxval)

    dxs, vardat_jet = utils.get_ensemble_dat(
            varname,ensemble_dir,default_case_options,dispersion_type='jet',dxval=dxval)
    
    dxs, vardat_reflections = utils.get_ensemble_dat(
            varname,ensemble_dir,default_case_options,dispersion_type='reflections',dxval=dxval)
    
    fig = plt.figure()
    fig.set_size_inches(4.,6.5)
    gs = fig.add_gridspec(14,1)
    gs_false = fig.add_gridspec(14,1)    
    ax_false = []
    ax_false.append(fig.add_subplot(gs_false[:5,:]))
    ax_false.append(fig.add_subplot(gs_false[5:10,:]))
    
    ax = []
    ax.append(fig.add_subplot(gs[:5,:]))
    ax.append(fig.add_subplot(gs[5:10,:]))
    ax.append(fig.add_subplot(gs[12:,:]))
    idx1, = np.where(np.sum(vardat_reflections==0.,axis=1)==0)
    vardat_default = vardat_default[idx1,:]
    vardat_jet = vardat_jet[idx1,:]    
    vardat_reflections = vardat_reflections[idx1,:]    
    
    yticks1 = np.linspace(-3,0,4)
    ax_false[0].set_frame_on(False)
    ax_false[0].set_yticks(10**yticks1);
    ax_false[0].set_yscale('log')
    
    
    ax[0].boxplot(np.log10(vardat_default),positions=dxs,widths=(dxs[1]-dxs[0])*0.8,showfliers=False,whis=whis);    
    ax[0].set_yticks([]);
    ax[0].set_xticks(np.linspace(0.,4.,5))
    ax[0].set_xticklabels(np.round(np.linspace(0.,4.,5),0))
    ax[0].set_xlim([0.,4.])
    ax_false[0].set_ylim([10**ax[0].get_ylim()[0],10**ax[0].get_ylim()[1]])
    ax_false[0].set_xticks([])
    ax_false[0].tick_params(top=False, bottom=False, left=True, right=False)
    
    yticks2 =np.linspace(0,4,5)
    ax_false[1].set_frame_on(False)
    ax_false[1].set_yticks(10**yticks2);
    ax_false[1].set_yscale('log')
    
    ax[1].boxplot(np.log10(vardat_default/vardat_reflections),positions=dxs,widths=(dxs[1]-dxs[0])*0.8,showfliers=False,whis=whis);
    
    ax[1].set_xticks(np.linspace(0.,4.,5))
    ax[1].set_yticks([]); 
    ax[1].set_xticks(np.linspace(0.,4.,5))
    ax[1].set_xticklabels(np.round(np.linspace(0.,4.,5),0))
    ax[1].set_xlim([0.,4.])
    ax[1].set_xlim(ax[0].get_xlim())
    ax[1].set_xticklabels(np.round(np.linspace(0.,4.,5),0)) 
    ax[1].set_yticks(yticks2)
    ax[1].set_yticklabels([int(tick) for tick in 10**yticks2])
    ax[1].set_ylim(np.log10([1.,1000]))
    ax[1].set_yticks([]);     
    ax_false[1].set_ylim([10**ax[1].get_ylim()[0],10**ax[1].get_ylim()[1]])
    ax_false[1].set_xticks([])
    ax_false[1].tick_params(top=False, bottom=False, left=True, right=False)
        
    
    x_near = np.zeros([len(thresh_vals),vardat_default.shape[0]])
    for jj,thresh in enumerate(thresh_vals):
        for ii in range(vardat_default.shape[0]):
            x_near[jj,ii] = interp1d((vardat_default/vardat_reflections)[ii,:],dxs,fill_value='extrapolate')(thresh)
            
    bplot = ax[2].boxplot(x_near.transpose(),positions=range(len(thresh_vals)),vert=False,widths=0.4,whis=whis,showfliers=False)
    ax[2].set_xlim(ax[1].get_xlim())
    ax[2].set_xticks(ax[0].get_xticks())
    ax[2].set_xticklabels(ax[0].get_xticklabels())
    ax[0].set_xticklabels('')
    ax[2].set_yticks([])
    ax[1].set_xlabel('distance downwind of\ninfectious person [m]', fontsize=fontsize)
    
    ax_false[0].set_ylabel('risk of initial infection', fontsize=fontsize)
    ax_false[1].set_ylabel('risk relative to risk\nin well-mixed room',fontsize=fontsize)
    ax[2].set_xlabel('$x_{\mathrm{near}}$ [m]',fontsize=fontsize)
    ax[0].tick_params(top=False, bottom=True, left=False, right=False)

    
#      horizontal extent enhanced\nrisk [m]')
    fig.subplots_adjust(left=0.25,right=0.85,hspace=1.)
    fig.tight_layout()
    
    shrinkx = 0.92
    shrinky = 0.9
    ax[0].annotate('A', xy=get_axlab_locs(ax[0], shrinkx=shrinkx,shrinky=shrinky),fontsize=fontsize)
    ax[1].annotate('B', xy=get_axlab_locs(ax[1], shrinkx=shrinkx,shrinky=shrinky),fontsize=fontsize)
    ax[2].annotate('C', xy=get_axlab_locs(ax[2], shrinkx=shrinkx,shrinky=0.7),fontsize=fontsize)
    
    fig.savefig(fig_dir+'short_range.png',dpi=500)
    print('fig 7 saved: ' + fig_dir + 'short_range.png')    
    return fig, ax, bplot

def fig_another1(scenario_dir,fontsize=11):
    fig_dir = get_fig_dir()
    normconcs_filename = scenario_dir + '02_normconcs.pkl'
    normconcs_dict = pickle.load(open(normconcs_filename, 'rb'))
    
    dxs = normconcs_dict[0]['vs_xz']['dxs']
    dzs = normconcs_dict[0]['vs_xz']['dzs']
    idx, = np.where(dzs==0)
    varname = 'Cq_per_emit'
    var_q = normconcs_dict[0]['vs_xz'][varname]['jet'][:,0,idx[0],:] + normconcs_dict[0]['vs_xz']['Cq_per_emit']['reflections'][0,:]
    ws_cov = normconcs_dict[0]['inputs']['w_cov']
    modes = normconcs_dict[0]['inputs']['mode']
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(3.5,2.2)
    N_plots = 1
    gs = fig.add_gridspec(N_plots,1)
    axs = []
    for ii in range(N_plots):
        axs.append(fig.add_subplot(gs[ii,:])) 
    
    all3_cols = ['C0','C1','C2']
    all3_modes = ['b','l','o']
    cols = []
    hln = []
    for qq,(w_cov,mode) in enumerate(zip(ws_cov,modes)):
        idx, = np.where([one_mode == mode for one_mode in all3_modes])
        col = all3_cols[idx[0]]
        cols.append(col)
        hln_one, = axs[0].plot(dxs,var_q[:,qq],color=col)        
        hln.append(hln_one)
    axs[0].set_yscale('log')
    axs[0].set_ylim([5e-3,500.])
    axs[0].set_xlabel('distance downwind [m]',fontsize=fontsize)
    axs[0].set_ylabel('$N_{v,i}$ [m$^3$]',fontsize=fontsize)
    axs[0].set_xlim([0.,4.])
    axs[0].set_xticks(np.linspace(0,4,5))
    axs[0].set_xticklabels(np.linspace(0.,4.,5))    
    axs[0].legend([hln[0],hln[1],hln[5]],['b-mode','l-mode','o-mode'])
    fig.savefig(fig_dir+'mode_contribution.png',dpi=500)
    
def SIfig1(scenario_dir,varname='p_infect',fontsize=11,dispersion_type='both'):
    fig_dir = get_fig_dir()    
    default_case_options = {'N_quad':[1,3,2]}
    dxs, vardat_default = utils.get_scenario_dat(
            varname,scenario_dir,default_case_options,dispersion_type=dispersion_type,dxval='all')
    
    case_options = {'N_quad':[200,200,200]}
    dxs, vardat_toomany = utils.get_scenario_dat(
            varname,scenario_dir,case_options,dispersion_type=dispersion_type,dxval='all')
    
    case_options = {'N_quad':[1,1,1]}
    dxs, vardat_toofew = utils.get_scenario_dat(
            varname,scenario_dir,case_options,dispersion_type=dispersion_type,dxval='all')
    
    fig = plt.figure();
    gs = fig.add_gridspec(1,1)
    ax= fig.add_subplot(gs[:,:])
    fig.set_size_inches(3.8,3.)
    
    hln = ax.plot(dxs,np.vstack([vardat_toomany,vardat_toofew,vardat_default]).transpose(),linewidth=2.)
    hln[-1].set_color('k')
    hln[-1].set_linestyle(':')
    hln[0].set_color('C8')
    hln[1].set_color('C9')    
    ax.set_yscale('log')
    
    ax.set_xlim([min(dxs),max(dxs)])
    
    print('too many:',np.mean((vardat_default)/vardat_toomany),', too few:',np.mean((vardat_toofew/vardat_toomany)))
    ax.set_xlabel('distanace downwind of infectious person [m]',fontsize=fontsize)
    if varname == 'p_infect':
        ax.set_ylabel('predicted risk',fontsize=fontsize)
        
    plt.legend([hln[2],hln[0],hln[1]],['6 points','600 points','3 points'],fontsize=10.)
    fig.subplots_adjust(bottom=0.2,left=0.2,right=0.85)
    fig.savefig(fig_dir+'quad_optimization.png',dpi=500)

def SIfig2(scenario_dir,fontsize=11,dispersion_type='jet',normalize=False,colorscale='log',qq_vals=[0,2]):
    # compares well with monte carlo model
    fig_dir = get_fig_dir()
    normconcs_filename = scenario_dir + '02_normconcs.pkl'
    normconcs_dict = pickle.load(open(normconcs_filename, 'rb'))
    dxs = normconcs_dict[0]['vs_xz']['dxs']
    dzs = normconcs_dict[0]['vs_xz']['dzs']
    if dispersion_type == 'both':
        Cq = normconcs_dict[0]['vs_xz']['Cq_per_emit']['jet'][:,0,:,:] + normconcs_dict[0]['vs_xz']['Cq_per_emit']['reflections'][0,:]
    elif dispersion_type == 'jet':
        Cq = normconcs_dict[0]['vs_xz']['Cq_per_emit']['jet'][:,0,:,:]
    elif dispersion_type == 'reflections':
        Cq = np.ones(dxs.shape)*normconcs_dict[0]['vs_xz']['Cq_per_emit']['reflections'][0,:]
    ws_cov = normconcs_dict[0]['inputs']['w_cov']
    ws = normconcs_dict[0]['inputs']['w']
    
    N_part = len(qq_vals)
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(6.,4.)
    gs = fig.add_gridspec(N_part,2)

    _max = 0.    
    _min = 1e12
    if normalize:
        for qq in qq_vals:
            _max = 10**np.floor(np.log10(max(np.append(Cq[:,:,qq].ravel(),_max))))
            _min = 10**np.ceil(np.log10(min(np.append(Cq[:,:,qq].ravel(),_max))))
    else:
        _max = 10**np.floor(np.log10(max(Cq.ravel())))
        _min = 10**np.ceil(np.log10(min(Cq.ravel())))   
    if colorscale != 'log':
        _max = _max
    
    _min = _max/1e3
    
    axs = []
    pcm = []
    axs2 = []
    
    for qqq,qq in enumerate(qq_vals):#range(N_part):    
        ax_one = fig.add_subplot(gs[qqq,0])
        if normalize:
            if colorscale == 'log':
                pcm_one = ax_one.pcolor(dxs,dzs,Cq[:,:,qq].transpose()/ws_cov[qq], norm=colors.LogNorm(vmin=_min, vmax=_max))
            else:
                pcm_one = ax_one.pcolor(dxs,dzs,Cq[:,:,qq].transpose()/ws_cov[qq], vmin=0.,vmax=_max)
        else:
            if colorscale == 'log':
                pcm_one = ax_one.pcolor(dxs,dzs,ws[qq]*Cq[:,:,qq].transpose()/ws_cov[qq], norm=colors.LogNorm(vmin=_min, vmax=_max))
            else:
                pcm_one = ax_one.pcolor(dxs,dzs,ws[qq]*Cq[:,:,qq].transpose()/ws_cov[qq], vmin=0.,vmax=_max)
        

        ax_one.set_xticks(np.linspace(0.,4.,5))
        ax_one.text(ax_one.get_xlim()[1]-0.05, ax_one.get_ylim()[0], '$D_{p,0}$='+str(np.round(normconcs_dict[0]['inputs']['D'][qq]*1e6,1)) + ' $\mu$m', 
                    color='white',horizontalalignment='right',verticalalignment='bottom',fontsize=fontsize)
        
        if qq < len(qq_vals)-1:
            ax_one.set_xticklabels('')
        
        axs.append(ax_one) 
        pcm.append(pcm_one)

        ax_two = fig.add_subplot(gs[qqq,1])
        ax_two.set_xticks(np.linspace(0.,4.,5))
        ax_two.text(ax_one.get_xlim()[1]-0.05, ax_two.get_ylim()[0], '$D_{p,0}$='+str(np.round(normconcs_dict[0]['inputs']['D'][qq]*1e6,1)) + ' $\mu$m', 
                    color='black',horizontalalignment='right',verticalalignment='bottom',fontsize=fontsize)
        ax_two.set_yticklabels('')
        if qq < len(qq_vals)-1:
            ax_two.set_xticklabels('')
        axs2.append(ax_two)    

    cb = plt.colorbar(pcm[0],ax=axs[0],location='top',shrink=0.7);
    if normalize:
        cb.set_label('conc. per number emitted [m$^{-3}$ s]')
    else:
        cb.set_label('particle concentration [m$^{-3}}$]',labelpad=8.)
    
    ax_false = fig.add_subplot(gs[:,:])    
    ax_false.set_frame_on(False)
    ax_false.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    ax_false.set_xlabel('distance downwind of infectious person [m]',labelpad=5.,fontsize=fontsize)
    ax_false.set_ylabel('vertical displacement from\ninfectious person\'s mouth [m]',labelpad=4.,fontsize=fontsize)

    fig.savefig(fig_dir+'scenario__Cq_compare.png',dpi=500)
    
def get_fig_dir():
    fig_dir = '../03_figs/'
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)
    return fig_dir



def get_axlab_locs(ax, shrinkx=0.9,shrinky=0.9):
    xscale = ax.get_xscale()
    yscale = ax.get_yscale()    
    if xscale == 'log':
        x = 10**(np.log10(ax.get_xlim()[0]) + (np.log10(ax.get_xlim()[1]) - np.log10(ax.get_xlim()[0]))*shrinkx)
    else:
        x = ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0])*shrinkx

    if yscale == 'log':
        y = 10**(np.log10(ax.get_ylim()[0]) + (np.log10(ax.get_ylim()[1]) - np.log10(ax.get_ylim()[0]))*shrinky)
    else:
        y = ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0])*shrinky
    return x,y


