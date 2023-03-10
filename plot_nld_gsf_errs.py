#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 17:15:38 2021

@author: francesco, updated December 28th 2022

Draw nld and gsf from the nld_whole_FG.txt and gsf_whole_FG.txt files produced from make_nlds_gsfs_lists.py
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from readlib import readstrength, readldmodel, search_string_in_file
from systlib import import_ocl, D2rho, chisquared, drho

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'font.serif': ['computer Modern'],
    'text.usetex': True,
    'pgf.rcfonts': False,
})


#constants. Don't play with these
hc = 197.3269804 #MeV*fm
const = 1/(3*np.pi**2*hc**2*10) #mb

NLD_pathstrings = ['FG']
talys_path = '/home/francesco/talys/'

#Don't modify unless you know what you're doing
Z = 67
A = 167
L1min = 4
L1max = 11
target_spin = 7.0
Sn = 7.282
a0 = -0.7560
a1 = 0.1280
spin_cutoff_low = 5.675 #G&C
spin_cutoff_high = 7.100 #RMI
cutoff_unc = 0.00

#data from Mughabghab
D0 = 2.32
D0_err = 0.232
Gg_mean = 88.5
Gg_sigma = 0.885


for NLD_pathstring in NLD_pathstrings:
    rho_Sn_err_up = drho(target_spin, spin_cutoff_high, spin_cutoff_high*cutoff_unc, D0, D0_err) #assume 10% error in spin_cutoff param
    rho_Sn_err_down = drho(target_spin, spin_cutoff_low, spin_cutoff_low*cutoff_unc, D0, D0_err) #assume 10% error in spin_cutoff param
    rho_lowerlim = D2rho(D0, target_spin, spin_cutoff_low)
    rho_upperlim = D2rho(D0, target_spin, spin_cutoff_high)
    rho_mean = (rho_lowerlim - rho_Sn_err_down + rho_upperlim + rho_Sn_err_up)/2
    rho_sigma = rho_upperlim + rho_Sn_err_up - rho_mean
    database_path = 'Make_dataset/167Ho-database_' + NLD_pathstring + '/'
    
    #load best fits
    best_fits = np.load('data/generated/best_fits_' + NLD_pathstring + '.npy', allow_pickle = True)
    best_gsf = best_fits[1]
    best_gsf.clean_nans()
    best_nld = best_fits[0]
    best_nld.clean_nans()
    extr_path = best_nld.path[:-10] + 'fermigas.cnt'
    extr_mat = import_ocl(extr_path,a0,a1, fermi = True)
    extr_vals = []
    nld_vals = []
    nld_errvals = []
    for idx, E in enumerate(best_nld.energies):
        if E > 1.5:
            idx2 = np.argwhere(extr_mat[:,0] == E)[0,0]
            extr_vals.append(extr_mat[idx2,1])
            nld_vals.append(best_nld.y[idx])
            nld_errvals.append(best_nld.yerr[idx])
            
    chisq = chisquared(extr_vals, nld_vals, nld_errvals, DoF=1, method = 'linear',reduced=True)
    print(chisq)
    
    #load 167Ho GognyM1
    pathM1 = talys_path + 'structure/gamma/gognyM1/Ho.psf'
    GognyM1 = np.loadtxt(pathM1, skiprows = search_string_in_file(pathM1, f'A= {A}') + 2, max_rows = 300)
    
    #import experimental nld and gsf
    nld_mat = np.genfromtxt('data/generated/nld_' + NLD_pathstring + '_whole.txt', unpack = True).T
    gsf_mat = np.genfromtxt('data/generated/gsf_' + NLD_pathstring + '_whole.txt', unpack = True).T
    #delete rows with nans
    nld_mat = nld_mat[~np.isnan(nld_mat).any(axis=1)]
    gsf_mat = gsf_mat[~np.isnan(gsf_mat).any(axis=1)]
    
    #delete some points?
    #delete last rows in gsf
    best_gsf.delete_point(-1)
    gsf_mat = np.delete(gsf_mat,[-1],0)
    
    #import known levels
    known_levs = import_ocl('data/rholev.cnt',a0,a1,fermi=True)
    
    #import TALYS calculated GSFs
    TALYS_strengths = [readstrength(Z, A, 1, 1, strength, 1) for strength in range(1,9)]
    
    #import TALYS calculated NLDs
    TALYS_ldmodels = [readldmodel(Z, A, ld, 1, 1, 1) for ld in range(1,7)]
    
    #start plotting
    cmap = matplotlib.cm.get_cmap('YlGnBu')
    fig0,ax0 = plt.subplots(figsize = (5.0, 3.75), dpi = 300)
    fig1,ax1 = plt.subplots(figsize = (5.0, 3.75), dpi = 300)
    ax0.plot(np.zeros(1), np.zeros([1,5]), color='w', alpha=0, label=' ')
    singleaxs = [ax0,ax1]
    chi2_lim = [6,9]
    chi2_lim_energies = [known_levs[int(chi2_lim[0]),0], known_levs[int(chi2_lim[1]),0]]
    ax0.axvspan(chi2_lim_energies[0], chi2_lim_energies[1], alpha=0.2, color='red',label='Fitting intv.')
    ax0.plot(known_levs[:,0],known_levs[:,1],'k-',label='Known lvs.')
    ax0.errorbar(Sn, rho_mean,yerr=rho_sigma,ecolor='g',linestyle=None, elinewidth = 4, capsize = 5, label=r'$\rho$ at Sn')
    
    #Plot experiment data
    for ax, val_matrix in zip(singleaxs, [nld_mat, gsf_mat]):
        ax.fill_between(val_matrix[:,0], val_matrix[:,2], val_matrix[:,-2], color = 'c', alpha = 0.2, label=r'2$\sigma$ conf.')
        ax.fill_between(val_matrix[:,0], val_matrix[:,3], val_matrix[:,-3], color = 'b', alpha = 0.2, label=r'1$\sigma$ conf.')
        ax.errorbar(val_matrix[:,0], val_matrix[:,1],yerr=val_matrix[:,-1], fmt = '.', color = 'b', ecolor='b', label='This work')
        ax.set_yscale('log')
    ax0.set_xlabel(r'$E_x$ [MeV]')
    ax1.set_xlabel(r'$E_\gamma$ [MeV]')
    
    #plot TALYS strengths
    stls = ['-','--','-.',':','-','--','-.',':']
    for i, TALYS_strength in enumerate(TALYS_strengths):
        if i<4:
            col = 3
        else:
            col = 8
        ax1.plot(TALYS_strength[:,0],TALYS_strength[:,1] + TALYS_strength[:,2], color = cmap(col/8), linestyle = stls[i], alpha = 0.8, label = 'strength %d'%(i+1))
    
    #plot TALYS nld
    for i, TALYS_ldmodel in enumerate(TALYS_ldmodels):
        if i<3:
            col = 3
        else:
            col = 5
        ax0.plot(TALYS_ldmodel[:,0],TALYS_ldmodel[:,1], color = cmap(col/6), linestyle = stls[i], alpha = 0.8, label = 'ldmodel %d'%(i+1))
    ax0.plot(extr_mat[9:,0], extr_mat[9:,1], color = 'k', linestyle = '--', alpha = 1, label = NLD_pathstring + ' extrap.')
    ranges = [[0,6.5], [np.log10(5e-9),np.log10(1e-6)]]
    #plot
    ax0.set_ylabel(r'NLD [MeV$^{-1}$]')
    ax1.set_ylabel(r'GSF [MeV$^{-3}$]')
    ax0.set_xlim(-0.5,7.5)
    ax0.set_ylim(1e0,1e7)    
    ax1.set_xlim(0.5,7.5)
    ax1.set_ylim(5e-9,1e-6)
    ax0.legend(loc = 'lower right', ncol = 2, frameon = False)
    ax1.legend(loc = 'upper left', ncol = 2, frameon = False)
    fig0.tight_layout()
    fig1.tight_layout()
    fig0.show()
    fig1.show()
    fig0.savefig('pictures/nld_'+NLD_pathstring+'.png', format = 'png')
    fig1.savefig('pictures/gsf_'+NLD_pathstring+'.png', format = 'png')
    fig0.savefig('pictures/nld_'+NLD_pathstring+'.pdf', format = 'pdf')
    fig1.savefig('pictures/gsf_'+NLD_pathstring+'.pdf', format = 'pdf')
    