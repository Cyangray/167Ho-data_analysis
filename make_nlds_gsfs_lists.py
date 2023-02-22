#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Thu Oct 14 09:49:41 2021

@author: francesco, updated 31 January 2022

To run this code, you have to have run run_cnt_nrm.py in Make_dataset/

Script looping thorugh all the produced rhopaw.cnt, strength.nrm (and astrorate.g
if astro = True), and:
1) create numpy arrays of nld and gsf (and astrorate) objects, storing all information
    produced by counting.c, normalization.c (and TALYS). These will be saved as
    nlds.npy, gsfs.npy and astrorates.npy
2) To each nld, gsf or astrorate is associated a chi2 score, telling how well
    the nld fits to the known levels in the two fitting intervals chi_lim and chi_lim2.
3) For each energy bin of nld, gsf and MACS, the uncertainty is found checking
    graphically where the chi2 parabola crosses the chi2+1 line. The results are
    saved to text files.
4) Save the nld-gsf couple with the least chi2 to best_fits.npy
5) If plot_chis = True, it plots the parabolas for one specific energy bin of 
    the nld and the gsf, + astrorate/MACS if astro = True
'''

import numpy as np
import matplotlib.pyplot as plt
from systlib import import_ocl, nld, gsf, import_Bnorm, astrorate, import_Anorm_alpha, chisquared, flat_distr_chi2_fade, calc_errors_chis, calc_errors_chis_MACS, D2rho, drho

#paths
NLD_pathstring = 'FG'
database_path = '/home/francesco/Documents/164Dy-experiment/Python_normalization/Make_dataset/167Ho-database_' + NLD_pathstring + '/'

#constants. Don't play with these
hc = 197.3269804 #MeV*fm
const = 1/(3*np.pi**2*hc**2*10) #mb

#Don't modify unless you know what you're doing
L1min = 4
L1max = 11
target_spin = 7.0
Sn = 7.282
a0 = -0.7560
a1 = 0.1280
spin_cutoff_low = 5.675 #G&C
spin_cutoff_high = 7.100 #RMI
cutoff_unc = 0

#data from Mughabghab
D0 = 2.32
D0_err = 0.232
Gg_mean = 88.5
Gg_sigma = 8.85

spin_cutoff_array_low = np.ones(10)*spin_cutoff_low
spin_cutoff_array_high = np.ones(10)*spin_cutoff_high
spin_cutoff_array_middle = np.linspace(spin_cutoff_low, spin_cutoff_high, 31)
spin_cutoff_array = np.concatenate((spin_cutoff_array_low, spin_cutoff_array_middle[1:-1], spin_cutoff_array_high))
rho_array_middle = D2rho(D0, target_spin, spin_cutoff_array_middle)
rho_min = np.amin(rho_array_middle)
rho_max = np.amax(rho_array_middle)
rho_Sn_err_down = drho(target_spin, spin_cutoff_low, spin_cutoff_low*cutoff_unc, D0, D0_err, rho=rho_min)
rho_Sn_err_up = drho(target_spin, spin_cutoff_high, spin_cutoff_high*cutoff_unc, D0, D0_err, rho=rho_max)
rho_array_low = np.linspace(rho_min - 3*rho_Sn_err_down, rho_min, 10)
rho_array_high = np.linspace(rho_max, rho_max + 3*rho_Sn_err_up, 10)
rho_array = np.concatenate((rho_array_low, rho_array_middle[1:-1], rho_array_high))
rho_mean = (rho_min + rho_max)/2
rho_sigma = rho_max - rho_mean
blist_limits = [(rho_min - 3*rho_Sn_err_down)/rho_mean, (rho_max + 3*rho_Sn_err_up)/rho_mean]
nrhos = len(rho_array)
blist = np.linspace(blist_limits[0],blist_limits[1],nrhos)
Gglist_limits = [int(Gg_mean-2.5*Gg_sigma)-0.5, int(Gg_mean+2.5*Gg_sigma)+0.5]
Gglist = np.linspace(Gglist_limits[0],Gglist_limits[1],int((Gglist_limits[1] - Gglist_limits[0])*2+1))#51
rho_flat_distr = True

#Play with these parameters
chi2_lim = [6,9]  #Fitting interval. Limits are included.
method = 'linear'

#some switches
load_lists = True
plot_chis = True
astro = True
energy_bin = 30
temp_bin = 7

'''
Initializing main loop. 
loop idea: loop through all L1 and L2 (the lower region to fit to in counting.c),
rho and Gg. Each such parameter combination points to a specific rhopaw.cnt 
produced in run_counting.py. Imports the NLD and the GSF, calculate the chi 
squared test to the region in chi2_lim, save all the parameters in the nld and 
gsf objects, and the objects in lists.
'''

Ho167_nld_lvl = import_ocl('data/rholev.cnt',a0,a1, fermi=True)
chi2_lim_e = [Ho167_nld_lvl[int(chi2_lim[0]),0], Ho167_nld_lvl[int(chi2_lim[1]),0]]
omps = ['localompy', 'jlmompy']
if load_lists:
    nlds = np.load('data/generated/nlds_' + NLD_pathstring + '.npy', allow_pickle = True)
    gsfs = np.load('data/generated/gsfs_' + NLD_pathstring + '.npy', allow_pickle = True)
    if astro:
        ncrates = np.load('data/generated/ncrates_' + NLD_pathstring + '.npy', allow_pickle = True)
        ncrates_localompy = np.load('data/generated/ncrates_localompy_' + NLD_pathstring + '.npy', allow_pickle = True)
        ncrates_jlmompy = np.load('data/generated/ncrates_jlmompy_' + NLD_pathstring + '.npy', allow_pickle = True)
        ncrates_omps = [ncrates_localompy, ncrates_jlmompy]
else:
    #beginning the big nested loop
    gsfs = []
    nlds = []
    if astro:
        ncrates = []
        ncrates_localompy = []
        ncrates_jlmompy = []
        ncrates_omps = [ncrates_localompy, ncrates_jlmompy]
    for indexb, b in enumerate(blist):
        bstr_int = "{:.2f}".format(b)
        bstr = bstr_int.translate(bstr_int.maketrans('','', '.'))
        new_spincutoff = spin_cutoff_array[indexb]
        new_rho = rho_array[indexb]
        new_drho = drho(target_spin, new_spincutoff, new_spincutoff*cutoff_unc, D0, D0_err, rho = new_rho)
        new_D = D0
        new_rho_str = '{:.6f}'.format(new_rho)
        new_D_str = '{:.6f}'.format(new_D)
        new_dir_rho = bstr + '-' + str(int(new_rho))
        print('Bstr: %s'%indexb) #print something to show the progression
        
        for L1n in range(L1min,L1max):
            L1 = str(L1n)
            if L1n == 1:
                L2_skip = 2
            else:
                L2_skip = 1
            for L2n in range(L1n + L2_skip, L1max):
                L1 = str(L1n)
                L2 = str(L2n)
                new_dir_L1_L2 = 'L1-'+L1+'_L2-'+L2
                curr_nld = nld(database_path + new_dir_rho + '/' + new_dir_L1_L2 + '/rhopaw.cnt',a0 = a0, a1 = a1, is_ocl = True)
                Anorm, alpha = import_Anorm_alpha(database_path +  new_dir_rho + '/' + new_dir_L1_L2 + '/alpha.txt')
                
                #calculate the reduced chi2
                lvl_values = Ho167_nld_lvl[chi2_lim[0]:(chi2_lim[1]+1),1]
                ocl_values = curr_nld.y[chi2_lim[0]:(chi2_lim[1]+1)]
                ocl_errs = curr_nld.yerr[chi2_lim[0]:(chi2_lim[1]+1)]
                chi2 = chisquared(lvl_values, ocl_values, ocl_errs, DoF = 1, method = method, reduced=False)
                
                #store values in objects
                curr_nld.L1 = L1
                curr_nld.L2 = L2
                curr_nld.Anorm = Anorm
                curr_nld.alpha = alpha
                curr_nld.rho = new_rho
                curr_nld.drho = new_drho
                curr_nld.b = b
                curr_nld.spin_cutoff = new_spincutoff
                curr_nld.D0 = new_D
                if rho_flat_distr:
                    curr_nld.chi2 = chi2 + flat_distr_chi2_fade(rho_max, rho_min, [rho_Sn_err_down,rho_Sn_err_up], new_rho)
                else:
                    curr_nld.chi2 = chi2 + ((rho_mean - new_rho)/rho_sigma)**2
                nlds.append(curr_nld)
                
                for Gg in Gglist:
                    Ggstr = str(int(Gg*10))
                    curr_gsf = gsf(database_path + new_dir_rho + '/' + new_dir_L1_L2 + '/' + Ggstr + '/strength.nrm', a0 = a0, a1 = a1, is_sigma = False, is_ocl = True)#, channels=78)
                    Bnorm = import_Bnorm(database_path + new_dir_rho + '/' + new_dir_L1_L2 + '/' + Ggstr + '/input.nrm')
                    
                    if astro:
                        found_astro = False
                        try:
                            curr_ncrates = []
                            for omp in omps:
                                curr_ncrates.append(astrorate(database_path + new_dir_rho + '/' + new_dir_L1_L2 + '/' + Ggstr + '/' + omp + '/astrorate.g'))
                            objs = [curr_gsf] + curr_ncrates
                            found_astro = True
                        except:
                            objs = [curr_gsf]
                    else:
                        objs = [curr_gsf]
    
                    #store values in objects
                    for el in objs:
                        el.L1 = L1
                        el.L2 = L2
                        el.Anorm = Anorm
                        el.Bnorm = Bnorm
                        el.alpha = alpha
                        el.Gg = Gg
                        el.rho = new_rho
                        el.drho = new_drho
                        el.b = b
                        el.chi2 = curr_nld.chi2 + ((Gg_mean - el.Gg)/Gg_sigma)**2
                        el.spin_cutoff = new_spincutoff
                        el.D0 = new_D
                    gsfs.append(curr_gsf)
                    if astro and found_astro:
                        ncrates_localompy.append(curr_ncrates[0])
                        ncrates_jlmompy.append(curr_ncrates[1])
                        #ncrates = ncrates + curr_ncrates
                        
    # save lists of nlds and gsfs to file
    np.save('data/generated/nlds_' + NLD_pathstring + '.npy', nlds)
    np.save('data/generated/gsfs_' + NLD_pathstring + '.npy', gsfs)
    if astro:
        np.save('data/generated/ncrates_' + NLD_pathstring + '.npy', ncrates)
        for omp, ncrates in zip(omps, ncrates_omps):
            np.save('data/generated/ncrates_' + omp + '_' + NLD_pathstring + '.npy', ncrates)

#Save in best_fits.npy the nld-gsf couple with the least chi2 score
nldchis = []
gsfchis = []
nldvals = []
gsfvals = []

for el in nlds:
        nldchis.append(el.chi2)
        nldvals.append(el.y[energy_bin])
for el in gsfs:
        gsfchis.append(el.chi2)
        gsfvals.append(el.y[energy_bin])

nldchi_argmin = np.argmin(nldchis)
gsfchi_argmin = np.argmin(gsfchis)
nldchimin = nldchis[nldchi_argmin]
gsfchimin = gsfchis[gsfchi_argmin]
least_nld_gsf = [nlds[nldchi_argmin], gsfs[gsfchi_argmin]]

if astro:
    ncrateschis = [[],[]]
    ncratesvals = [[],[]]
    ncrateschi_argmin = [0,0]
    ncrateschimin = [0,0]
    for i, ncrates in enumerate(ncrates_omps):
        for el in ncrates:
            ncrateschis[i].append(el.chi2)
            ncratesvals[i].append(el.ncrate[temp_bin])
        ncrateschi_argmin[i] = np.argmin(ncrateschis[i])
        ncrateschimin[i] = ncrateschis[i][ncrateschi_argmin[i]]
        least_nld_gsf.append(ncrates[ncrateschi_argmin[i]])

#save best fits
np.save('data/generated/best_fits_' + NLD_pathstring + '.npy', least_nld_gsf)

#Save values and uncertainties in tables
valmatrices = [[],[]]
for lst, lab, i in zip([nlds, gsfs], ['nld_' + NLD_pathstring,'gsf_' + NLD_pathstring], [0,1]):
    valmatrices[i] = calc_errors_chis(lst)
    header = 'Energy [MeV], best_fit, best_fit-sigma, best_fit+sigma, staterr' 
    writematr = np.c_[valmatrices[i],least_nld_gsf[i].yerr]
    np.savetxt('data/generated/' + lab + '_whole.txt', writematr, header = header) 
if astro:
    astrovalmatrices = [0,0]
    MACSvalmatrices = [0,0]
    for i,omp,ncrates in zip([0,1], omps, ncrates_omps):
        astrovalmatrix = calc_errors_chis(ncrates)
        astrovalmatrices[i] = astrovalmatrix
        header = 'T [GK], best_fit, best_fit-sigma, best_fit+sigma' 
        np.savetxt('data/generated/ncrates_' + omp + '_' + NLD_pathstring + '_whole.txt',astrovalmatrix, header = header)
    
        MACSvalmatrix = calc_errors_chis_MACS(ncrates)
        MACSvalmatrices[i] = MACSvalmatrix
        header = 'E [keV], best_fit, best_fit-sigma, best_fit+sigma' 
        np.savetxt('data/generated/MACS_' + omp + '_' + NLD_pathstring + '_whole.txt',MACSvalmatrix, header = header)


#plot chi2?
if plot_chis:
    #plot chi2s for one energy
    fig1, axs = plt.subplots(nrows = 1, ncols = 2, sharey = True)
    if astro:
        fig2, axs2 = plt.subplots(nrows = 1, ncols = 1)
    #plot chi distributions
    axs[0].plot(nldvals,nldchis,'b.',alpha=0.4, markersize = 3)
    axs[1].plot(gsfvals,gsfchis,'b.',alpha=0.1, markersize = 3)
    axs[0].plot(nldvals[nldchi_argmin],nldchimin,'k^', label=r'$\chi_{min}^2$')
    axs[1].plot(gsfvals[gsfchi_argmin],gsfchimin,'k^', label=r'$\chi_{min}^2$')
    for i,chimin in enumerate([nldchimin, gsfchimin]):
        axs[i].plot(valmatrices[i][energy_bin,2], chimin+1, 'ro')
        axs[i].plot(valmatrices[i][energy_bin,3], chimin+1, 'ro')
    axs[0].axhline(y=nldchimin+1, color='r', linestyle='--', label=r'$\chi_{min}^2$+1 score')
    axs[1].axhline(y=gsfchimin+1, color='r', linestyle='--', label=r'$\chi_{min}^2$+1 score')
    rng = 2
    if astro:
        colors = ['b', 'm']
        for i, astrovalmatrix in zip([0,1],astrovalmatrices):
            axs2.plot(ncratesvals[i],ncrateschis[i], color = colors[i], marker = '.', linestyle = None, alpha=0.5)
            #axs2.plot(ncratesvals[i],ncrateschis[i], color = colors[i], marker = '.', alpha=1)
            axs2.plot(ncratesvals[i][ncrateschi_argmin[i]],ncrateschimin[i],'go', label=r'$\chi_{min}^2$')
            axs2.plot(astrovalmatrix[temp_bin,2], ncrateschimin[i]+1, 'ro')
            axs2.plot(astrovalmatrix[temp_bin,3], ncrateschimin[i]+1, 'ro')
        axs2.axhline(y=ncrateschimin[0]+1, color='r', linestyle='--', label=r'$\chi_{min}^2$+1 score')
        
    #axs[0].set_title(r'$\chi^2$-scores for $\rho(E_x=$ %s MeV)'%"{:.2f}".format(nld.energies[energy_bin]))
    #axs[0].set_ylim(26.4,40)
    #axs[0].set_xlim(180,232)
    axs[0].set_xlabel('NLD [MeV$^{-1}$]')
    axs[0].set_ylabel(r'$\chi^2$-score')
    axs[0].text(0.9, 0.05, 'a)', fontsize='medium', verticalalignment='center', fontfamily='serif', transform = axs[0].transAxes)
    #axs[1].set_title(r'$\chi^2$-scores for $f(E_\gamma=$ %s MeV)'%"{:.2f}".format(gsf.energies[energy_bin]))
    #axs[1].set_ylim(26.4,40)
    #axs[1].set_xlim(7e-9,1.6e-8)
    axs[1].set_xlabel('GSF [MeV$^{-3}$]')
    axs[1].text(0.9, 0.05, 'b)', fontsize='medium', verticalalignment='center', fontfamily='serif', transform = axs[1].transAxes)
    for i in range(rng):
        #axs[i].grid()
        axs[i].legend(loc='upper right', framealpha = 1.)
    if astro:
        axs2.set_xlabel(r'$\sigma_n$ [mb]')
        #axs2.grid()
        axs2.legend()
        axs2.set_title(r'$\chi^2$-scores for $\sigma_n(T=$ %s GK)'%"{:.2f}".format(ncrates[0].T[temp_bin]))
        fig2.show()
        
    plt.tight_layout()
    fig1.show()