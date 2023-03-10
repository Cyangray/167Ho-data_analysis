#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 19:09:37 2022

@author: francesco updated 1st February 2022

Import data from calculated best fit gsf, and try to fit for the GDR with a GLO plus one Gaussian for the pygmy.
Plot together.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from systlib import GLO_arglist, GLO_hybrid_arglist, SLO_arglist, load_known_gsf
from readlib import search_string_in_file
from scipy import interpolate
from iminuit import Minuit
from iminuit.cost import LeastSquares
'''
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'font.serif': ['computer Modern'],
    'text.usetex': True,
    'pgf.rcfonts': False,
})
'''
NLD_pathstring = 'FG'

#constants
hc = 197.3269804 #MeV*fm
const = 1/(3*np.pi**2*hc**2*10) #mb
BSR_constant = 2.5980e8 #u_N^2 MeV^2
Z = 67
A = 167

#useful lists
E_range1 = np.linspace(2.0,4.0,200)
E_range2 = np.linspace(0,10,500)

#Colormap
cmap = matplotlib.cm.get_cmap('YlGnBu')

#import best fits
best_fits = np.load('data/generated/best_fits_' + NLD_pathstring + '.npy', allow_pickle = True)
best_gsf = best_fits[1]
best_gsf.clean_nans()
best_gsf.delete_point(-1)
best_nld = best_fits[0]
best_nld.clean_nans()
curr_gsf = best_gsf

#load known gsfs
Ho165_varlamov = load_known_gsf(165, 'Ho', author = 'varlamov')
Ho165_varlamov.label = r"$^{165}$Ho, Varlamov"
gdr_data = Ho165_varlamov

#load 167Ho GognyM1
pathM1 = '/home/francesco/talys/structure/gamma/gognyM1/Ho.psf'
GognyM1 = np.loadtxt(pathM1, skiprows = search_string_in_file(pathM1, f'A= {A}') + 2, max_rows = 300)
GognyM1_x = GognyM1[:,0]
GognyM1_y = GognyM1[:,1]    
f_gogny = interpolate.interp1d(GognyM1_x, GognyM1_y*const)

#import experimental nld and gsf
gsf_mat = np.genfromtxt('data/generated/gsf_' + NLD_pathstring + '_whole.txt', unpack = True).T
#delete rows with nans
gsf_mat = gsf_mat[~np.isnan(gsf_mat).any(axis=1)]
gsf_mat = gsf_mat[:-1]

yerrs = np.zeros_like(best_gsf.yerr)
for i in range(len(yerrs)):
    y_syst_down = best_gsf.y[i] - gsf_mat[i,3]
    y_syst_up = gsf_mat[i,4] - best_gsf.y[i]
    max_systerr = max(y_syst_down, y_syst_up)
    yerrs[i] = np.sqrt(best_gsf.yerr[i]**2 + max_systerr**2)
    
#initialize common stuff
x_values_cont = np.linspace(0.1, 20, 1000)

plot_switch = '0.59'

#Pick_function to model the second gdr (GLO_hybrid for berman and bergere. GLO for renstr??m)
gdr2_func = GLO_arglist

def BSR_integral_old(m, E_range):
    mean = BSR_constant*np.trapz(SLO_arglist(E_range, m.values[7:10]),E_range)
    low_err = mean - BSR_constant*np.trapz(SLO_arglist(E_range, np.subtract(m.values[7:10],m.errors[7:10])),E_range)
    upp_err = BSR_constant*np.trapz(SLO_arglist(E_range, np.add(m.values[7:10],m.errors[7:10])),E_range) - mean
    return [mean, low_err, upp_err]
    
def BSR_integral(m, E_range):
    mean = BSR_constant*np.trapz(SLO_arglist(E_range, m.values[7:10]),E_range)
    rel_errs = [(j/k)**2 for j,k in zip(m.errors[7:10],m.values[7:10])]
    rel_err = np.sqrt(np.sum(rel_errs))
    abs_err = rel_err*mean
    return [mean, abs_err, abs_err]

def Sum_function(x, params):
    sum_func = (gdr2_func(x,params[:4])
                + gdr2_func(x,[params[0], *params[4:7]])
                + SLO_arglist(x, params[7:10])
                + SLO_arglist(x, params[10:13])
                #+ f_gogny(x)
                #+ SLO_arglist(x, params[13:16])
                #+ SLO_arglist(x, params[16:])
                )
    return sum_func

def Sum_gdrs(x, params):
    sum_func = (gdr2_func(x,params[:4])
                + gdr2_func(x,[params[0], *params[4:7]])
                )
    return sum_func


fits = []
Ts = [0.59, 0.66]
for i, T in enumerate(Ts):
    #join data to experimental for the fit
    data_x = np.concatenate([curr_gsf.energies, gdr_data.energies])
    data_y = np.concatenate([curr_gsf.y, gdr_data.y])
    data_yerr = np.concatenate([yerrs, gdr_data.yerr])
    least_squares_gdr = LeastSquares(data_x, data_y, data_yerr, Sum_gdrs)
    least_squares = LeastSquares(data_x, data_y, data_yerr, Sum_function)
    
    #temperature
    T_GEDR = T
    
    if T == 0.66:
        #GEDR1
        E_GEDR1 = 12.341
        G_GEDR1 = 3.22
        S_GEDR1 = 330
        #GEDR2
        E_GEDR2 = 14.78
        G_GEDR2 = 1.897
        S_GEDR2 = 194
    else:
        #GEDR1
        E_GEDR1 = 12.359
        G_GEDR1 = 3.353
        S_GEDR1 = 323.6
        #GEDR2
        E_GEDR2 = 14.784
        G_GEDR2 = 1.891
        S_GEDR2 = 189.3
        
    params_init_gdr = [T_GEDR,
                   E_GEDR1, G_GEDR1, S_GEDR1, 
                   E_GEDR2, G_GEDR2, S_GEDR2
                   ]
    names_gdr = ['T_GEDR',
             'E_GEDR1', 'G_GEDR1', 'S_GEDR1',
             'E_GEDR2', 'G_GEDR2', 'S_GEDR2'
             ]
    
    m1 = Minuit(least_squares_gdr, params_init_gdr, name = names_gdr)
    m1.fixed['T_GEDR'] = True
    '''
    for indx, p in enumerate(params_init_gdr):
        m1.limits[indx] = (p/2,p*2)
    
    m1.fixed['E_GEDR2'] = True
    m1.fixed['G_GEDR2'] = True
    m1.fixed['S_GEDR2'] = True
    '''
    #m1.limits['T_GEDR'] = (T_GEDR*0.99,T_GEDR*1.01)
    m1.migrad()
    m1.hesse()
    
    print(m1.params)
    
    #temperature
    T_GEDR = m1.values[0]
    #GEDR1
    E_GEDR1 = m1.values[1]
    G_GEDR1 = m1.values[2]
    S_GEDR1 = m1.values[3]
    #GEDR2
    E_GEDR2 = m1.values[4]
    G_GEDR2 = m1.values[5]
    S_GEDR2 = m1.values[6]
    
    if T == 0.66:
        #SR1
        E_SR1 = 3.18#2.6#2.84
        G_SR1 = 0.794#1.3
        S_SR1 = 0.43
        #PDR1
        E_PDR1 = 5.50
        G_PDR1 = 1.05#2.1
        S_PDR1 = 3.7
    else:
        #SR1
        E_SR1 = 3.15#2.6#2.84
        G_SR1 = 1.01#0.794#1.3
        S_SR1 = 0.44
        #PDR1
        E_PDR1 = 5.82
        G_PDR1 = 1.87#2.1
        S_PDR1 = 4.2#3.83
    
    params_init = [T_GEDR,
                   E_GEDR1, G_GEDR1, S_GEDR1, 
                   E_GEDR2, G_GEDR2, S_GEDR2,
                   E_SR1, G_SR1, S_SR1,
                   E_PDR1, G_PDR1, S_PDR1
                   ]
    names = ['T_GEDR',
             'E_GEDR1', 'G_GEDR1', 'S_GEDR1',
             'E_GEDR2', 'G_GEDR2', 'S_GEDR2',
             'E_SR1', 'G_SR1', 'S_SR1',
             'E_PDR1', 'G_PDR1', 'S_PDR1'
             ]
    
    m = Minuit(least_squares, params_init, name = names)
    '''
    for name,p in zip(names,params_init):
        m.limits[name] = (p*0.80,p*1.4)
    m.limits['E_PDR1'] = (p*0.90,p*1.2)
    '''
    for name,param in zip(names,params_init):
        if name in ['T_GEDR','E_GEDR1', 'G_GEDR1', 'S_GEDR1','E_GEDR2', 'G_GEDR2', 'S_GEDR2']:
            m.fixed[name] = True
        else:
            m.limits[name] = (0,param*1.05)
    '''
    for indx in range(0,7):
        m.fixed[indx] = True
    '''
    
    m.limits['E_PDR1'] = (E_PDR1*0,E_PDR1*1.1)
    m.limits['G_PDR1'] = (G_PDR1*0.99,G_PDR1*1.01)
    m.limits['S_PDR1'] = (S_PDR1*0,S_PDR1*1.2)
    m.limits['E_SR1'] = (E_SR1*0.50,E_SR1*1.5)
    m.limits['G_SR1'] = (G_SR1*0.50,G_SR1*1.5)
    m.limits['S_SR1'] = (S_SR1*0.50,S_SR1*1.5)
    
    #m.limits['E_SR1'] = (E_SR1*0,E_SR1*1.2)
    #m.limits['G_SR1'] = (G_SR1*0,G_SR1*1.2)
    #m.limits['S_SR1'] = (S_SR1*0,S_SR1*10)
    #m.scan()
    #print(m.params)
    m.migrad()
    m.hesse()
    print(m.params)
    #calculate actual chi2:
    chi2_Oslo_data = 0
    for j, energy in enumerate(curr_gsf.energies):
        chi2_Oslo_data += (Sum_function(energy, m.values) - curr_gsf.y[j])**2/(curr_gsf.yerr[j])**2
    print('chi2 Oslo data: %f'%chi2_Oslo_data)
    
    # display legend with some fit info
    fit_info = [
        f"$\\chi^2$ / $n_\\mathrm{{dof}}$ = {m.fval:.1f} / {len(data_x) - m.nfit}",
    ]
    dof = len(data_x)-m.nfit
    fvalint = int(m.fval)
    print("chi2 / n_dof = %d / %d"%(fvalint, dof))
    
    #plot
    if plot_switch == '%.2f'%(Ts[i]):
        fig0,ax0 = plt.subplots(figsize = (5.0, 3.75), dpi = 300)
        ax0.plot(x_values_cont, gdr2_func(x_values_cont, m.values[:4]), color = cmap(1/3), linestyle = '--', label = 'GEDR1 fit')
        ax0.plot(x_values_cont, gdr2_func(x_values_cont, [m.values[0], *m.values[4:7]]), color = cmap(2/3), linestyle = '--', label = 'GEDR2 fit')
        ax0.plot(x_values_cont, SLO_arglist(x_values_cont, m.values[7:10]), color = cmap(1/3), linestyle = '-.', label = 'SR fit')
        ax0.plot(x_values_cont, SLO_arglist(x_values_cont, m.values[10:13]), color = cmap(2/3), linestyle = '-.', label = 'PDR fit')
        ax0.plot(GognyM1[:,0], GognyM1[:,1]*const, color = cmap(1/3), linestyle = ':', label = 'Gogny M1')
        ax0.semilogy(x_values_cont, Sum_function(x_values_cont, m.values), color = cmap(2/3), linestyle = '-', label="Total fit")
        gdr_data.plot(color = cmap(1/4), ax=ax0, linestyle = 'None', marker = '.')
        curr_gsf.plot(color = cmap(4/4), ax=ax0, linestyle = 'None', marker = '.', label=r"$^{166}$Ho, this work")
        
        ax0.set_xlim([0,10])
        ax0.set_ylim([7e-9, 3e-6])
        ax0.set_ylabel(r'GSF [MeV$^{-3}$]')
        ax0.set_xlabel(r'$E_\gamma$ [MeV]')
        ax0.legend(ncol = 2, frameon = False)# loc = 'lower right')
        Tstr = str(int(T*100))
        fig0.tight_layout()
        fig0.savefig('pictures/gsf_fit_minuit3_' + Tstr + '.pdf', format = 'pdf')#, bbox_inches='tight')
        fig0.show()
    
    
    #TRK
    N=A-Z
    TRK_sum = 60*N*Z/A #mb*MeV
    TRK_fac = np.trapz(SLO_arglist(x_values_cont, m.values[10:13]), x = x_values_cont)/(const*TRK_sum)
    
    #TRK_fac_max = PDR_to_MeVbarn(C1+C1_unc,Sigma1+Sigma1_unc,E1+E1_unc)/TRK_sum
    #TRK_fac_min = PDR_to_MeVbarn(C1-C1_unc,Sigma1-Sigma1_unc,E1-E1_unc)/TRK_sum
    print('TRK fraction: %s percent'%(TRK_fac*100))
    #print('TRK err up: %s percent'%((TRK_fac_max-TRK_fac)*100))
    #print('TRK err down: %s percent'%((-TRK_fac_min+TRK_fac)*100))
    
    #BSR        
    BSR_1 = BSR_integral(m, E_range1)
    BSR_2 = BSR_integral(m, E_range2)
    print('B_SR for 2.0-4.0 MeV: %f, -%f, +%f'%(BSR_1[0], BSR_1[1], BSR_1[2]))
    print('B_SR for 0-10 MeV: %f, -%f, +%f'%(BSR_2[0], BSR_2[1], BSR_2[2]))
    
    fits.append(m)
    
    print('LaTeX output')
    printstring1 = ''
    printstring2 = ''
    for p, v, e in zip(m.parameters, m.values, m.errors):
        if p in ['E_SR1', 'G_SR1', 'S_SR1']:
            printstring2 += "%.3f(%f) & "%(v,e)
        else:
            printstring1 += "%.3f(%f) & "%(v,e)
    
    for BSR in [BSR_1, BSR_2]:
        unc = max([BSR[1], BSR[2]])
        printstring2 += '%f(%f) & '%(BSR[0], unc)
    
    print(printstring1)
    print(printstring2)

for i, T in enumerate(Ts):
    fit = fits[i]
    #Save results
    BSR2040_0 = BSR_integral(fit, E_range1)
    BSR0010_0 = BSR_integral(fit, E_range2)
    
    output = np.array([[A, fit.values[7], fit.errors[7], fit.values[8], fit.errors[8], fit.values[9], fit.errors[9], BSR2040_0[0], (BSR2040_0[1] + BSR2040_0[2])/2,  BSR0010_0[0], (BSR0010_0[1] + BSR0010_0[2])/2, 0.3424]])
    
    header = 'line 1: Varlamov. E, err, gamma, err, sigma, err, BSR 2.0-4.0MeV, err, BSR 0-10MeV, err, def'
    Tstr = str(int(T*100))
    np.savetxt('data/generated/metode2_'+Tstr+'.txt', output, header = header, fmt='%.4f')













