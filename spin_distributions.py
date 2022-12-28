#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 11:11:19 2022

@author: francesco

Script to find out what is the average spin of a nucleus at a certain excitation
energy, given its list of levels.
"""

import numpy as np
import matplotlib.pyplot as plt

path_spinfile = 'spins_167Ho'
energy_sigma = 150 #+- window for which to calculate the average
centroids = np.arange(energy_sigma,1800,10)

def spin_parity(line_chunk):
    print(line_chunk)
    if '+' in line_chunk:
        parity = 1
        line_chunk = line_chunk.replace('+', '')
    elif '-' in line_chunk:
        parity = 0
        line_chunk = line_chunk.replace('-', '')
    else:
        parity = 0.5
    spin = float(line_chunk)
    return parity, spin

def separate_parity_certainty(energies, lss):
    #separate string inputs a la "1+, (6-)" and so on, in a five-column matrix
    #where 1+ = [energy,1,1,1,mult], (6-)=[energy,6,0,0,mult], where the third column is the parity
    #(0 is -, 1 is +), and the fourth column is the certainty (1 is certain, no 
    #parenthesis, while 0 is uncertain, spin in parenthesis), the fifth is how
    #many spins are suggested for the energy level
    output_list = []
    for energy, line in zip(energies[:,0], lss):
        if '/2' in line:
            factor = 0.5
            line = line.replace('/2', '')
        else:
            factor = 1
        #certain?
        if line[0]=='(':
            certainty = 0
        else:
            certainty = 1
        line = line.replace('(', '')
        line = line.replace(')', '')
        #number of possible spins:
        line = line +','
        nspins = line.count(',')
        positions_of_commas = [pos for pos, char in enumerate(line) if char == ',']
        for i, comma_idx in enumerate(positions_of_commas):
            end = comma_idx
            if i == 0:
                start = 0
            else:
                start = positions_of_commas[i-1]+1
            line_chunk = line[start:end]
            parity, spin = spin_parity(line_chunk)
            out_line = [energy, spin*factor, parity, certainty, nspins]
            output_list.append(out_line)
        
    return np.array(output_list)

level_energies = np.genfromtxt(path_spinfile)[:,:-1]
level_spins_str = np.genfromtxt(path_spinfile, dtype = str)[:,-1]
level_spins_whole = separate_parity_certainty(level_energies, level_spins_str)

def gen_hist_matrix(energy_centroid, energy_sigma, level_spins_whole):
    spin_range = max(level_spins_whole[:,1])
    hist_matrix = np.zeros((int(spin_range + 1), 2))
    hist_matrix[:,0] = np.arange(int(spin_range + 1))
    for level in level_spins_whole:
        if (level[0] > (energy_centroid - energy_sigma)) and (level[0] < (energy_centroid + energy_sigma)):
            hist_matrix[int(level[1]),1] += 1/level[4]
    return hist_matrix

matrices = []
avgs = []
for centroid in centroids:
    matrices.append(gen_hist_matrix(centroid, energy_sigma, level_spins_whole))
    avg = np.sum(matrices[-1][:,0] * matrices[-1][:,1])/np.sum(matrices[-1][:,1])
    avgs.append(avg)

show_mat = np.zeros((len(matrices[0][:,1]), len(matrices)))
for i, matrix in enumerate(matrices):
    show_mat[:,i] = matrix[:,1]


fig1,ax1 = plt.subplots()
ax1.matshow(show_mat, origin = 'lower')
ax1.plot(avgs, 'w-')
fig1.show()

fig2,ax2 = plt.subplots()
ax2.scatter(level_spins_whole[:,0], level_spins_whole[:,1], alpha = 1/level_spins_whole[:,4])
ax2.plot(centroids, avgs, 'k-')
fig2.show()