# -*- coding: utf-8 -*-

from scipy.stats import linregress
from scipy.stats import sem
import math
import numpy as np


class _MSDMixin:

    def calcMsd(self, trj):
        '''
        Function to calculate the MSD curve for a given trajectory.

        '''
        
        msd = np.zeros(len(trj))
        
        for deltaT in range(1, len(trj)):
            
            disp = np.array([math.dist(a, b) for a, b in zip(trj[deltaT:], trj[:-deltaT])])
            msd[deltaT] = np.mean(disp ** 2)

        return msd

    def calc_mean_msd(self):
        '''
        fucntion: calculates mean msd values (requires msd_list prior)
        
        '''
        
        # length of longest list (or trajectory)
        max_length = np.max([len(i) for i in self.msd_list])

        # create vector to store mean and sem values
        self.msd_mean_values = np.zeros(max_length)
        self.msd_sem_values = np.zeros(max_length)
        
        for i in range(0, max_length):
            vector = []
            for msd_vals in self.msd_list:
                if i < len(msd_vals):
                    vector.append(msd_vals[i])
                
            self.msd_mean_values[i] = np.mean(vector)
            self.msd_sem_values[i] = sem(vector)

    def calc_MSD(self, dictionary_output = False):
        '''
        Wrapper to calculate the MSD curve for each trajectory, stored
        within msd_list.

        '''
        
        msd_dict = {}
        
        for trj in self.data['tr'].unique():
            trj_data = self.data[self.data['tr'] == trj][['x', 'y']].values
            msd_dict[trj] = self.calcMsd(trj_data)
        
        self.msd_list = list(msd_dict.values())
        self.msd_dict = msd_dict
        
        if dictionary_output == True:
            return msd_dict

    def calc_msdauc(self, p1, p2, DT):
        
        self.msdauc = np.trapz(self.msd_mean_values[p1:p2], dx=DT)
        print('msd_auc has been calculated from {} to {}, note unit is in (s)'.format(p1, p2 - 1))

    def calc_diffusion_coefficient(self, DT, fit_len = 4, dimensions = 2):
        '''
        Calculates the diffusion coefficient for each trajectory via
        performing linear regression on its MSD curve.
        
        Parameters
        ----------
        DT : float
            Time interval between acquisition frames.
        fit_len : int, optional
            Number of points in MSD to fit the data to. The default is 4.
        dimensions : int, optional
            Dimensions for data (i.e., 1D, 2D, 3D). The default is 2.

        '''
        
        diff_dict = {}
        difflog_dict = {}
        
        for i in self.msd_dict.keys():
            
            # perform linear regression and calculate slope
            fit = linregress(np.arange(1, fit_len + 1) * DT, self.msd_dict[i][1:fit_len + 1]) # note msd[0] is 0; we do not include this for fitting
            value = (fit.slope / (2 * dimensions))
            
            # according to PALMTracer
            if value <= 10 ** -5: 
                value = 10 ** -5
            
            diff_dict[i] = value
            difflog_dict[i] = np.log10(value)
        
        self.diff_dict = diff_dict
        self.difflog_dict = difflog_dict
        
        
        self.diff_list = np.array(list(diff_dict.values()))
        self.difflog_list = np.array(list(difflog_dict.values()))

    def calc_immobile_fraction(self, dcoef_thresh):
        '''
        Determines the proportion (0 to 1) of molecules classified as 'immobile' based on threshold
        
        '''
        
        if len(self.diff_list) > 0:
            imm = self.diff_list < dcoef_thresh
            proportion_immobile = sum(imm) / len(self.diff_list)
        
        else:
            proportion_immobile = None
        
        return proportion_immobile
