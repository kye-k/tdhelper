# -*- coding: utf-8 -*-

from scipy.stats import linregress
from scipy.stats import sem
from scipy.integrate import trapezoid
import math
import numpy as np


class _MSDMixin:

    def _MSD(self, trj):
        '''
        Calculate the MSD curve for a trajectory.

        Parameters
        ----------
        trj: numpy.ndarray
            2D array of trajectory localizations [[x, y]...].

        Returns
        -------
        numpy.ndarray
            MSD value for each time lag.

        '''
        
        msd = np.zeros(len(trj))
        
        for deltaT in range(1, len(trj)):
            disp = np.array([math.dist(a, b) for a, b in zip(trj[deltaT:], trj[:-deltaT])])
            msd[deltaT] = np.mean(disp ** 2)

        return msd

    def calc_meanMSD(self, msd_dict):
        '''
        Calculate mean and SEM of MSD curves.

        Parameters
        ----------
        msd_dict: dict
            Dictionary of MSDs (from self.calc_MSD)

        Returns
        -------
        msd_mean_values:
            Mean MSD curve.
        msd_sem_values:
            SEM of the mean MSD curve.
        
        '''

        # convert to list
        msd_list = list(msd_dict.values())
        
        # length of longest list (or trajectory)
        max_length = np.max([len(i) for i in msd_list])

        # create vector to store mean and sem values
        msd_mean_values = np.zeros(max_length)
        msd_sem_values = np.zeros(max_length)
        
        for i in range(0, max_length):
            vector = []
            for msd_vals in msd_list:
                if i < len(msd_vals):
                    vector.append(msd_vals[i])
            msd_mean_values[i] = np.mean(vector)
            msd_sem_values[i] = sem(vector)

        return msd_mean_values, msd_sem_values

    def calc_MSD(self):
        '''
        Calculate MSD curves for each trajectory.

        Returns
        -------
        msd_dict:
            MSD curves for each trajectory.

        '''
        
        msd_dict = {}
        
        for trj in self.data['tr'].unique():
            trj_data = self.data[self.data['tr'] == trj][['x', 'y']].values
            msd_dict[trj] = self._MSD(trj_data)
        
        return msd_dict

    def calc_MSDAUC(self, msd_mean_values, p1, p2, DT):
        '''
        Calculate the area under the curve (AUC) from using scipy.integrate.trapezoid.
        Integration occurs from index p1 up to, but not including index p2, using DT as the spacing between samples.

        Parameters
        ----------
        msd_mean_values: np.array
            List of MSD values.
        p1: int, optional
            Index of point to integrate from.
        P2: int, optional
            Index of point to integrate up to.
        DT:
            Time interval between acquisition frames (seconds).

        Returns
        -------
        float:
            AUC.

        '''
        
        return trapezoid(msd_mean_values[p1:p2], dx=DT)

    def calc_DCOEF(self, msd_dict, DT, fit_len=4, dimensions=2):
        '''
        Calculate the diffusion coefficient for each trajectory via performing linear regression on its MSD curve.
        Diffusion coefficients are calculated in the same way as PALMTracer (https://github.com/jbsiba/PALMTracer).
        Here, MSD curves are fit with equation, MSD(dt) = 4Ddt + a; and D is 'bottomed out' at 10 ** -5.

        Parameters
        ----------
        msd_dict: dict
            Dictionary of MSD curves for each trajectory (from calc_MSD).
        DT : float
            Time interval between acquisition frames (seconds).
        fit_len : int, optional
            Number of points in MSD to fit the data to. The default is 4.
        dimensions : int, optional
            Dimensions for data (i.e., 1D, 2D, 3D). The default is 2.
        
        Returns
        -------
        dict:
            Dictionary of diffusion coefficients (um2/s), where the key is trajectory ID.

        '''
        
        dcoef_dict = {}

        for i in msd_dict.keys():
            
            # perform linear regression and calculate slope
            # note msd[0] is 0; we do not include this for fitting
            fit = linregress(np.arange(1, fit_len + 1) * DT, msd_dict[i][1:fit_len + 1]) 
            value = (fit.slope / (2 * dimensions))
            
            # according to PALMTracer
            if value <= 10 ** -5: 
                value = 10 ** -5
            
            dcoef_dict[i] = value

        return dcoef_dict

    def calc_IMMFRAC(self, dcoef_dict, dcoef_thresh):
        '''
        Calculate the proportion (0 to 1) of molecules classified as immobile based on being < diffusion coefficient threshold.

        Parameters
        ----------
        dcoef_dict: dict
            Dictionary of diffusion coefficients (um2/s) (from calc_DCOEF), key is trajectory ID.
        dcoef_thresh: float
            Threshold diffusion coefficient (um2/s) at which trajectories are considered immobile. 
        
        Returns
        -------
        float:
            Proportion of trajectories with diffusion coefficient < diffusion coefficient threshold.

            
        '''

        diff_list = np.array(list(dcoef_dict.values()))
        
        if len(diff_list) > 0:
            imm = diff_list < dcoef_thresh
            proportion_immobile = sum(imm) / len(diff_list)
        
        else:
            proportion_immobile = None
        
        return proportion_immobile

    def calc_DISPLACEMENTS(self):
        '''
        Calculates the displacements between each step of for all trajectories.

        Returns
        -------
        dict :
            Dictionary of displacements between each step of a trajectory, with trajectory ID as keys.
        
        '''
        
        displacement_vector_dict = {}
        
        for i in self.data['tr'].unique():
            
            disp_vec = [] # create empty vector
            
            trj = self.data[self.data['tr'] == i]
            trj = np.array([trj.x, trj.y]).T.reshape(-1, 1, 2)
                        
            disp_vec = [np.sqrt((n0[0]-n1[0])**2 + (n0[1]-n1[1])**2) for n0, n1 in zip(trj[:-1, 0], trj[1:, 0])]
            displacement_vector_dict[i] = disp_vec

        return displacement_vector_dict