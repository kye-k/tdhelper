# -*- coding: utf-8 -*-

from scipy.stats import linregress
from scipy.stats import sem
from scipy.integrate import trapezoid
import math
import numpy as np


def calc_msd(trj):
    '''
    function to calculate the MSD curve for a given trajectory

    '''

    msd = np.zeros(len(trj))

    for deltaT in range(1, len(trj)):

        disp = np.array([math.dist(a, b) for a, b in zip(trj[deltaT:], trj[:-deltaT])])
        msd[deltaT] = np.mean(disp ** 2)

    return msd


def calc_mean_msd(msd_list):
    '''
    function to calculatee mean msd values (requires msd_list prior)

    '''

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


def calc_msdauc(msd_mean_values, p1, p2, DT):
    '''
    integrates the mean MSD values from index p1 up to, but not including, index p2, using DT as the spacing between samples.
    '''

    msdauc = trapezoid(msd_mean_values[p1:p2], dx=DT)
    print('msd_auc has been calculated from {} to {}'.format(p1, p2 - 1))
    return msdauc


def calc_diffusion_coefficient(msd_dict, DT, fit_len = 4, dimensions = 2):
    '''
    calculates the diffusion coefficient for each trajectory via performing linear regression on its MSD curve.

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
    for i in msd_dict.keys():

        # perform linear regression and calculate slope
        fit = linregress(np.arange(1, fit_len + 1) * DT, msd_dict[i][1:fit_len + 1]) # note msd[0] is 0; we do not include this for fitting
        value = (fit.slope / (2 * dimensions))

        # according to PALMTracer
        if value <= 10 ** -5:
            value = 10 ** -5

        diff_dict[i] = value

    return diff_dict


def calc_immobile_fraction(diff_dict, dcoef_thresh):
    '''
    determines the proportion (0 to 1) of molecules classified as 'immobile' based on threshold

    '''

    if len(diff_dict) > 0:
        imm = np.array(list(diff_dict.values())) < dcoef_thresh
        proportion_immobile = sum(imm) / len(diff_dict)

    else:
        proportion_immobile = None

    return proportion_immobile


class _MSDMixin:

    def calc_MSD(self):
        '''
        wrapper to calculate the MSD curve for each trajectory, keyed by trajectory ID

        '''

        msd_dict = {}

        for trj in self.data['tr'].unique():
            trj_data = self.data[self.data['tr'] == trj][['x', 'y']].values
            msd_dict[trj] = calc_msd(trj_data)

        return msd_dict
