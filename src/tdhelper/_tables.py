# -*- coding: utf-8 -*-

import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class _TableMixin:

    def create_countDF(self):
        self.countDF = self.data.groupby('tr', as_index=False)['t'].count().rename(columns={'t': 'Count'})

    def create_centroidDF(self):
        self.centroidDF = self.data.groupby('tr', as_index=False).mean()[['tr', 'x', 'y']]

    def create_startendDF(self):
        data_vec = []
        
        for i in self.data.tr.unique():
            tempDF = self.data[self.data['tr'] == i]
            data_vec.append(list(tempDF.iloc[0, :]))
            data_vec.append(list(tempDF.iloc[-1, :]))
            
        self.startendDF = pd.DataFrame(np.array(data_vec), columns=['tr', 'x', 'y', 't'])

    def create_surv_probDF(self, DT, plot = False):
        '''
        Creates surv_probDF, which contains cumulative distribution
        information about the trajectory lengths observed.
        Requires countDF to be created prior.

        Parameters
        ----------
        DT : float
            Sampling rate of image acquisition.
        plot : bool, optional
            To plot survival probability graph or not. The default is False.

        Returns
        -------
        None.

        '''
        
        # creating dataframe
        self.surv_probDF = copy.deepcopy(self.countDF)
        self.surv_probDF.rename(columns={'Count': 'trj_len'}, inplace=True)
        self.surv_probDF = self.surv_probDF.groupby('trj_len', as_index=False).count().rename(columns={'tr': 'count'})
        self.surv_probDF.sort_values('trj_len', ascending=False, inplace=True)
        self.surv_probDF['cumsum'] = self.surv_probDF['count'].cumsum()
        self.surv_probDF['cumsum_perc'] = self.surv_probDF['cumsum'] / self.surv_probDF['cumsum'].max() # turn to percentages
        self.surv_probDF['trj_len_s'] = self.surv_probDF['trj_len'] * DT
        
        # plotting
        if plot == True:
            fig, ax = plt.subplots()
            self.plot_surv_prob(ax)
        
        return fig, ax
