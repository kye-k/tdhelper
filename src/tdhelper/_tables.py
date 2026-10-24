# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


class _TableMixin:

    def create_countDF(self):
        return self.data.groupby('tr', as_index=False)['t'].count().rename(columns={'t': 'Count'})

    def create_centroidDF(self):
        return self.data.groupby('tr', as_index=False).mean()[['tr', 'x', 'y']]

    def create_startendDF(self):
        data_vec = []
        
        for i in self.data.tr.unique():
            tempDF = self.data[self.data['tr'] == i]
            data_vec.append(list(tempDF.iloc[0, :]))
            data_vec.append(list(tempDF.iloc[-1, :]))
            
        return pd.DataFrame(np.array(data_vec), columns=['tr', 'x', 'y', 't'])

    def create_surv_probDF(self, DT, countDF):
        '''
        Creates surv_probDF, which contains cumulative distribution
        information about the trajectory lengths observed.

        Parameters
        ----------
        countDF : pandas.DataFrame
            Output from create_countDF().
        DT : float
            Sampling rate of image acquisition.

        Returns
        -------
        pandas.DataFrame
            Survival probability information.

        '''
        
        # creating dataframe
        surv_probDF = countDF.copy(deep=True)
        surv_probDF.rename(columns={'Count': 'trj_len'}, inplace=True)
        surv_probDF = surv_probDF.groupby('trj_len', as_index=False).count().rename(columns={'tr': 'count'})
        surv_probDF.sort_values('trj_len', ascending=False, inplace=True)
        surv_probDF['cumsum'] = surv_probDF['count'].cumsum()
        surv_probDF['cumsum_perc'] = surv_probDF['cumsum'] / surv_probDF['cumsum'].max() # turn to percentages
        surv_probDF['trj_len_s'] = surv_probDF['trj_len'] * DT

        return surv_probDF
