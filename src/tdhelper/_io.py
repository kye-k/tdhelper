# -*- coding: utf-8 -*-

import copy
import warnings
import numpy as np
import pandas as pd


class _IOMixin:

    def _check_and_sort_data(self):
        order = pd.MultiIndex.from_frame(self.data[['tr', 't']])
        if not order.is_monotonic_increasing:
            warnings.warn(
                'Trajectory data is not ordered by trajectory ID and time; sorting it.',
                UserWarning,
                stacklevel=2,
            )
            self.data.sort_values(by=['tr', 't'], kind='stable', inplace=True)
        self.data.reset_index(drop=True, inplace=True)

        time_gaps = self.data.groupby('tr', sort=False)['t'].diff() > 1
        if time_gaps.any():
            trajectories = self.data.loc[time_gaps, 'tr'].unique().tolist()
            warnings.warn(
                f'Missing timesteps detected in trajectories: {trajectories}.',
                UserWarning,
                stacklevel=2,
            )

    def import_trc(self, fname, pixel_size):
        '''
        imports trc file generated from PALMTracer

        '''
        trcDF = pd.read_csv(fname, delimiter='\t', skiprows=2)
        self.data['tr'] = trcDF['Track']
        self.data['x'] = trcDF['CentroidX(px)'] * pixel_size
        self.data['y'] = trcDF['CentroidY(px)'] * pixel_size
        self.data['t'] =  trcDF['Plane']
        self._check_and_sort_data()

    def import_trxyframe(self, fname):
        '''
        imports .trxyt file format; last column is frame id

        '''
        self.data = pd.read_csv(fname, delimiter=' ', header=None)
        self.data.columns = ['tr', 'x', 'y', 't']
        self._check_and_sort_data()

    def import_trxytime(self, fname, DT):
        '''
        imports .trxyt file format; last column is time (seconds)
        function converts time to frame id

        '''
        self.data = pd.read_csv(fname, delimiter=' ', header=None)
        self.data.columns = ['tr', 'x', 'y', 't']
        self.data['t'] = np.round(self.data['t'] / DT).astype('int')
        self._check_and_sort_data()

    def import_straight(self, trj_vec, x_vec_um, y_vec_um, t_vec):
        '''
        imports data straight from vectors, specifying tr, x, y, and t

        '''
        self.data['tr'] = trj_vec
        self.data['x'] = x_vec_um
        self.data['y'] = y_vec_um
        self.data['t'] =  t_vec
        self._check_and_sort_data()

    def import_trackmatecsv(self, fname):
        '''
        imports a .csv file generated from TrackMate
        compatible with TrackMate v7.6.1

        '''
        print('importing trackmate.csv data...')
        self.data = pd.read_csv(fname, skiprows=[0, 1, 3], encoding='unicode_escape')
        self.data = self.data[['Track ID', 'X', 'Y', 'Frame']]
        self.data.rename(columns={'Track ID': 'tr',
                                  'X': 'x',
                                  'Y': 'y',
                                  'Frame': 't'},
                        inplace=True)
        self._check_and_sort_data()

    def import_trackmatecsv_v520(self, fname):
        '''
        imports a .csv file generated from TrackMate
        compatible with TrackMate v5.2.0

        '''
        print('importing trackmate.csv data...')
        self.data = pd.read_csv(fname)
        self.data = self.data[['TRACK_ID', 'POSITION_X', 'POSITION_Y', 'FRAME']]
        self.data.rename(columns={'TRACK_ID': 'tr',
                                  'POSITION_X': 'x',
                                  'POSITION_Y': 'y',
                                  'FRAME': 't'}, 
                         inplace=True)
        self.data = self.data.mask(self.data.eq('None')).dropna()
        self.data['tr'] = self.data['tr'].astype('int')
        self._check_and_sort_data()

    def export_trxyframe(self, fname):
        '''
        writes a .trxyt file format; last column is frame id
        
        '''
        with open(fname, 'a', newline='') as file:
            file.truncate(0)
            self.data.to_csv(file, index=False, sep=' ', header=None)

    def export_trxytime(self, fname, DT, round=False):
        '''
        writes a .trxyt file format; last column is time (seconds)

        '''
        with open(fname, 'a', newline='') as file:
            file.truncate(0)
            export_data = copy.deepcopy(self.data)
            export_data['t'] = export_data['t'] * DT
            
            if round:
                export_data['t'] = export_data['t'].round(3)
            
            export_data.to_csv(file, index=False, sep=' ', header=None)

    def export_saSPT(self, fname):
        '''
        writes a .csv file format that is compatible with saSPT (v0.4.0)
        
        '''
        with open(fname, 'a', newline='') as file:
            file.truncate(0)
            export_data = copy.deepcopy(self.data)
            export_data = pd.DataFrame(data={'x': self.data['x'],
                                             'y': self.data['y'],
                                             'trajectory': self.data['tr'],
                                             'frame': self.data['t']})
            export_data.to_csv(file, index=False, sep=' ', header=True)
