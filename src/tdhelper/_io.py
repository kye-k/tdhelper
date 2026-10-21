# -*- coding: utf-8 -*-

import copy
import numpy as np
import pandas as pd


class _IOMixin:

    def import_trc(self, fname, pixel_size):
        '''
        imports trc file generated from PALMTracer

        '''
        trcDF = pd.read_csv(fname, delimiter='\t', skiprows=2)
        self.data['tr'] = trcDF['Track']
        self.data['x'] = trcDF['CentroidX(px)'] * pixel_size
        self.data['y'] = trcDF['CentroidY(px)'] * pixel_size
        self.data['t'] =  trcDF['Plane']

    def import_trxyframe(self, fname):
        '''
        imports .trxyt file format; last column is frame id

        '''
        self.data = pd.read_csv(fname, delimiter=' ', header=None)
        self.data.columns = ['tr', 'x', 'y', 't']

    def import_trxytime(self, fname, DT):
        '''
        imports .trxyt file format; last column is time (seconds)
        function converts time to frame id

        '''
        self.data = pd.read_csv(fname, delimiter=' ', header=None)
        self.data.columns = ['tr', 'x', 'y', 't']
        self.data['t'] = np.round(self.data['t'] / DT).astype('int')

    def import_straight(self, trj_vec, x_vec_um, y_vec_um, t_vec):
        '''
        imports data straight from vectors, specifying tr, x, y, and t

        '''
        self.data['tr'] = trj_vec
        self.data['x'] = x_vec_um
        self.data['y'] = y_vec_um
        self.data['t'] =  t_vec

    def import_trackmatecsv(self, fname):
        '''
        imports a .csv file generated from TrackMate
        compatible with TrackMate v7.6.1

        '''
        print('importing trackmate.csv data...')
        self.data = pd.read_csv(fname, skiprows=[0, 1, 3], encoding='unicode_escape')
        self.data = self.data[['Track ID', 'X', 'Y', 'Frame']]
        self.data.sort_values(by=['Track ID', 'Frame'], axis=0, ascending=True, inplace=True)
        self.data.rename(columns={'Track ID': 'tr',
                                  'X': 'x',
                                  'Y': 'y',
                                  'Frame': 't'},
                        inplace=True)

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
        self.data.sort_values(by=['tr', 't'], axis=0, ascending=True, inplace=True)

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
