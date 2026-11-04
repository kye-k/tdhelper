# -*- coding: utf-8 -*-

from shapely.geometry import Point
import numpy as np


class _FilteringMixin:

    def filter_by_count(self, min_length=0, max_length=None):
        '''
        filters data by trajectory length;
        min_length: minimum trajectory length to keep (inclusive)
        max_length: maximum trajectory length to keep (inclusive)

        '''
        
        if max_length != None:
        
            self.countDF = self.data.groupby('tr', as_index=False)['t'].count().rename(columns={'t': 'Count'})
            self.countDF = self.countDF[self.countDF['Count'] >= min_length]
            self.countDF = self.countDF[self.countDF['Count'] <= max_length]
            self.data = self.data[self.data['tr'].isin(self.countDF['tr'])]
        
        else:
            
            self.countDF = self.data.groupby('tr', as_index=False)['t'].count().rename(columns={'t': 'Count'})
            self.countDF = self.countDF[self.countDF['Count'] >= min_length]
            self.data = self.data[self.data['tr'].isin(self.countDF['tr'])]

        print()

    def filter_by_tr(self, tr_id):
        '''
        filters data by trajectory id;
        tr_id: list of trajectory id to keep
        
        '''
        self.data = self.data[self.data['tr'].isin(tr_id)]

    def filter_by_frame(self, min_frame, max_frame):
        '''
        filters data by frame number; note, makes use of the centroid time...

        Parameters
        ----------
        min_frame : int
            inclusive; minimum frame number
        max_frame : int
            inclusive; maximum frame number

        Returns
        -------
        filteres the data        

        '''
        
        helperDF = self.data.groupby('tr').mean().reset_index()
        tr_id = helperDF.query('t >= @min_frame and t <= @max_frame').tr
        self.data = self.data[self.data['tr'].isin(tr_id)]

    def filter_by_centroid(self, xmin, xmax, ymin, ymax):
        '''

        Parameters
        ----------
        xmin : TYPE
            DESCRIPTION.
        xmax : TYPE
            DESCRIPTION.
        ymin : TYPE
            DESCRIPTION.
        ymax : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        
        tempDF = self.data.groupby('tr').mean().reset_index()
        tr_id = tempDF.query('x > @xmin and x < @xmax and y > @ymin and y < @ymax').tr
        self.data = self.data[self.data['tr'].isin(tr_id)]

    def filter_by_roi(self, roi, inverse=False):
        '''

        Parameters
        ----------
        roi : TYPE
            shapely polygon

        Returns
        -------
        None.

        '''

        # quick query
        minx, miny, maxx, maxy = roi.bounds
        self.create_centroidDF()
        temp_data = self.centroidDF.query('x > @minx and x < @maxx and y > @miny and y < @maxy')

        # further query
        centroid_array = np.asarray(temp_data[['x', 'y']])
        
        # count trajectories in ROI
        tr_list = []
        
        for xy, tr in zip(centroid_array, np.array(temp_data.tr)):
            if roi.contains(Point(xy)):
                tr_list.append(tr)
        
        if not inverse:
            self.filter_by_tr(tr_list)
        elif inverse:
            self.filter_by_tr([x for x in self.centroidDF.tr if x not in tr_list])
