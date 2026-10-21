# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 10:35:10 2023
THIS VERSION IS THE VERSION THAT IS USED FOR ALL CODE IN THE PUBLICATION; DO NOT ALTER THIS FILE
THIS FILE WILL CONTINUE TO BE UPDATED IN "CORE" STARTING WITH V0

@author: uqkkudo
"""

#%% import

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mplcolors
import os
import seaborn as sns
import random
import time
from scipy.stats import rv_continuous
import copy
import cmocean
# import napari
from skimage import io
# import pytrackmate
import math
from scipy.stats import gaussian_kde
from scipy.stats import linregress
from scipy.stats import sem
import plotly.express as px
import plotly.io as pio

from scipy.spatial import Delaunay
from shapely.geometry import MultiLineString
from shapely.ops import unary_union, polygonize
import math

from shapely.geometry import Polygon
from shapely.geometry import MultiPolygon
from shapely.geometry import MultiPoint
from shapely.geometry import Point
from matplotlib import ticker

from shapely import distance as shapelydistance
from shapely import buffer as shapelybuffer
from shapely import difference as shapelydifference
from scipy.stats import binned_statistic_2d

import datetime

pio.renderers.default='browser'

class TrajectoryData:
    
    '''
    data; trxyt formated data;
        data.tr is the trajectory id
        data.x is localization in um
        data.y is localization in um
        data.t is the frame id
    
    countDF; dataframe containing trajectory length information (note, filtered)
        countDF.tr is the trajectory id
        countDF.Count is the trajectory length
        
    centroidDF; dataframe containing trajectory centroids
        centroidDF.tr is the trajectory id
        centroidDF.x is the localization in um
        centroidDF.y is the localization in um
    
    startendDF; dataframe containing first and last localizations of a trajectory
        startendDF.tr is the trajectory id
        startendDF.x is localization in um
        startendDF.y is localization in um
        startendDF.t is the frame id
    
    angleDF; dataframe containing angular information for the angular direction for each segment in trajectory
        angleDF.tr is the trajectory id
        angleDF.Angle is the associated angle
        angleDF.Count is the length of given trajectory
        angleDF.Angle_Adjusted is the adjusted angle around the eigCentre
    
    centroidHull; shapely.Polygon storing the convex hull of the centroid data
    
    eigCenter; the angle for the covariance matrix of the centroid data
        
    '''
    
    def __init__(self):
        self.data = pd.DataFrame()
        
    ### WRANGLING METHODS ###
    def import_trc(self, fname, pixel_size):
        trcDF = pd.read_csv(fname, delimiter='\t', skiprows=2)
        self.data['tr'] = trcDF['Track']
        self.data['x'] = trcDF['CentroidX(px)'] * pixel_size
        self.data['y'] = trcDF['CentroidY(px)'] * pixel_size
        self.data['t'] =  trcDF['Plane']
    
    def import_trxyframe(self, fname):       
        self.data = pd.read_csv(fname, delimiter=' ', header=None)
        self.data.columns = ['tr', 'x', 'y', 't']
    
    def import_trxytime(self, fname, DT):
        self.data = pd.read_csv(fname, delimiter=' ', header=None)
        self.data.columns = ['tr', 'x', 'y', 't']
        self.data['t'] = np.round(self.data['t'] / DT).astype('int')
        
    # def import_trackmatexml(self, fname):
    #     print('importing xml data... (this may take time)')
    #     df = pytrackmate.trackmate_peak_import(fname, get_tracks=True)
    #     df = df.sort_values(by=['label', 't_stamp'])
    #     self.data['tr'] = df['label'].astype('int')
    #     self.data['x'] = df['x']
    #     self.data['y'] = df['y']
    #     self.data['t'] = df['t_stamp'].astype('int')
        
    #     self.extra_data = df

    def import_straight(self, trj_vec, x_vec_um, y_vec_um, t_vec):
        self.data['tr'] = trj_vec
        self.data['x'] = x_vec_um
        self.data['y'] = y_vec_um
        self.data['t'] =  t_vec
    
    def import_trackmatecsv(self, fname):
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

    ### FILTERING METHODS ###
        
    def filter_by_count(self, min_length, max_length):
        
        if max_length != None:
        
            self.countDF = self.data.groupby('tr', as_index=False)['t'].count().rename(columns={'t': 'Count'})
            self.countDF = self.countDF[self.countDF['Count'] >= min_length]
            self.countDF = self.countDF[self.countDF['Count'] <= max_length]
            self.data = self.data[self.data['tr'].isin(self.countDF['tr'])]
        
        else:
            
            self.countDF = self.data.groupby('tr', as_index=False)['t'].count().rename(columns={'t': 'Count'})
            self.countDF = self.countDF[self.countDF['Count'] >= min_length]
            self.data = self.data[self.data['tr'].isin(self.countDF['tr'])]
        
    def filter_by_tr(self, tr_id):
        ''' tr_id: list of tr_id to extract'''
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
        
        helperDF = self.data.groupby('tr').mean().reset_index()
        tr_id = helperDF.query('x > @xmin and x < @xmax and y > @ymin and y < @ymax').tr
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
        
    ### EXPORTING METHODS ###
    
    def export_trxytime(self, fname, DT, round=False):
        with open(fname, 'a', newline='') as file:
            file.truncate(0)
            export_data = copy.deepcopy(self.data)
            export_data['t'] = export_data['t'] * DT
            
            if round:
                export_data['t'] = export_data['t'].round(3)
            
            export_data.to_csv(file, index=False, sep=' ', header=None)
            
    def export_saSPT(self, fname):
        with open(fname, 'a', newline='') as file:
            file.truncate(0)
            export_data = copy.deepcopy(self.data)
            export_data = pd.DataFrame(data={'x': self.data['x'],
                                             'y': self.data['y'],
                                             'trajectory': self.data['tr'],
                                             'frame': self.data['t']})
            export_data.to_csv(file, index=False, sep=' ', header=True)
    
    def export_trxyframe(self, fname):
        with open(fname, 'a', newline='') as file:
            file.truncate(0)
            self.data.to_csv(file, index=False, sep=' ', header=None)
    
    ### CREATION METHODS ###
    
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
    
    def create_centroidHull(self, concave = False, alpha = None):
        
        def alpha_shape(points, alpha):
            """
            Compute the alpha shape (concave hull) of a set of points.

            @param points: Iterable container of points.
            @param alpha: alpha value to influence the gooeyness of the border. Smaller
                          numbers don't fall inward as much as larger numbers. Too large,
                          and you lose everything!
            
            From: https://gist.github.com/dwyerk/10561690
            
            """
            if len(points.geoms) < 4:
                # When you have a triangle, there is no sense in computing an alpha
                # shape.
                return MultiPoint(list(points)).convex_hull

            def add_edge(edges, edge_points, coords, i, j):
                """Add a line between the i-th and j-th points, if not in the list already"""
                if (i, j) in edges or (j, i) in edges:
                    # already added
                    return
                edges.add( (i, j) )
                edge_points.append(coords[ [i, j] ])

            coords = np.array([[point.x, point.y] for point in points.geoms])

            tri = Delaunay(coords)
            edges = set()
            edge_points = []
            # loop over triangles:
            # ia, ib, ic = indices of corner points of the triangle
            for ia, ib, ic in tri.simplices:
                pa = coords[ia]
                pb = coords[ib]
                pc = coords[ic]

                # Lengths of sides of triangle
                a = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
                b = math.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
                c = math.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)

                # Semiperimeter of triangle
                s = (a + b + c)/2.0

                # Area of triangle by Heron's formula
                area = math.sqrt(s*(s-a)*(s-b)*(s-c))
                circum_r = a*b*c/(4.0*area)

                # Here's the radius filter.
                #print circum_r
                if circum_r < 1.0/alpha:
                    add_edge(edges, edge_points, coords, ia, ib)
                    add_edge(edges, edge_points, coords, ib, ic)
                    add_edge(edges, edge_points, coords, ic, ia)

            m = MultiLineString(edge_points)
            triangles = list(polygonize(m))
            return unary_union(triangles)
        
        # obtain Points class
        centroidPoints = MultiPoint(self.centroidDF[['x', 'y']].values)
        
        # calculate the convexhull 
        if concave == False:
            self.centroidHull = centroidPoints.convex_hull
        
        elif concave == True and alpha != None:
            self.centroidHull = alpha_shape(centroidPoints, alpha)
        
        else: raise Exception('alpha not specified for concave hull')
            
    
    def create_angleDF(self, startend = False):
        trj_dict = {}
        
        if startend == False:
            for trj in self.data['tr'].unique():
                trj_dict[trj] = self.data[self.data['tr'] == trj][['x', 'y']].values
        
        else:
            for trj in self.data['tr'].unique():
                trj_dict[trj] = self.startendDF[self.startendDF['tr'] == trj][['x', 'y']].values
        
        angles = [self.calc_angle_of_direction(trj_dict[trj]) for trj in trj_dict]
        idx = trj_dict.keys()

        self.angleDF = pd.merge(pd.DataFrame({'tr': idx, 'Angle': angles}),
                                self.countDF, left_on='tr', right_on='tr', how='left') \
            .explode('Angle', ignore_index=True)
            
    def create_stepwiseangleDF(self, startend = False):
        
        trj_dict = {}
        
        if startend == False:
            for trj in self.data['tr'].unique():
                trj_dict[trj] = self.data[self.data['tr'] == trj][['x', 'y']].values
        
        else:
            for trj in self.data['tr'].unique():
                trj_dict[trj] = self.startendDF[self.startendDF['tr'] == trj][['x', 'y']].values
                
        angles = [self.calc_stepwise_angle(trj_dict[trj]) for trj in trj_dict]
        idx = trj_dict.keys()

        self.angleswDF = pd.merge(pd.DataFrame({'tr': idx, 'Angle_r': angles}), self.countDF, left_on='tr', right_on='tr', how='left').explode('Angle_r', ignore_index=True)
        self.angleswDF['Angle_deg'] = self.angleswDF.Angle_r.apply(math.degrees)
        
    def create_stepwiseangleDFFrequency(self, bins):
        
        if not hasattr(self, 'angleswDF'):
            self.create_stepwiseangleDF()
            
        bin_vals, bin_edges = np.histogram(self.angleswDF['Angle_deg'], bins=bins, range=(0, 180))
        bin_vals = bin_vals / len(self.angleswDF['Angle_deg'])
        bin_centers = 0.5*(bin_edges[1:]+bin_edges[:-1])
        
        self.angleswfreqDF = pd.DataFrame({'angle_value': bin_centers,
                                           'angle_density': bin_vals})
        
    def create_stepwiseangleskipDF(self, skip=2):
        '''
        Skip 2 means, it takes every second step, starting from step 0

        Parameters
        ----------
        skip : TYPE, optional
            DESCRIPTION. The default is 2.

        Returns
        -------
        None.

        '''
        
        trj_dict = {}
        
        for trj in self.data['tr'].unique():
            trj_dict[trj] = self.data[self.data['tr'] == trj][['x', 'y']].values[::skip]
                
        angles = [self.calc_stepwise_angle(trj_dict[trj]) for trj in trj_dict]
        idx = trj_dict.keys()

        self.angleswskipDF = pd.merge(pd.DataFrame({'tr': idx, 'Angle_r': angles, 'skip': [skip] * len(idx)}), self.countDF, left_on='tr', right_on='tr', how='left').explode('Angle_r', ignore_index=True)
        self.angleswskipDF['Angle_deg'] = self.angleswskipDF.Angle_r.apply(math.degrees)
        
    def create_stepwiseangle_distanceDF(self):
        
        trj_dict = {}

        for trj in self.data['tr'].unique():
            trj_dict[trj] = self.data[self.data['tr'] == trj][['x', 'y']].values

        angles = [self.calc_stepwise_angle(trj_dict[trj]) for trj in trj_dict]
        distances = [self.calc_stepwise_distance(trj_dict[trj]) for trj in trj_dict]
        idx = trj_dict.keys()

        self.anglesw_distanceDF = pd.merge(pd.DataFrame({'tr': idx, 'Angle_r': angles, 'Distance': distances}), self.countDF, left_on='tr', right_on='tr', how='left').explode(['Angle_r', 'Distance'], ignore_index=True)
        self.anglesw_distanceDF['Angle_deg'] = self.anglesw_distanceDF.Angle_r.apply(math.degrees)

    def create_roi_towards_away_inside_angle(self, roi, buffer=False):
        '''
        Given a ROI, determine whether a displacement in a trajectory is moving
        towards, away or is inside (i.e., both points are inside) the roi.
        
        The angular output is the intuitive angle, where 180 deg is a 180 deg turn,
        and 0 deg is continueing on a straight line

        Returns
        -------
        None.

        '''
        # create a data frame, for each displacement decide if inside, towards or away a roi
        # if one point is in and one point is out its considered towards or away
        # if both points are in its considered inside
        
        # loop through each trajectory to create a dictionary of trajectories
        trj_dict = {}
        
        for trj in self.data['tr'].unique():
            trj_dict[trj] = self.data[self.data['tr'] == trj][['x', 'y']].values
        
        # from the trajectory dictionary calculate the angle between each consecutive step
        angles = [self.calc_stepwise_angle(trj_dict[trj]) for trj in trj_dict]
        
        # from the trajectory dictionary determine which way trajectory is moving (Roi Towards Away Inside) (JUST BEFORE THAT ANGLE)
        output_list = [self.calc_roi_towards_away_inside(trj_dict[trj][:-1], roi)for trj in trj_dict]
        
        # point aray (AT THAT ANGLE)
        point_array_x = [trj_dict[trj][1:-1][:,0] for trj in trj_dict]
        point_array_y = [trj_dict[trj][1:-1][:,1] for trj in trj_dict]

        self.RTAIangle = pd.merge(pd.DataFrame({'tr': trj_dict.keys(), 'Angle_r': angles, 'roi_towards_away': output_list, 'x': point_array_x, 'y': point_array_y}),
                                  self.countDF,
                                  left_on='tr',
                                  right_on='tr',
                                  how='left').explode(['Angle_r', 'roi_towards_away', 'x', 'y'], ignore_index=True)
        self.RTAIangle['Angle_deg'] = 180 - self.RTAIangle.Angle_r.apply(math.degrees)
        
        
        # determine if a roi contains the points
        if buffer != False:
            roi_buffed = shapelybuffer(roi, buffer)
            
            self.RTAIangle['in_buffer'] = np.array(list(map(lambda point: roi_buffed.contains(Point(point)), np.array(self.RTAIangle[['x', 'y']])))) * -2 + 1
            
        # TOWARDS ROI = -ve = TRUE
        # AWAY ROI= +ve = FALSE


    def calc_roi_towards_away_inside(self, trj, roi):
        '''
        usage for create_roi_towards_away_inside_angle

        Parameters
        ----------
        trj : TYPE
            DESCRIPTION.
        roi : TYPE
            DESCRIPTION.

        Returns
        -------
        roi_towards_away : TYPE
            DESCRIPTION.
        consecutive_in_array : TYPE
            DESCRIPTION.

        '''
        
        # distance from point to roi boundary (always positive)
        distance_array = np.array(list(map(lambda point: shapelydistance(roi.boundary, Point(point)), trj)))
        
        # determine if point inside boundary (returns array of -1 or 1; -1 is inside, 1 is outside)
        in_array = np.array(list(map(lambda point: roi.contains(Point(point)), trj))) * -2 + 1
        
        # determine if two points in a row are inside the roi
        # 2 = two consecutive OUT
        # 0 = going crossing the ROI
        # -2 = two consecutive IN
        consecutive_in_array = (in_array[:-1] + in_array[1:])
        
        # disstance from point to roi boundary (-ve values are inside)
        # now determine direction RTAI
        # GETTING CLOSER TO BOUNDARY = -ve = TRUE
        # GETTING FURTHER AWAY FROM BOUNDARY = +ve = FALSE
        roi_towards_away = np.diff(distance_array) <= 0
        
        output_list = [self.sorter_function(consec, towards, inar) for consec, towards, inar in zip(consecutive_in_array, roi_towards_away, in_array[:-1])]
            
        return output_list

    def sorter_function(self, consec, towards, inar):
        if consec == 2:
            return towards # those points outside of buffer get filtered out later when querying buffer for plot
        if consec == -2:
            return None # this gets filtered out later when plotting and calculations
        if consec == 0 and inar == -1:
            return False
        if consec == 0 and inar == 1:
            return True

    def create_buffer_end_in_or_end_out(self, roi, buffer):
        '''
        Given a ROI, find all trajectories that initially start in a buffer region outside the ROI.
        Then find the percentage of trajectories that end inside the ROI, in the buffer region outside of the ROI, and outside the buffer
        region of the ROI.

        Parameters
        ----------
        roi : TYPE
            DESCRIPTION.
        buffer : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        
        # loop through each trajectory to create a dictionary of trajectories
        trj_dict = {}
        
        for trj in self.data['tr'].unique():
            trj_dict[trj] = self.data[self.data['tr'] == trj][['x', 'y']].values
            
        # list of trajectories whose initial point is in the buffer region
        roi_buffed = shapelybuffer(roi, buffer)
        roi_of_buffer = shapelydifference(roi_buffed, roi)
        
        fpoint_in = [roi_of_buffer.contains(Point(trj_dict[key][0])) for key in trj_dict.keys()]
        fpoint_in = np.array(list(trj_dict.keys()))[fpoint_in]
        
        o_dict = {}
        
        for tr_id in fpoint_in:
            if roi.contains(Point(trj_dict[tr_id][-1])):
                o_dict[tr_id] = 'central'
                
            elif roi_of_buffer.contains(Point(trj_dict[tr_id][-1])):
                o_dict[tr_id] = 'buffer'
                
            elif not roi_of_buffer.contains(Point(trj_dict[tr_id][-1])):
                o_dict[tr_id] = 'peripheral'
            
            else:
                o_dict[tr_id] = 'error'
                
        return o_dict

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
    
    
    def create_displacement_vector(self):
        ''' for each trajectory, calculate the displacement between each step and return dictionary'''
        
        self.displacement_vector_dict = {}
        
        for i in self.data['tr'].unique():
            
            disp_vec = [] # create empty vector
            
            trj = self.data[self.data['tr'] == i]
            trj = np.array([trj.x, trj.y]).T.reshape(-1, 1, 2)
                        
            disp_vec = [np.sqrt((n0[0]-n1[0])**2 + (n0[1]-n1[1])**2) for n0, n1 in zip(trj[:-1, 0], trj[1:, 0])]
            self.displacement_vector_dict[i] = disp_vec
            
    def create_displacementinfo_vector(self, roi, ax=None, **kwargs):
        '''
        For each trajectory, calculate the displacement between each step, and return dictionary,
        additionally returns disctionary regarding whether both points of the step are within a ROI;
        True; at least on point was in the ROI
        False; neither points were in the ROI

        Parameters
        ----------
        roi : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        
        self.displacement_roivector_dict = {}
        self.displacement_roivectorinfo_dict = {}
        
        cols = kwargs.pop('cols', {True: 'Red', False: 'Black'})
        
        if ax != None:
            ax.set_aspect('equal')
            mean_x = self.data.x.mean()
            mean_y = self.data.y.mean()
        
        for i in self.data['tr'].unique():
            
            disp_vec = [] # create empty vector
            
            trj = self.data[self.data['tr'] == i]
            trj = np.array([trj.x, trj.y]).T.reshape(-1, 1, 2)
                        
            disp_vec = [np.sqrt((n0[0]-n1[0])**2 + (n0[1]-n1[1])**2) for n0, n1 in zip(trj[:-1, 0], trj[1:, 0])]
            disp_info = [roi.contains(Point(n0)) or roi.contains(Point(n1)) for n0, n1 in zip(trj[:-1, 0], trj[1:, 0])]
            
            if ax != None:
                for n0, n1, cc in zip(trj[:-1, 0], trj[1:, 0], disp_info):
                    ax.plot([n0[0]-mean_x, n1[0]-mean_x], [n0[1]-mean_y, n1[1]-mean_y], color=cols[cc], **kwargs)
                    
            self.displacement_roivector_dict[i] = disp_vec
            self.displacement_roivectorinfo_dict[i] = disp_info
            
            
    ### PLOTTING METHODS ###
    
    def plot_trajectories(self, ax, sample_num=None, color_by_time=False, color_by_random=False, DT=None, re_centre=False, save_cbar=None, fix_frames=None, **kwargs):
        
        '''
        function: plots trajectories using matplolib background

        arguments:
            ax: axis to plot onto
            sample_num: sample n of the trajectories from the data...
            color_by_time: if wanting to color by the time
        
        '''

        cmap = kwargs.pop('cmap', 'viridis')
        cbar = kwargs.pop('cbar', True)
        
        cmap = plt.get_cmap(cmap)
            

        if sample_num == None:
            data_to_plot = self.data
        else:
            data_to_plot = self.data[self.data['tr'].isin(np.random.choice(self.data['tr'].unique(), sample_num))]
        
        if re_centre == True:
            data_to_plot['x'] = data_to_plot['x'] - data_to_plot.x.mean()
            data_to_plot['y'] = data_to_plot['y'] - data_to_plot.y.mean()
        
        if color_by_time:
            
            # create a cmap
            
            if DT == None and fix_frames == None:
                norm = mplcolors.Normalize(vmin=data_to_plot['t'].min(),
                                           vmax=data_to_plot['t'].max(),
                                           clip=False)
                maximum_time = data_to_plot['t'].max()
                
            elif fix_frames != None:
                
                norm = mplcolors.Normalize(vmin=fix_frames[0],
                                           vmax=fix_frames[1],
                                           clip=False)
                
                maximum_time = fix_frames[1]
                
            else:
                norm = mplcolors.Normalize(vmin=data_to_plot['t'].min()*DT,
                                           vmax=data_to_plot['t'].max()*DT,
                                           clip=False)
            
            # plot each trajectory, color by fraction within cmap
            for i in data_to_plot['tr'].unique():
                
                # getting data
                df = data_to_plot[data_to_plot['tr'] == i]
                time_fraction = df['t'].min()/maximum_time
                
                # plotting
                ax.plot(df['x'],
                        df['y'],
                        color=cmap(time_fraction),
                        **kwargs)
            
            if cbar == True:
                
                cfig = plt.figure(figsize=(0.2, 0.5))
                cax = cfig.add_axes([0.3, 0.05, 0.3, 0.9])
                cbar = cax.figure.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
                tick_locator = ticker.MaxNLocator(nbins=4)
                cbar.locator = tick_locator
                cbar.update_ticks()
                cbar.set_label('Time (frame)')
                
                if save_cbar != None:
                    cfig.savefig(save_cbar)
                    
        # plot with random colors for trajectories
        elif color_by_random:
            
            cmap_len = cmap.N
            
            for i in data_to_plot['tr'].unique():
                df = data_to_plot[data_to_plot['tr'] == i]
                ax.plot(df['x'],
                        df['y'],
                        color=cmap(np.random.randint(0, cmap_len)),
                        **kwargs)
        
        # plot with all trajectories of the same color
        else:
                
            for i in data_to_plot['tr'].unique():
                df = data_to_plot[data_to_plot['tr'] == i]
                ax.plot(df['x'],
                        df['y'],
                        **kwargs)
    
        ax.set_aspect('equal')
        

    def plot_trajectories_matrix(self, ax, sample_num=None, color_by_time=False, color_by_random=False, DT=None, sp_axis='x', **kwargs):
        
        '''
        function: plots trajectories using matplolib background

        arguments:
            ax: axis to plot onto
            sample_num: sample n of the trajectories from the data...
            color_by_time: if wanting to color by the time
        
        '''

        xlim = kwargs.get('xlim', None)
        ylim = kwargs.get('ylim', None)
        cmap = kwargs.get('cmap', 'viridis')
        color = 'black'
        linewidth = kwargs.get('linewidth', 1)
        alpha = kwargs.get('alpha', 1)

        if sample_num == None:
            data_to_plot = self.data
        else:
            data_to_plot = self.data[self.data['tr'].isin(np.random.choice(self.data['tr'].unique(), sample_num))]
        
        if color_by_time:
            
            # create a cmap
            cmap = plt.get_cmap(cmap)
            
            if DT == None:
                norm = mplcolors.Normalize(vmin=data_to_plot['t'].min(),
                                           vmax=data_to_plot['t'].max(),
                                           clip=False)
            else:
                norm = mplcolors.Normalize(vmin=data_to_plot['t'].min()*DT,
                                           vmax=data_to_plot['t'].max()*DT,
                                           clip=False)
            
            # plot each trajectory, color by fraction within cmap
            for i in data_to_plot['tr'].unique():
                
                # getting data
                df = data_to_plot[data_to_plot['tr'] == i]
                time_fraction = df['t'].min()/data_to_plot['t'].max()
                
                # plotting
                ax.plot(df[sp_axis],
                        df['t'],
                        color=cmap(time_fraction),
                        linewidth=linewidth,
                        alpha=alpha)
            
            cbar = ax.figure.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap))
            cbar.set_label('Time (Frame)')
        
        # plot with random colors for trajectories
        elif color_by_random:
            
            cmap = plt.get_cmap(cmap)
            cmap_len = cmap.N
            
            for i in data_to_plot['tr'].unique():
                df = data_to_plot[data_to_plot['tr'] == i]
                ax.plot(df[sp_axis],
                        df['t'],
                        color=cmap(np.random.randint(0, cmap_len)),
                        linewidth=linewidth,
                        alpha=alpha)
        
        # plot with all trajectories of the same color
        else:
                
            for i in data_to_plot['tr'].unique():
                df = data_to_plot[data_to_plot['tr'] == i]
                ax.plot(df[sp_axis],
                        df['t'],
                        color=color,
                        linewidth=linewidth,
                        alpha=alpha)
        
    def plot_trajectories_diff_v2(self, ax, **kwargs):
        """
        function: plots trajectories using matplolib background ~ color by diffusion coefficient

        arguments:
            ax: axis to plot onto
            
        """
        
        palette = kwargs.pop('palette', 'flare')
        cmap = kwargs.pop('cmap', None)
        cbar = kwargs.pop('cbar', True)
        vmin = kwargs.pop('vmin', -5)
        vmax = kwargs.pop('vmax', 1)
        
        # get data
        data_to_plot = copy.deepcopy(self.data)
        
        # colors
        if cmap == None:
            cmap = plt.get_cmap(palette)
        
        norm = mplcolors.Normalize(vmin=vmin, vmax=vmax, clip=False)
        
        # sorting
        sorted_difflog_dict = {k: v for k, v in sorted(self.difflog_dict.items(), key=lambda item: item[1], reverse=True)}
        
        # plotting
        for tr_id, dcoef in sorted_difflog_dict.items():
            df = data_to_plot[data_to_plot['tr'] == tr_id]
            ax.plot(df['x'], df['y'], color=cmap(norm(dcoef)), **kwargs)
        
        if cbar:
            cbar = ax.figure.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap))

        ax.set_aspect('equal')

    def plot_trajectory(self, ax, tr_id, **kwargs):
        """
        function: plots trajectories using matplolib background

        arguments:
            ax: axis to plot onto
            tr_id: id's of the trajectory to plot (list)
        
        """

        # xlim = kwargs.get('xlim', None)
        # ylim = kwargs.get('ylim', None)
        # palette = kwargs.get('palette', 'flare')
        # color = kwargs.get('color', 'black')
        
        data_to_plot = self.data[self.data['tr'].isin(tr_id)]
        
        sns.lineplot(x='x', y='y', units='tr', ax=ax, data=data_to_plot, sort=False, estimator=None,
                     **kwargs)        

        # if xlim and ylim != None:
        #     ax.set_ylim(ylim)
        #     ax.set_xlim(xlim)
        
        # ax.get_legend().remove()
        ax.set_aspect('equal')
    
    def plot_trajectory_color_time(self, ax, tr_id, DT, **kwargs):
        
        palette = kwargs.pop('palette', 'mako')
        cmap = kwargs.pop('cmap', None)
        cbar = kwargs.pop('cbar', True)
        save_cbar = kwargs.pop('save_cbar', None)
        
        data_to_plot = copy.deepcopy(self.data[self.data['tr'] == tr_id])
        data_to_plot['t'] = data_to_plot['t'] * DT
        data_to_plot['t'] = data_to_plot['t'] - np.array(data_to_plot['t'])[0]
        
        if cmap == None:
            cmap = plt.get_cmap(palette)
            
        p1 = np.array([data_to_plot.x, data_to_plot.y]).T.reshape(-1, 1, 2)
        seg1 = np.concatenate([p1[:-1], p1[1:]], axis = 1)
        col1 = [cmap(float(i)/(len(data_to_plot)-1)) for i in range(len(data_to_plot)-1)]
        
        for ii in range(len(data_to_plot)-1):
            segii = seg1[ii]
            lii, = ax.plot(segii[:,0], segii[:,1], color = col1[ii], **kwargs)
            lii.set_solid_capstyle('round')

        ax.set_aspect('equal', 'box')
        
        if cbar:
            
            cfig = plt.figure(figsize=(0.2, 0.5))
            cax = cfig.add_axes([0.3, 0.05, 0.3, 0.9])
            norm = mplcolors.Normalize(vmin=min(data_to_plot['t']), vmax=max(data_to_plot['t']), clip=False)
            cbar = cax.figure.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
            tick_locator = ticker.MaxNLocator(nbins=4)
            cbar.locator = tick_locator
            cbar.update_ticks()
            cbar.set_label('Time (s)')
            
            if save_cbar != None:
                cfig.savefig(save_cbar)
    
    def plot_trajectory_color_time_3d(self, ax, tr_id, DT, **kwargs):
        
        palette = kwargs.get('palette', 'viridis')
        show_c_bar = kwargs.get('show_c_bar', True)
        linewidth = kwargs.get('linewidth', 0.3)
        
        data_to_plot = copy.deepcopy(self.data[self.data['tr'] == tr_id])
        data_to_plot['t'] = data_to_plot['t'] * DT
        data_to_plot['t'] = data_to_plot['t'] - np.array(data_to_plot['t'])[0]
        
        cmap1 = plt.get_cmap(palette)
        p1 = np.array([data_to_plot.x, data_to_plot.y, data_to_plot.t]).T.reshape(-1, 1, 3)
        seg1 = np.concatenate([p1[:-1], p1[1:]], axis = 1)
        col1 = [cmap1(float(i)/(len(data_to_plot)-1)) for i in range(len(data_to_plot)-1)]
        
        for ii in range(len(data_to_plot)-1):
            segii = seg1[ii]
            lii, = ax.plot(segii[:,0], segii[:,2], segii[:,1], color = col1[ii], linewidth = linewidth)
            lii.set_solid_capstyle('round')
        
        if show_c_bar:
            norm = mplcolors.Normalize(vmin=min(data_to_plot['t']), vmax=max(data_to_plot['t']), clip=False)
            cbar = ax.figure.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap1))
            cbar.set_label('Time (s)')
    
    def plot_trajectories_px(self):
        fig = px.line(x='x', y='y', color='tr', data_frame=self.data)
        fig.update_yaxes(scaleanchor = "x", scaleratio = 1)
        fig.update_traces(marker=dict(size=5,
                                      line=dict(width=0,
                                                color='DarkSlateGrey'),
                                      symbol='square'),
                          selector=dict(mode='markers'))
        fig.show()
    
    def plot_3dtrajectories_px(self):
        fig = px.line_3d(x='x', y = 'y', z='t', color='tr', line_group='tr', data_frame=self.data)
        # fig.update_yaxes(
        #     scaleanchor = "x",
        #     scaleratio = 1)
        fig.update_traces(marker=dict(size=5,
                                      line=dict(width=0,
                                                color='DarkSlateGrey'),
                                      symbol='square'),
                          selector=dict(mode='markers'))
        fig.show()

    
    def plot_scatter(self, ax, centroids = False, **kwargs):
        xlim = kwargs.get('xlim', None)
        ylim = kwargs.get('ylim', None)
        alpha = kwargs.get('alpha', 0.3)
        s = kwargs.get('s', 1)
        
        if centroids == False:
            sns.scatterplot(x='x', y='y', ax=ax, data=self.data, **kwargs)
        else:
            sns.scatterplot(x='x', y='y', ax=ax, data=self.centroidDF, **kwargs)
        
        if xlim and ylim != None:
            ax.set_ylim(ylim)
            ax.set_xlim(xlim)
        
        ax.set_aspect('equal')
    
    def plot_scatter_kde(self, ax, **kwargs):
        
        print('estimating kde...')
        xy = np.vstack([self.data['x'], self.data['y']])
        gaus = gaussian_kde(xy)(xy)
        print('kde estimation complete...')
        
        ax.scatter(self.data['x'], self.data['y'], c=gaus, **kwargs)
        ax.set_aspect('equal')
    
    def plot_kde(self, ax, cbar=True, **kwargs):
        
        if cbar==True:
            figbar = plt.figure(figsize=(0.2, 0.5))
            axbar = figbar.add_axes([0.3, 0.05, 0.3, 0.9])
        
            sns.kdeplot(data=self.data, x="x", y="y", ax=ax, cbar=cbar, cbar_kws={'label': 'Probability density estimate', 'format': '%.2f'}, cbar_ax=axbar, **kwargs)
            ax.set_aspect('equal')
            
            print('saving cbar_density')
            figbar.savefig('cbar_density.svg')
        
        else:
            sns.kdeplot(data=self.data, x="x", y="y", ax=ax, cbar=cbar, cbar_kws={'label': 'Probability density estimate', 'format': '%.2f'}, **kwargs)
            ax.set_aspect('equal')
    
    def plot_2dhist(self, ax, cbar=True, binsize=0.1, **kwargs):
        
        bins = [np.arange(self.data.x.min(), self.data.x.max(),binsize), np.arange(self.data.y.min(), self.data.y.max(),binsize)]
        
        if cbar==True:
            figbar = plt.figure(figsize=(0.2, 0.5))
            axbar = figbar.add_axes([0.3, 0.05, 0.3, 0.9])
        
            sns.histplot(data=self.data, x="x", y="y", ax=ax, bins=bins, cbar=cbar, cbar_kws={'label': 'Detections', 'format': '%.2f'}, cbar_ax=axbar, **kwargs)
            ax.set_aspect('equal')
            
            print('saving cbar_density')
            figbar.savefig('cbar_density.svg')
        
        else:
            sns.histplot(data=self.data, x="x", y="y", ax=ax, bins=bins, cbar=cbar, cbar_kws={'label': 'Detections', 'format': '%.2f'}, **kwargs)
            ax.set_aspect('equal')
    
    def plot_2dhist_scatter(self, ax, cbar=True, binsize=1, **kwargs):
        
        x = self.data['x'].values
        y = self.data['y'].values
        
        x_edges = np.arange(x.min(), x.max() + binsize, binsize)
        y_edges = np.arange(y.min(), y.max() + binsize, binsize)
        
        counts, x_edges, y_edges, binnumber = binned_statistic_2d(
            x, y, None, statistic='count', bins=[x_edges, y_edges]
        )
        
        x_bin = np.clip(np.digitize(x, x_edges) - 1, 0, counts.shape[0] - 1)
        y_bin = np.clip(np.digitize(y, y_edges) - 1, 0, counts.shape[1] - 1)
        point_density = counts[x_bin, y_bin]
        
        # sort ascending by density so highest-density points are drawn last (on top)
        order = np.argsort(point_density)
        x_sorted = x[order]
        y_sorted = y[order]
        density_sorted = point_density[order]

        sc = ax.scatter(x_sorted, y_sorted, c=density_sorted, **kwargs)
        ax.set_aspect('equal')
        
        if cbar == True:
            figbar = plt.figure(figsize=(0.2, 0.5))
            axbar = figbar.add_axes([0.3, 0.05, 0.3, 0.9])
            cbar = plt.colorbar(sc,cax=axbar)
            cbar.set_label('Density')
            figbar.savefig('cbar.svg') 
        
    def plot_centroids_kde(self, ax, bw_method, **kwargs):
        
        print('estimating kde...')
        xy = np.vstack([self.centroidDF['x'], self.centroidDF['y']])
        gaus = gaussian_kde(xy, bw_method=bw_method)(xy)
        print('kde estimation complete...')
        
        ax.scatter(self.centroidDF['x'], self.centroidDF['y'], c=gaus, **kwargs)
        ax.set_aspect('equal')
            
    def plot_countfreq(self, ax, **kwargs):
        '''
        reqires; countDF
        function; plots frequency distribution for trajectory lengths
        
        
        Parameters
        ----------
        ax : matplotlib.axes
            axes to plot countfreq data onto.
            
        bins : number of bins

        Returns
        -------
        None.

        '''
        
        bins = kwargs.get('bins', None)
        
        if bins == None:
            ax.hist(self.countDF['Count'], bins=max(self.countDF['Count']))
        else:
            ax.hist(self.countDF['Count'], bins=bins)

        ax.text(0, 0, 'mean {}, med {}'.format(self.countDF['Count'].mean(),
                                               self.countDF['Count'].median()))
        ax.set_box_aspect(1)
        ax.set_xlabel('Count')
        ax.set_ylabel('Freq')
    
    def plot_centroidHull(self, ax):
        ax.plot(*self.centroidHull.exterior.xy, color='grey', alpha=0.4)
    
    def plot_msds(self, ax, DT = None, **kwargs):

        '''
        function; plots the msds for all trajectories within self.data (note, msds_list must have been calculated prior)
        ax; specify axis to plot on
        DT; if None, then plots with no time information, if specified float, then will these times...

        '''
        
        if DT == None:
        
            for i in self.msd_list:
                x = np.arange(0, len(i))
                ax.plot(x[1:], i[1:], **kwargs) # note, first index contains 0
                ax.set_xlabel('Time (frame)')  
        
        else:
            for i in self.msd_list:
                x = np.arange(0, len(i)) * DT
                ax.plot(x[1:], i[1:], **kwargs) # note, first index contains 0
                ax.set_xlabel('Time (s)')  
              
        ax.set_box_aspect(1)
        ax.set_ylabel(r'MSD ($\mu$m$^2$)')
        
    def plot_mean_msds(self, ax, plot_x_limit, DT, **kwargs):

        '''
        function; plots the mean msds for all trajectories within self.data (note, msds_list must have been calculated prior)
        ax; specify axis to plot on
        plot_x_limit; specify maximum t value to plot (in frames)
        DT; delta T for acquisition
        
        Note; degrees of freedom <= 0 for slice indicates that only one value is piped into 'sem'
        this occurs for the longest trajectory; please ignore...

        '''
        
        color = kwargs.get('color', 'black')
        label = kwargs.get('label', 'none')
        linestyle = kwargs.get('linestyle', 'solid')
        
        # length of longest list (or trajectory)
        max_length = np.max([len(i) for i in self.msd_list])

        # create vector to store mean and sem values
        mean_values = np.zeros(max_length)
        sem_values = np.zeros(max_length)

        # calculate for all... 
        for i in range(0, max_length):
            vector = []
            for msd_vals in self.msd_list:
                if i < len(msd_vals):
                    vector.append(msd_vals[i])
                
            mean_values[i] = np.mean(vector)
            sem_values[i] = sem(vector)
        
        ax.errorbar(np.arange(0, max_length)[1:plot_x_limit] * DT, mean_values[1:plot_x_limit], yerr=sem_values[1:plot_x_limit], color=color,
                    label=label, linestyle=linestyle, marker='o', markeredgecolor='white')
        ax.set_box_aspect(1)
        ax.set_xlabel('Time (sec)')
        ax.set_ylabel(r'MSD ($\mu$m$^2$)')

    
    # def calc_mean_msd(self, ax, DT):
    
    #     # length of longest list (or trajectory)
    #     max_length = np.max([len(i) for i in self.msd_list])

    #     # create vector to store mean and sem values
    #     mean_values = np.zeros(max_length)
    #     sem_values = np.zeros(max_length)

    #     # calculate for all... 
    #     for i in range(0, max_length):
    #         vector = []
    #         for msd_vals in self.msd_list:
    #             if i < len(msd_vals):
    #                 vector.append(msd_vals[i])
                
    #         mean_values[i] = np.mean(vector)
    #         sem_values[i] = sem(vector)
            
    #     return mean_values, sem_values
    
    
    def plot_surv_prob(self, ax, **kwargs):
        '''
        Plot survival probability information.

        Parameters
        ----------
        ax : matplotlib.pyplot.axis
            Axis to plot onto.

        Returns
        -------
        None.

        '''
        
        # sns.scatterplot(data=self.surv_probDF, x='trj_len_s', y='cumsum_perc',
        #                 ax=ax, **kwargs)
        
        ax.step(data=self.surv_probDF, x='trj_len_s', y='cumsum_perc', **kwargs)
        
        
        ax.set(xscale='log')
        ax.set(yscale='log')
        ax.set_ylabel('Survival probability')
        ax.set_xlabel('Time (s)')
        

    
    ### SIMULATED TRAJECTORY DATA ###
    
    def simulate_particles_2D(self, num_particles, steps, D, DT):
        
        trj_dict = {}
        
        for trajID in range(num_particles):
            
            x = np.sqrt(2 * D * DT) * np.random.normal(size=steps)
            y = np.sqrt(2 * D * DT) * np.random.normal(size=steps)
            
            tr_vec = (np.ones(steps) * trajID).astype(int)
            x_vec = (np.cumsum(x))
            y_vec = (np.cumsum(y))
            t_vec = (np.arange(0, steps))
            
            trj_dict[trajID] = pd.DataFrame({'tr': tr_vec,
                                             'x': x_vec,
                                             'y': y_vec,
                                             't': t_vec})
        
        self.data = pd.concat(trj_dict.values(), ignore_index=True)
        
    def simulate_particle_confined(self, D, tau_sim, steps, hull, num_traj, **kwargs):
        
        dict_store = {}
        
        x0 = kwargs.get('x0', None)
        y0 = kwargs.get('y0', None)
        
        # calculate sigma
        s = np.sqrt(2 * D * tau_sim)
        
        for tr_id in range(num_traj):
            
            # specified initial location of trajectory
            if x0 != None and y0 != None:
                x = x0
                y = y0
            
            # otherwise choose random point in hull
            else:
                
                # find boundary of hull
                minx, miny, maxx, maxy = hull.bounds
                
                # wrap until find a start location within hull
                x = random.uniform(minx, maxx)
                y = random.uniform(miny, maxy)
                loc = Point((x, y))
                while not hull.contains(loc):
                   x = random.uniform(minx, maxx)
                   y = random.uniform(miny, maxy)
                   loc = Point((x, y))
            
            # vectors for storage
            x_data = np.zeros(steps)
            y_data = np.zeros(steps)
            t_data = np.arange(0, steps)
    
            # for each step
            for i in range(steps):
                
                # brownian motion (create temporary location for check)
                x_temp = x + np.random.randn() * s
                y_temp = y + np.random.randn() * s
                
                # simulate until point lays within hull
                while not hull.contains(Point(x_temp, y_temp)):
                    
                    x_temp = x + np.random.randn() * s
                    y_temp = y + np.random.randn() * s
                
                # store as new coordinates
                x = x_temp
                y = y_temp
                
                # store to vector
                x_data[i] = x
                y_data[i] = y
        
            dict_store[tr_id] = pd.DataFrame({'tr': [tr_id] * steps,
                                              'x': x_data,
                                              'y': y_data,
                                              't': t_data})
            
            print('particle {} complete'.format(tr_id))
        
        self.data = pd.concat(dict_store)
        self.data = self.data.reset_index(drop=True)          
            

    def visualized_particle_displacements(self, tau_sim, tau_viz):
        
        # assertions
        assert tau_viz >= tau_sim
        assert (tau_viz/tau_sim).is_integer()
        
        # find difference between simulation and visualization
        dtau = int(tau_viz/tau_sim)
        
        # take every nth element
        tr_viz = self.data.tr.values[0::dtau]
        x_viz = self.data.x.values[0::dtau]
        y_viz = self.data.y.values[0::dtau]
        t_viz = self.data.t.values[0::dtau]
        
        # create new instance of TrajectoryData
        viz_data = TrajectoryData()
        viz_data.import_straight(tr_viz, x_viz, y_viz, t_viz)
        
        return viz_data
    
    ### MSD ###
    
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
        
        
    # ### Diffusion Coefficient ###
    
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
        
    ### Stepwise Angular Motion ##
    
    def calc_stepwise_angle(self, trj):
            
        angle_list = []
        
        for i in range(len(trj)-2):
            a = trj[i]
            b = trj[i+1]
            c = trj[i+2]
            
            ba = a - b
            bc = c - b
            
            norm_ba = np.linalg.norm(ba)
            norm_bc = np.linalg.norm(bc)
            
            if norm_ba == 0 or norm_bc == 0:
                angle_list.append(np.nan)
            
            else:
                cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
                cosine_angle = np.clip(cosine_angle, -1.0, 1.0) # clamping to remove runtime errors
                angle = np.arccos(cosine_angle)
                angle_list.append(angle)
        
        return angle_list

    def calc_stepwise_distance(self, trj):
        
        dist_list = []

        for i in range(len(trj)-2):
            b = trj[i+1]
            c = trj[i+2]
            
            dist_list.append(math.dist(b, c))
        
        return dist_list
        
    ### Directed Angular Motion ###
        
    def calc_angle_of_direction(self, trj):

        angle_list = []
        
        for n0, n1 in zip(trj[:-1], trj[1:]):
            vec = n1 - n0
            base = np.array([1, 0])
            theta = np.arccos(np.dot(vec, base) / (np.linalg.norm(vec) * np.linalg.norm(base)))
            
            if vec[1] < 0:
                theta = 2 * np.pi - theta
            
            angle_list.append(theta)
        
        return angle_list
    
    def eigvecfunc(self, m, c, x):
        return m * x + c

    def calc_eigCentre(self, sanity_check = False, ax = None):
        
        # calculate covariance matrix of centroid data
        cov_matrix = np.cov(self.centroidDF.iloc[:, 1:].values, rowvar=False)
        
        # calculate eigen vectors
        w, v = np.linalg.eig(cov_matrix) 

        # select eigen vector, with largest w
        vec = v[:, np.argmax(w)]
        base = np.array([1, 0])

        # if vec[0] > 0:
        #     self.eigCentre = np.arccos((vec[1])/np.sqrt((vec[0])**2 + (vec[1])**2))
        # else:
        #     self.eigCentre = -np.arccos((vec[1])/np.sqrt((vec[0])**2 + (vec[1])**2))
        
        
        self.eigCentre = np.arccos(np.dot(vec, base) / (np.linalg.norm(vec) * np.linalg.norm(base)))
        
        if vec[1] < 0:
            self.eigCentre = 2 * np.pi - self.eigCentre
            self.eigCentre = self.eigCentre % np.pi
        
        if sanity_check == True and ax != None: # plotting
            
            m = vec[1] / vec[0] # rise over run
            x = np.linspace(self.centroidDF['x'].min() - np.mean(self.centroidDF['x']),
                            self.centroidDF['x'].max() - np.mean(self.centroidDF['x']),
                            100)
            
            y = self.eigvecfunc(m, 0, x)
            
            ax.scatter(self.centroidDF['x'] - np.mean(self.centroidDF['x']),
                       self.centroidDF['y'] - np.mean(self.centroidDF['y']), color='black', alpha=0.5)
            ax.plot(x, y)
            ax.text(0, 0, 'eigCentre: ' + str(self.eigCentre))
            ax.set_aspect('equal')
        
    def adjust_angles_around_eigCentre(self, override_angle=None):
        '''
        requires;
            angleDF and eigCentre
        function;
            adds Angle_Adjusted to angleDF ~ in which angleDF.Angle is rotated
            to align with eigCentre
            
        override_angle; 
            value between 0 and pi
        '''
        
        angles = self.angleDF['Angle']

        # shift angles by eigCentre
        if override_angle == None:
            angles = angles - self.eigCentre
        else:
            assert override_angle <= np.pi and override_angle >= 0
            angles = angles - override_angle
            
        angles = angles % (2 * np.pi) # shift angles back to (0 ~ 2 * np.pi)
        
        self.angleDF['Angle_Adjusted'] = angles
    
    def calculate_angle_relative_freq_histogram(self, num_bins, angle_adjusted):
        '''
        requires;
            angleDF
        function;
            calculates relative frequency histogram for angular data

        Parameters
        ----------
        num_bins : int
            the number of bins for binning angular data
        angle_adjusted : boolean
            whether or not to use angle_adjusted data

        '''
        if angle_adjusted == True:
            bin_vals, bin_borders = np.histogram(self.angleDF['Angle_Adjusted'], bins=num_bins, range=(0, 2*np.pi))
        else:
            bin_vals, bin_borders = np.histogram(self.angleDF['Angle'], bins=num_bins, range=(0, 2*np.pi))
        
        # determine bin centres and relative frequency
        bin_centres = bin_centres = bin_borders[:-1] + np.diff(bin_borders)/2
        bin_vals = bin_vals/sum(bin_vals)
        
        return bin_vals, bin_borders, bin_centres

    def rose_plot(self, angle_adjusted, ax, num_bins=20):
        
        '''
        requires;
            angleDF
        function;
            calculates relative frequency histogram for angular data, plots as a rose plot

        Parameters
        ----------
        ax : matplotlib.axis
            axis to plot data
        num_bins : int
            the number of bins for binning angular data
        angle_adjusted : boolean
            whether or not to use angle_adjusted data

        '''
        
        bin_vals, bin_borders, bin_centres = self.calculate_angle_relative_freq_histogram(num_bins, angle_adjusted)
        
        # color map based on angle
        cmap = cmocean.cm.phase
        cvec = [cmap(i) for i in (bin_centres)/(2 * np.pi)]
        
        # compute width of each bin
        widths = np.diff(bin_borders)
        
        # compute radius of each bin
        radius = (bin_vals / np.pi)**.5

        # Plot data on ax
        ax.bar(bin_borders[:-1], radius, zorder=10, align='edge', width=widths,
               edgecolor='black', fill=True, linewidth=1, color=cvec)

        # Set the direction of the zero angle
        ax.grid(linewidth=0.8, linestyle=':', color='black', alpha=0.5)

        ax.set_yticks([])
        ax.set_frame_on(True)
        ax.set_xticks(np.linspace(0, 2*np.pi, 13)[:-1])
    
    ### view using napari ###
    
    # def napari_view_trajectories_series(self, series_fname, pixel_size):
        
    #     series = io.imread(series_fname)
        
    #     # format data
    #     track_df = pd.DataFrame({'track_id': self.data['tr'],
    #                              't': self.data['t'],
    #                              'y': self.data['y'] / pixel_size,
    #                              'x': self.data['x'] / pixel_size})
        
    #     track_df.sort_values(by=['track_id', 't'])
    #     track_list = track_df.values.tolist()
    #     point_list = track_df.values[:, 1:]

    #     viewer = napari.view_image(series)
    #     viewer.add_tracks(track_list, name='tracks')
    #     viewer.add_points(point_list, name='points', size=1)








#### DEPRICATED ####

# def plot_trajectories_diff(self, ax, startend=False, sample_num=None, **kwargs):
#     """
#     function: plots trajectories using matplolib background ~ color by difflist_log

#     arguments:
#         ax: axis to plot onto
#         startend: True to plot direct line from the start of trajectory to end of trajectory
#         sample_num: sample n of the trajectories from the data...
    
#     """

#     xlim = kwargs.get('xlim', None)
#     ylim = kwargs.get('ylim', None)
#     palette = kwargs.get('palette', 'flare')
    
#     data_to_plot = copy.deepcopy(self.data)
    
#     # adding diff information to data
#     counts = data_to_plot.groupby('tr', as_index=False)['t'].count().rename(columns={'t': 'Count'})
    
#     stretched_difflist = []
    
#     for i in range(len(counts)):
#         stretched_difflist.extend([self.difflog_list[i]] * counts['Count'][i])
    
#     data_to_plot['logdiff'] = stretched_difflist
    
#     cmap = plt.get_cmap(palette)
    
#     sns.lineplot(x='x', y='y', hue='logdiff', units='tr', ax=ax, data=data_to_plot, linewidth=0.3, sort=False, palette=cmap,
#                  estimator=None, alpha=0.3, hue_norm=(mplcolors.Normalize(vmin=-5, vmax=0, clip=False))) 
    
#     cbar = ax.figure.colorbar(cm.ScalarMappable(norm=mplcolors.Normalize(vmin=-5,
#                                                                          vmax=0,
#                                                                          clip=False),
#                                                 cmap=cmap))

#     if xlim and ylim != None:
#         ax.set_ylim(ylim)
#         ax.set_xlim(xlim)
    
#     ax.get_legend().remove()
#     ax.set_aspect('equal')