# -*- coding: utf-8 -*-

import math
import numpy as np
import pandas as pd


class _AngleMixin:

    def create_angleDF(self, startend = False):
        '''
        creates a dataframe (angleDF) that contains the angular DIRECTION for each segment in a trajectory;
        if startend == True, then the angleDF reports the angulare DIRECTION for the whole trajectory;
        
        '''
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

    def create_stepwiseangleDF(self):
        '''
        creates a dataframe (angleswDF) that contains the stepwise change in angle for each triplet of points in a trajectory;
        in the current implementation, 180 degrees is the "forward, straight motion" direction, and 0 degrees is the "reversing, backward motion" direction;
        this should be changed in later implementations to be more intuitive, but for now until publication, this is the convention;
        
        '''
        
        trj_dict = {}

        for trj in self.data['tr'].unique():
            trj_dict[trj] = self.data[self.data['tr'] == trj][['x', 'y']].values

        angles = [self.calc_stepwise_angle(trj_dict[trj]) for trj in trj_dict]
        idx = trj_dict.keys()

        self.angleswDF = pd.merge(pd.DataFrame({'tr': idx, 'Angle_r': angles}), self.countDF, left_on='tr', right_on='tr', how='left').explode('Angle_r', ignore_index=True)
        self.angleswDF['Angle_deg'] = self.angleswDF.Angle_r.apply(math.degrees)

    def create_stepwiseangleDFFrequency(self, bins):
        '''
        creates a dataframe (angleswfreqDF) that contains the relative frequency histogram for the stepwise change in angle;

        '''
        
        if not hasattr(self, 'angleswDF'):
            self.create_stepwiseangleDF()
            
        bin_vals, bin_edges = np.histogram(self.angleswDF['Angle_deg'], bins=bins, range=(0, 180))
        bin_vals = bin_vals / len(self.angleswDF['Angle_deg'])
        bin_centers = 0.5*(bin_edges[1:]+bin_edges[:-1])
        
        self.angleswfreqDF = pd.DataFrame({'angle_value': bin_centers,
                                           'angle_density': bin_vals})

    def create_stepwiseangleskipDF(self, skip=2):
        '''
        creates a dataframe (angleswskipDF) that contains the stepwise change in angle for each triplet of points in a trajectory, skipping every n points;
        skip = 2 means, it takes every second step and calculates the skipwise change in angle;

        '''
        
        trj_dict = {}
        
        for trj in self.data['tr'].unique():
            trj_dict[trj] = self.data[self.data['tr'] == trj][['x', 'y']].values[::skip]
                
        angles = [self.calc_stepwise_angle(trj_dict[trj]) for trj in trj_dict]
        idx = trj_dict.keys()

        self.angleswskipDF = pd.merge(pd.DataFrame({'tr': idx, 'Angle_r': angles, 'skip': [skip] * len(idx)}), self.countDF, left_on='tr', right_on='tr', how='left').explode('Angle_r', ignore_index=True)
        self.angleswskipDF['Angle_deg'] = self.angleswskipDF.Angle_r.apply(math.degrees)

    def create_stepwiseangle_distanceDF(self):
        '''
        creates a dataframe (anglesw_distanceDF) that contains the stepwise change in angle and the stepwise displacement (initial step) for each triplet of points in a trajectory;
        basically an addition to create_stepwiseangleDF

        '''
        
        trj_dict = {}

        for trj in self.data['tr'].unique():
            trj_dict[trj] = self.data[self.data['tr'] == trj][['x', 'y']].values

        angles = [self.calc_stepwise_angle(trj_dict[trj]) for trj in trj_dict]
        distances = [self.calc_stepwise_distance(trj_dict[trj]) for trj in trj_dict]
        idx = trj_dict.keys()

        self.anglesw_distanceDF = pd.merge(pd.DataFrame({'tr': idx, 'Angle_r': angles, 'Distance': distances}), self.countDF, left_on='tr', right_on='tr', how='left').explode(['Angle_r', 'Distance'], ignore_index=True)
        self.anglesw_distanceDF['Angle_deg'] = self.anglesw_distanceDF.Angle_r.apply(math.degrees)

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

    ### DEPRICATED FUNCTIONS BELOW;

    # def eigvecfunc(self, m, c, x):
    #     return m * x + c

    # def calc_eigCentre(self, sanity_check = False, ax = None):
        
    #     # calculate covariance matrix of centroid data
    #     cov_matrix = np.cov(self.centroidDF.iloc[:, 1:].values, rowvar=False)
        
    #     # calculate eigen vectors
    #     w, v = np.linalg.eig(cov_matrix) 

    #     # select eigen vector, with largest w
    #     vec = v[:, np.argmax(w)]
    #     base = np.array([1, 0])

    #     # if vec[0] > 0:
    #     #     self.eigCentre = np.arccos((vec[1])/np.sqrt((vec[0])**2 + (vec[1])**2))
    #     # else:
    #     #     self.eigCentre = -np.arccos((vec[1])/np.sqrt((vec[0])**2 + (vec[1])**2))
        
        
    #     self.eigCentre = np.arccos(np.dot(vec, base) / (np.linalg.norm(vec) * np.linalg.norm(base)))
        
    #     if vec[1] < 0:
    #         self.eigCentre = 2 * np.pi - self.eigCentre
    #         self.eigCentre = self.eigCentre % np.pi
        
    #     if sanity_check == True and ax != None: # plotting
            
    #         m = vec[1] / vec[0] # rise over run
    #         x = np.linspace(self.centroidDF['x'].min() - np.mean(self.centroidDF['x']),
    #                         self.centroidDF['x'].max() - np.mean(self.centroidDF['x']),
    #                         100)
            
    #         y = self.eigvecfunc(m, 0, x)
            
    #         ax.scatter(self.centroidDF['x'] - np.mean(self.centroidDF['x']),
    #                    self.centroidDF['y'] - np.mean(self.centroidDF['y']), color='black', alpha=0.5)
    #         ax.plot(x, y)
    #         ax.text(0, 0, 'eigCentre: ' + str(self.eigCentre))
    #         ax.set_aspect('equal')

    # def adjust_angles_around_eigCentre(self, override_angle=None):
    #     '''
    #     requires;
    #         angleDF and eigCentre
    #     function;
    #         adds Angle_Adjusted to angleDF ~ in which angleDF.Angle is rotated
    #         to align with eigCentre
            
    #     override_angle; 
    #         value between 0 and pi
    #     '''
        
    #     angles = self.angleDF['Angle']

    #     # shift angles by eigCentre
    #     if override_angle == None:
    #         angles = angles - self.eigCentre
    #     else:
    #         assert override_angle <= np.pi and override_angle >= 0
    #         angles = angles - override_angle
            
    #     angles = angles % (2 * np.pi) # shift angles back to (0 ~ 2 * np.pi)
        
    #     self.angleDF['Angle_Adjusted'] = angles

    # def calculate_angle_relative_freq_histogram(self, num_bins, angle_adjusted):
    #     '''
    #     requires;
    #         angleDF
    #     function;
    #         calculates relative frequency histogram for angular data

    #     Parameters
    #     ----------
    #     num_bins : int
    #         the number of bins for binning angular data
    #     angle_adjusted : boolean
    #         whether or not to use angle_adjusted data

    #     '''
    #     if angle_adjusted == True:
    #         bin_vals, bin_borders = np.histogram(self.angleDF['Angle_Adjusted'], bins=num_bins, range=(0, 2*np.pi))
    #     else:
    #         bin_vals, bin_borders = np.histogram(self.angleDF['Angle'], bins=num_bins, range=(0, 2*np.pi))
        
    #     # determine bin centres and relative frequency
    #     bin_centres = bin_centres = bin_borders[:-1] + np.diff(bin_borders)/2
    #     bin_vals = bin_vals/sum(bin_vals)
        
    #     return bin_vals, bin_borders, bin_centres

    # def rose_plot(self, angle_adjusted, ax, num_bins=20):
        
    #     '''
    #     requires;
    #         angleDF
    #     function;
    #         calculates relative frequency histogram for angular data, plots as a rose plot

    #     Parameters
    #     ----------
    #     ax : matplotlib.axis
    #         axis to plot data
    #     num_bins : int
    #         the number of bins for binning angular data
    #     angle_adjusted : boolean
    #         whether or not to use angle_adjusted data

    #     '''
        
    #     bin_vals, bin_borders, bin_centres = self.calculate_angle_relative_freq_histogram(num_bins, angle_adjusted)
        
    #     # color map based on angle
    #     cmap = cmocean.cm.phase
    #     cvec = [cmap(i) for i in (bin_centres)/(2 * np.pi)]
        
    #     # compute width of each bin
    #     widths = np.diff(bin_borders)
        
    #     # compute radius of each bin
    #     radius = (bin_vals / np.pi)**.5

    #     # Plot data on ax
    #     ax.bar(bin_borders[:-1], radius, zorder=10, align='edge', width=widths,
    #            edgecolor='black', fill=True, linewidth=1, color=cvec)

    #     # Set the direction of the zero angle
    #     ax.grid(linewidth=0.8, linestyle=':', color='black', alpha=0.5)

    #     ax.set_yticks([])
    #     ax.set_frame_on(True)
    #     ax.set_xticks(np.linspace(0, 2*np.pi, 13)[:-1])
