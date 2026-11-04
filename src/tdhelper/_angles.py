# -*- coding: utf-8 -*-

import math
import numpy as np
import pandas as pd


class _AngleMixin:

    def calc_angleDIR(self, startend = False):
        '''
        Calculate the direction of movement for each segment in a trajectory (between 0 to 2pi).
        If startend == True, then calculate the direction from the first point to last point of the trajectory. 
        
        Parameters
        ----------
        startend: bool
            If True, calculate the direction from the first point to last point of the trajectory.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the direction of movement for each segment in a trajectory (radians).
        
        '''

        trj_dict = {}
        
        if startend == False:
            for trj in self.data['tr'].unique():
                trj_dict[trj] = self.data[self.data['tr'] == trj][['x', 'y']].values
        
        else:

            startendDF = self.create_startendDF()
            for trj in self.data['tr'].unique():
                trj_dict[trj] = startendDF[startendDF['tr'] == trj][['x', 'y']].values
        
        angles = [self._calc_angleDIR(trj_dict[trj]) for trj in trj_dict]
        idx = trj_dict.keys()
        angleDF = pd.DataFrame({'tr': idx, 'Angle_r': angles}).explode('Angle_r', ignore_index=True)
        
        return angleDF 

    def calc_angleSW(self, skip=None):
        '''
        Calculate the stepwise change in angle for each triplet of points in a trajectory.
        180 (pi); 180 degree change in direction, 0 (0); no change in direction, 90 (pi/2); 90 degree change in direction.

        Parameters
        ----------
        skip: int
            If skip = 2, then calculate the stepwise change in angle for every second point in a trajectory, and so forth.
            If skip = None, then calculate the stepwise change in angle for every point in a trajectory.
        
        Returns
        -------
        pandas.DataFrame
            DataFrame containing the stepwise change in angle for each triplet of points in a trajectory (radian and degrees).
        
        '''
        
        trj_dict = {}

        if skip == None:
            for trj in self.data['tr'].unique():
                trj_dict[trj] = self.data[self.data['tr'] == trj][['x', 'y']].values

        else:
            for trj in self.data['tr'].unique():
                trj_dict[trj] = self.data[self.data['tr'] == trj][['x', 'y']].values[::skip] ### NOTE MAKE SURE SKIP WORKS AS IT SHOULD!!!!!!

        angles = [self._calc_angleSW(trj_dict[trj]) for trj in trj_dict]
        idx = trj_dict.keys()

        angleDF = pd.DataFrame({'tr': idx, 'Angle_r': angles}).explode('Angle_r', ignore_index=True)
        angleDF['Angle_deg'] = angleDF.Angle_r.apply(math.degrees)

        return angleDF

    def calc_angleSW_DISPLACEMENTS(self):
        '''
        Calculate the stepwise change in angle and the displacement (p0 -> p1 in p0, p1, p2) for each triplet of points in a trajectory.
        In essence, a combination of calc_angleSW and calc_DISPLACEMENTS.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the stepwise change in angle and displacement for each triplet of points in a trajectory.

        '''
        
        trj_dict = {}

        for trj in self.data['tr'].unique():
            trj_dict[trj] = self.data[self.data['tr'] == trj][['x', 'y']].values

        angles = [self._calc_angleSW(trj_dict[trj]) for trj in trj_dict]
        distances = [self._calc_DISPLACEMENTS(trj_dict[trj]) for trj in trj_dict]
        idx = trj_dict.keys()

        DF = pd.merge(pd.DataFrame({'tr': idx, 'Angle_r': angles, 'Distance': distances}), self.countDF, left_on='tr', right_on='tr', how='left').explode(['Angle_r', 'Distance'], ignore_index=True)
        DF['Angle_deg'] = DF.Angle_r.apply(math.degrees)

        return DF

    def _calc_angleSW(self, trj):
            
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
                angle = np.arccos(-cosine_angle)
                angle_list.append(angle)
        
        return angle_list

    def _calc_DISPLACEMENTS(self, trj):
        
        dist_list = []

        for i in range(len(trj)-2):
            b = trj[i+1]
            c = trj[i+2]
            
            dist_list.append(math.dist(b, c))
        
        return dist_list

    def _calc_angleDIR(self, trj):

        angle_list = []
        
        for n0, n1 in zip(trj[:-1], trj[1:]):
            vec = n1 - n0
            base = np.array([1, 0])
            theta = np.arccos(np.dot(vec, base) / (np.linalg.norm(vec) * np.linalg.norm(base)))
            
            if vec[1] < 0:
                theta = 2 * np.pi - theta
            
            angle_list.append(theta)
        
        return angle_list
