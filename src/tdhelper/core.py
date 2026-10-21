# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 10:35:10 2023
Branched on 20261020 

@author: uqkkudo
"""

import pandas as pd

from ._io import _IOMixin
from ._filtering import _FilteringMixin
from ._tables import _TableMixin
from ._geometry import _GeometryMixin
from ._angles import _AngleMixin
from ._plotting import _PlottingMixin
from ._simulation import _SimulationMixin
from ._msd import _MSDMixin


class TrajectoryData(_IOMixin, _FilteringMixin, _TableMixin, _GeometryMixin, _AngleMixin, _PlottingMixin, _SimulationMixin, _MSDMixin):
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