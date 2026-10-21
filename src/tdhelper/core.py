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
        
    '''

    def __init__(self):
        self.data = pd.DataFrame()
