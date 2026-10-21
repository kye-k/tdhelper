# -*- coding: utf-8 -*-

from shapely.geometry import Point
import numpy as np
import pandas as pd
import random


class _SimulationMixin:

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