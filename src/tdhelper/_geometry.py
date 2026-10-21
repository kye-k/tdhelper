# -*- coding: utf-8 -*-

from scipy.spatial import Delaunay
from shapely import buffer as shapelybuffer
from shapely import difference as shapelydifference
from shapely import distance as shapelydistance
from shapely.geometry import MultiLineString
from shapely.geometry import MultiPoint
from shapely.geometry import Point
from shapely.ops import unary_union, polygonize
import math
import numpy as np
import pandas as pd


class _GeometryMixin:

    def create_centroidHull(self, concave = False, alpha = None):
        
        def alpha_shape(points, alpha):
            """
            Compute the alpha shape (concave hull) of a set of points.

            @param points: Iterable container of points.
            @param alpha: alpha value to influence the gooeyness of the border. Smaller
                          numbers don't fall inward as much as larger numbers. Too large,
                          and you lose everything!
            
            This snippet is from: https://gist.github.com/dwyerk/10561690
            
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

    def create_roi_towards_away_inside_angle(self, roi, buffer=False):
        '''
        given a ROI, determine whether a displacement in a trajectory is moving towards, away or is inside (i.e., both points are inside) the roi.
        roi; a solid polygon (no holes).
        buffer; boolean, whether to consider a buffer around the roi

        creates a dataframe (RTAIangle) that contains information about the angle of the displacement and whether it is moving towards, away or inside the roi
        angular output is the intuitive angle, where 180 deg is a 180 deg turn, and 0 deg is continueing on a straight line

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
        
        # determine if a roi contains the points (note, this applies to the whole buffered region, so if its inside the buffer its inside the original roi)
        if buffer != False:
            roi_buffed = shapelybuffer(roi, buffer)
            self.RTAIangle['in_buffer'] = np.array(list(map(lambda point: roi_buffed.contains(Point(point)), np.array(self.RTAIangle[['x', 'y']])))) * -2 + 1

    def calc_roi_towards_away_inside(self, trj, roi):
        '''
        usage for create_roi_towards_away_inside_angle

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
        given a ROI, find all trajectories that initially start in a buffer region outside the ROI.
        then find the percentage of trajectories that end inside the ROI, in the buffer region outside of the ROI, and outside the buffer region of the ROI.
        returns a dictionary o_dict

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

    def create_displacement_vector(self):
        '''
        for each trajectory, calculate the displacement between each step and return dictionary
        
        '''
        
        self.displacement_vector_dict = {}
        
        for i in self.data['tr'].unique():
            
            disp_vec = [] # create empty vector
            
            trj = self.data[self.data['tr'] == i]
            trj = np.array([trj.x, trj.y]).T.reshape(-1, 1, 2)
                        
            disp_vec = [np.sqrt((n0[0]-n1[0])**2 + (n0[1]-n1[1])**2) for n0, n1 in zip(trj[:-1, 0], trj[1:, 0])]
            self.displacement_vector_dict[i] = disp_vec

    def create_displacementinfo_vector(self, roi, ax=None, **kwargs):
        '''
        for each trajectory, calculate the displacement between each step, and return dictionary, additionally returns disctionary regarding whether both points of the step are within a ROI;
        True; at least on point was in the ROI
        False; neither points were in the ROI

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
