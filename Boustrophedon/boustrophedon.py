import numpy as np
from bspline import bspline

class Boustrophedon:
    def __init__(self, d_waypoint_distance=2.5):
        self.full_path = None
        self.obs_wp = None
        self.name = "Boustrophedon"
        self.Boustrophedon(d_waypoint_distance)

    def Boustrophedon(self, d_waypoint_distance):
        cv = np.array([
                [ 0.5,  0.5],
                [ 0.5, 39.5],
                [ 5.,  39.5],
                [ 5.,  0.5],
                [ 10.,  0.5],
                [ 10., 39.5],
                [ 15., 39.5],
                [ 15.,  0.5],
                [ 20.,  0.5],
                [ 20., 39.5],
                [ 25., 39.5],
                [ 25.,  0.5],
                [ 30.,  0.5],
                [ 30., 39.5],
                [ 35., 39.5],
                [ 35.,  0.5],
                [ 39.5,  0.5],
                [ 39.5, 39.5]
                ])
        d = 2
        p = bspline(cv,n=1000,degree=d)
        p_way = [p[0]]
        for i in range(1, len(p)):
            if np.linalg.norm(p[i] - p_way[-1]) > d_waypoint_distance:
                p_way.append(p[i])
        p_way = np.array(p_way)
        self.full_path = p.T
        self.obs_wp = p_way




