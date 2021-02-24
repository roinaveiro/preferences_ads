import numpy as np
import pandas as pd


class simulator:
    """
    Class to simulate road
    Args:
        l (int): road lenght
    """

    def __init__(self, l, dynamics=None):
        self._l  = l
        self._mu = 2
        self._pi = 0.6
            
        ##
        if dynamics is None:
            self._road_dynamics = pd.read_csv("data/road_state_evol", index_col=0, delim_whitespace=True)
            ##
        else:
            self._road_dynamics = dynamics[0]
            ##
            self._driver_dynamics = dynamics[1]
            ##
            self._driver_char = dynamics[2]

    def simulate_road(self):

        road = np.empty(self._l, dtype=int)
        road[0] = 2

        for i in range(1, self._l):
            p = self._road_dynamics.loc[ road[i-1] ]
            road[i] = np.random.choice(self._road_dynamics.columns, p = p)

        return road

    def simulate_pedestrians(self):

        return np.array([np.random.poisson(self._mu)*(np.random.random()<self._pi) 
            for i in range(1, self._l)])

    def simulate_environment(self):

        road = np.array([self.simulate_road(),self.simulate_road()]).T
        people = np.array([self.simulate_pedestrians(),
                self.simulate_pedestrians()]).T

        return {"road"    : road,
                "people"  :  people}