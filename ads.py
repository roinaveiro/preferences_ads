import numpy as np
import pandas as pd
from scipy.stats import multinomial


class ADS:
    """
    Class to simulate road
    Args:
        l (int): road lenght
    """

    def __init__( self, env, accident_prob, inj_fat_in,
    inj_fat_out, weights, n_passengers=4 ):

        # Road details
        self.road    = env["road"]
        self.people  = env["people"]

        self.n_passengers = n_passengers

        self.inj_fat_in = inj_fat_in.values
        self.inj_fat_out = inj_fat_out.values
        
        self.accident_prob     = accident_prob
        self.N                 = len(self.road)

        self.ut_params         = {"ut_speed" : [-1, 0.5, 1, 2],
                                  "ut_accident" : -6,
                                  "ut_injury"   : -30,
                                  "ut_fatality" : -100,
                                  "w_comfort"   : weights[0],
                                  "w_sec_in"    : weights[1],
                                  "w_sec_out"   : weights[2]
                                  }

        self.speed_decisions = accident_prob.columns.values.astype('int')
        self.lane_decisions = np.array([0,1])

        # To store decisions and utilities
        self.speed_sel = np.zeros(self.N)
        self.lane_sel  = np.zeros(self.N)
        self.utilities = np.zeros(self.N)

        # Pre-compute expected utilities
        self.eut_inj_in = []
        self.eut_fat_in = []

        for i in self.speed_decisions:
            a,b = self.compute_fat_inj_eut_in(i)
            self.eut_inj_in.append(a)
            self.eut_fat_in.append(b)


        # Init ads
        self.current_cell = 0 ## Current cell
        


        # Counters
        self.accidents = 0
        self.injuries_out = 0
        self.injuries_in = 0
        self.fatalities_out = 0
        self.fatalities_in = 0


        ##
    def move(self):

        # Make decisions
        cell_speed, cell_lane = self.decide()
        self.speed_sel[self.current_cell] = cell_speed
        self.lane_sel[self.current_cell] = cell_lane

        # Evaluate utilities
        self.utilities[self.current_cell] = self.compute_cell_utility(cell_speed, cell_lane)


        # Move 
        self.current_cell += 1

    def decide(self):

        cut = -1000
        for i in self.speed_decisions:

            for j in self.lane_decisions:
                cell_ut = self.compute_cell_exp_utility(i, j)
                if cell_ut > cut:
                    final_speed = i
                    final_lane  = j
                    cut = cell_ut

        return final_speed, final_lane



    def compute_cell_exp_utility(self, speed, lane):

        p_acc = self.compute_accident_probability(speed, lane)
        
        comfort_eut = self.ut_params["ut_speed"][speed] + p_acc*self.ut_params["ut_accident"]
        sec_in_eut  = p_acc*( self.eut_fat_in[speed] + self.eut_inj_in[speed] )

        a, b = self.compute_fat_inj_eut_out(speed, lane)
        sec_out_eut = p_acc*(a+b)
        
        eut = (self.ut_params["w_comfort"]*comfort_eut +
               self.ut_params["w_sec_in"]*sec_in_eut   +
               self.ut_params["w_sec_out"]*sec_out_eut)

        return eut


    def compute_accident_probability(self, speed, lane):

        content = self.road[self.current_cell, lane]
        people  = self.people[self.current_cell, lane]

        indicator_people = 3 if people >= 2 else 4 if people>0 else 5

        return ( self.accident_prob.iloc[content][speed] + 
                self.accident_prob.iloc[indicator_people][speed] )

    def compute_fat_inj_eut_out(self, speed, lane):

        n_ped = self.people[self.current_cell, lane]

        if n_ped == 0:
            return 0.0, 0.0 
        else:
            inj_fat = multinomial(n_ped, 
                self.inj_fat_out[:, speed])
            
            eut_fat = 0.0
            eut_inj = 0.0
            for i in range(n_ped + 1):
                ## i fatalities i injuries
                pr_fat = 0
                pr_inj = 0

                for j in range(n_ped + 1 - i):
                    pr_fat += inj_fat.pmf([n_ped - (i+j), j, i])
                    pr_inj += inj_fat.pmf([n_ped - (i+j), i, j])

                eut_fat += pr_fat * (i * self.ut_params["ut_fatality"])
                eut_inj += pr_inj * (i * self.ut_params["ut_injury"])
        
        return eut_inj, eut_fat

    def compute_fat_inj_eut_in(self, speed):

        n_ped = self.n_passengers
        inj_fat = multinomial(n_ped, 
            self.inj_fat_in[:, speed])
        
        eut_fat = 0.0
        eut_inj = 0.0
        for i in range(n_ped + 1):
            ## i fatalities i injuries
            pr_fat = 0
            pr_inj = 0

            for j in range(n_ped + 1 - i):
                pr_fat += inj_fat.pmf([n_ped - (i+j), j, i])
                pr_inj += inj_fat.pmf([n_ped - (i+j), i, j])

            eut_fat += pr_fat * (i * self.ut_params["ut_fatality"])
            eut_inj += pr_inj * (i * self.ut_params["ut_injury"])
        
        return eut_inj, eut_fat


    def compute_cell_utility(self, speed, lane):

        p_acc = self.compute_accident_probability(speed, lane)
        accident = np.random.choice([0,1], p=[1-p_acc, p_acc])

        comfort_ut = self.ut_params["ut_speed"][speed] + accident*self.ut_params["ut_accident"]

        if accident == 1:
            self.accidents += 1

            out_events = self.sample_fat_inj_out(speed, lane)
            sec_out_ut = ( out_events[1] * self.ut_params["ut_injury"] +
                            out_events[2] * self.ut_params["ut_fatality"])

            in_events  = self.sample_fat_inj_in(speed)
            sec_in_ut = ( in_events[1] * self.ut_params["ut_injury"] +
                            in_events[2] * self.ut_params["ut_fatality"])

            # Update counters
            self.injuries_out   += out_events[1]
            self.injuries_in    += in_events[1]
            self.fatalities_out += out_events[2]
            self.fatalities_in  += in_events[2]

        else:
            sec_out_ut = 0.0
            sec_in_ut  = 0.0

        ut =  (self.ut_params["w_comfort"]*comfort_ut +
               self.ut_params["w_sec_in"]*sec_in_ut   +
               self.ut_params["w_sec_out"]*sec_out_ut)

        return ut

    def sample_fat_inj_out(self, speed, lane):

        n_ped = self.people[self.current_cell, lane]
        if n_ped == 0:
            return np.array([0, 0, 0])
        else:
            inj_fat = multinomial(n_ped, 
                self.inj_fat_out[:, speed])
            
            return inj_fat.rvs()[0]
    
    def sample_fat_inj_in(self, speed):

        n_ped = self.n_passengers
        inj_fat = multinomial(n_ped, 
            self.inj_fat_in[:, speed])
        
        return inj_fat.rvs()[0]

        
    

    def complete_road(self):
        for i in range(self.N - 1 ):
            self.move()

    def get_info(self):
        pass

        

    @staticmethod
    def normalize(arr):
        return arr / np.sum(arr)

    @staticmethod
    def normalize_arr(arr):
        return arr / np.sum(arr, axis=0)




