import numpy as np
import pandas as pd
from simulator import simulator
from ads_regulator import ADS

from joblib import Parallel, delayed
import multiprocessing


def do_exp(w):
    N_sim=1
    l = 10
    avg_ut  = np.zeros(N_sim)
    acc     = np.zeros(N_sim)
    fat_in  = np.zeros(N_sim)
    fat_out = np.zeros(N_sim)
    inj_in  = np.zeros(N_sim)
    inj_out = np.zeros(N_sim)
    trip_duration = np.zeros(N_sim)
    skids = np.zeros(N_sim)
    crashes = np.zeros(N_sim)
    
    ##
    for i in range(N_sim):
        sim = simulator(l)
        env = sim.simulate_environment()
        ads =  ADS(env, accident_prob, inj_fat_in, inj_fat_out, w)
        ads.complete_road()

        avg_ut[i]  = ads.utilities.mean()
        acc[i]     = ads.accidents
        fat_in[i]  = ads.fatalities_in
        fat_out[i] = ads.fatalities_out
        inj_in[i]  = ads.injuries_in
        inj_out[i] = ads.injuries_out
        trip_duration[i] = l/ads.speed_sel.mean()
        crashes[i] = ads.crashes
        skids[i] = ads.skids
        
    df = pd.DataFrame({"avg_ut": avg_ut,
             "n_accidents": acc,
             "n_fat_in": fat_in,
             "n_fat_out": fat_out,
             "n_inj_in" : inj_in,
             "n_inj_out": inj_out,
             "trip_duration": trip_duration,
             "crashes" : crashes,
             "skids" : skids})

    name = "results/reg_exp_wcomf_" + str(np.round(w[0], 2) ) + ".csv"
    df.to_csv(name, index=False)
    

if __name__ == "__main__":

    accident_prob =   pd.read_csv("data/accident_prob", index_col=0, delim_whitespace=True)
    inj_fat_out   =   pd.read_csv("data/inj_fat_prob_out", index_col=0, delim_whitespace=True)
    inj_fat_in    =   pd.read_csv("data/inj_fat_prob_in", index_col=0, delim_whitespace=True)

    inj_fat_in  = [x.drop(columns="type").values for _, x in inj_fat_in.groupby(inj_fat_in['type']) ]
    inj_fat_out = [x.drop(columns="type").values for _, x in inj_fat_out.groupby(inj_fat_out['type']) ]

    w = np.arange(0.1, 1.0, 0.1)
    w = np.insert(w, 9, 0.99)
    w = np.insert(w, 0, 0.01)

    grid = []
    for i in w:
        grid.append([np.round(i, 2), np.round(1-i, 2)])
    

    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(delayed(do_exp)(i) for i in grid)


