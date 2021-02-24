import numpy as np
import pandas as pd
from simulator import simulator
from ads import ADS


def do_exp(w, N_sim):
    avg_ut  = np.zeros(N_sim)
    acc     = np.zeros(N_sim)
    fat_in  = np.zeros(N_sim)
    fat_out = np.zeros(N_sim)
    inj_in  = np.zeros(N_sim)
    inj_out = np.zeros(N_sim)
    ##
    for i in range(N_sim):
        sim = simulator(200)
        env = sim.simulate_environment()
        ads =  ADS(env, accident_prob, inj_fat_in, inj_fat_out, w)
        ads.complete_road()

        avg_ut[i]  = ads.utilities.mean()
        acc[i]     = ads.accidents
        fat_in[i]  = ads.fatalities_in
        fat_out[i] = ads.fatalities_out
        inj_in[i]  = ads.injuries_in
        inj_out[i] = ads.injuries_out
        
    df = pd.DataFrame({"avg_ut": avg_ut,
             "n_accidents": acc,
             "n_fat_in": fat_in,
             "n_fat_out": fat_out,
             "n_inj_in" : inj_in,
             "n_inj_out": inj_out})
    
    return df

if __name__ == "__main__":

    accident_prob =   pd.read_csv("data/accident_prob", index_col=0, delim_whitespace=True)
    inj_fat_out   =   pd.read_csv("data/inj_fat_prob_out", index_col=0, delim_whitespace=True)
    inj_fat_in    =   pd.read_csv("data/inj_fat_prob_in", index_col=0, delim_whitespace=True)

    for i in np.arange(0.0, 0.8, 0.1):
        w = [0.3, i, 1-(0.3+i)]
        df = do_exp(w, 500)
        name = "results/exp_wsecin_" + str(np.round(w[1], 1) ) + ".csv"
        df.to_csv(name, index=False)