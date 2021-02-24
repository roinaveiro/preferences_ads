import numpy as np
import pandas as pd
from simulator import simulator
from ads import ADS

accident_prob = pd.read_csv("data/accident_prob", index_col=0, delim_whitespace=True)
inj_fat_out   =   pd.read_csv("data/inj_fat_prob_out", index_col=0, delim_whitespace=True)
inj_fat_in    =   pd.read_csv("data/inj_fat_prob_in", index_col=0, delim_whitespace=True)

sim = simulator(1000)
env = sim.simulate_environment()

ads =  ADS(env, accident_prob, inj_fat_in, inj_fat_out)

print("done")
