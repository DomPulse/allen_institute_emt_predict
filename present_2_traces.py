import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

true_trace_path = r'F:\arbor_ubuntu\kimura_P7-control-for-TRAK2-MBD-14-003.CNG.swc_random_genome_19.csv'
true_trace_path = r'F:\arbor_ubuntu\MORE_MOUSE_BITES\kimura_P7-control-for-TRAK2-MBD-14-003.CNG.swc_11.csv'
best_cand_trace_path = r'F:\arbor_ubuntu\MORE_MOUSE_BITES\kimura_P7-control-for-TRAK2-MBD-14-003.CNG.swc_13.csv'

true_trace = pd.read_csv(true_trace_path)
best_cand_trace = pd.read_csv(best_cand_trace_path)

plt.plot(true_trace['t/ms'].to_numpy(), true_trace['U/mV'].to_numpy())
plt.plot(best_cand_trace['t/ms'].to_numpy(), best_cand_trace['U/mV'].to_numpy())
plt.show()