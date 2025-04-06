from pynwb import NWBHDF5IO
import matplotlib.pyplot as plt
import numpy as np
from pynwb.icephys import CurrentClampSeries, VoltageClampSeries

# Load your NWB file
nwb_path = "D:\\Neuro_Sci\\morph_ephys_trans_stuff\\fine_and_dandi_ephys\\000020\\sub-599387254\\sub-599387254_ses-601506492_icephys.nwb"

with NWBHDF5IO(nwb_path, 'r') as io:
    nwbfile = io.read()

    stims = nwbfile.stimulus
    stim_names = []
    for i in stims:
        print('stim', i)
        stim_names.append(i)
    print(stim_names)
    
    response = nwbfile.acquisition
    response_names = []
    for i in response:
        print('res', i)
        response_names.append(i)
    print(stim_names)
    
    for index in range(len(stim_names)): 
        this_stim = nwbfile.get_stimulus(stim_names[index])
        this_response = nwbfile.get_acquisition(response_names[index])
        print(this_stim.data[()])
        plt.plot(this_stim.data[()])
        plt.plot(this_response.data[()])
        plt.show()