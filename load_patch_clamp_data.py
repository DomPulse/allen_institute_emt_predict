from pynwb import NWBHDF5IO
import matplotlib.pyplot as plt
import numpy as np
from pynwb.icephys import CurrentClampSeries, VoltageClampSeries
from math import log10, floor

#chatGPT is bad at the actual NWB processing but great at this
def round_to_sigfig(value, sigfig=1):
    if value == 0:
        return 0.0
    exponent = floor(log10(abs(value)))
    factor = 10 ** (exponent - sigfig + 1)
    return round(value / factor) * factor

SI_PREFIXES = {
    -24: 'y',   # yocto
    -21: 'z',   # zepto
    -18: 'a',   # atto
    -15: 'f',   # femto
    -12: 'p',   # pico
    -9:  'n',   # nano
    -6:  'µ',   # micro
    -3:  'm',   # milli
    -2:  'c',   # centi
    -1:  'd',   # deci
     0:  '',    # base unit
     1:  'da',  # deca
     2:  'h',   # hecto
     3:  'k',   # kilo
     6:  'M',   # mega
     9:  'G',   # giga
    12:  'T',   # tera
    15:  'P',   # peta
    18:  'E',   # exa
    21:  'Z',   # zetta
    24:  'Y',   # yotta
}

def get_si_prefix(value):
    if value == 0:
        return '', 0
    exponent = int(floor(log10(abs(value))))
    si_exponent = (exponent // 3) * 3
    prefix = SI_PREFIXES.get(si_exponent, f"e{si_exponent}")
    return prefix
#ok end of chatGPT code

# Load your NWB file
nwb_path = "D:\\Neuro_Sci\\morph_ephys_trans_stuff\\fine_and_dandi_ephys\\000020\\sub-599387254\\sub-599387254_ses-601506492_icephys.nwb"

with NWBHDF5IO(nwb_path, 'r') as io:
    nwbfile = io.read()

    stims = nwbfile.stimulus

    stim_names = []
    for i in stims:

        print('stim', i, str(type(stims[i])) == "<class 'pynwb.icephys.VoltageClampStimulusSeries'>")
        stim_names.append(i)
    #print(stim_names)
    
    response = nwbfile.acquisition
    response_names = []
    for i in response:
        print('res', i, type(response[i]))
        response_names.append(i)
    #print(stim_names)
    
    for index in range(len(stim_names)): 
        

        this_stim = nwbfile.get_stimulus(stim_names[index])
        this_response = nwbfile.get_acquisition(response_names[index])
        
        stim_units = this_stim.conversion
        response_units = this_response.unit
        
        print(f"Stimulus units {get_si_prefix(round_to_sigfig(this_stim.conversion))} {this_stim.unit}")
        print(f"Response units {get_si_prefix(round_to_sigfig(this_response.conversion))} {this_response.unit}")
        print(this_response.data[()])
        plt.plot(this_stim.data[()])
        plt.plot(this_response.data[()])
        plt.show()