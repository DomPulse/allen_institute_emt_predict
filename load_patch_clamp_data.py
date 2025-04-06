from pynwb import NWBHDF5IO
import matplotlib.pyplot as plt
import numpy as np
from pynwb.icephys import CurrentClampSeries, VoltageClampSeries
from math import log10, floor
import os

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

def crop_on_voltage_flatline(voltage, current, time=None, zero_thresh=1e-3, min_flat_length=100):
    """
    Crops voltage, current, and optionally time arrays when voltage flattens to ~0.
    
    Args:
        voltage (np.ndarray): Voltage recording (e.g., in mV)
        current (np.ndarray): Current recording (e.g., in pA)
        time (np.ndarray or None): Optional time array
        zero_thresh (float): Threshold to consider "close to 0" (default: 1e-3)
        min_flat_length (int): Number of consecutive near-zero samples to trigger cropping

    Returns:
        Cropped (voltage, current, [time]) tuple
    """
    voltage = np.asarray(voltage)
    current = np.asarray(current)
    near_zero = np.abs(voltage) < zero_thresh

    # Find where the voltage becomes flat zero-like for a sustained period
    count = 0
    for i, val in enumerate(near_zero):
        if val:
            count += 1
            if count >= min_flat_length:
                cutoff = i - min_flat_length + 1
                break
        else:
            count = 0
    else:
        # No cutoff found, return original
        if time is not None:
            return voltage, current, time
        return voltage, current

    # Slice arrays
    if time is not None:
        return voltage[:cutoff], current[:cutoff], time[:cutoff]
    else:
        return voltage[:cutoff], current[:cutoff]


#ok end of chatGPT code

# Load your NWB file

def give_me_the_stuff(folder_path, files):
    stim_res_combo = []
    for nwb_path in files:
        with NWBHDF5IO(folder_path + nwb_path, 'r') as io:
            nwbfile = io.read()
        
            stims = nwbfile.stimulus
            stim_names = []
            for i in stims:
                #print('stim', i, str(type(stims[i])) == "<class 'pynwb.icephys.CurrentClampStimulusSeries'>")
                stim_names.append(i)
            #print(stim_names)
            
            response = nwbfile.acquisition
            response_names = []
            for i in response:
                #print('res', i, str(type(response[i])) == "<class 'pynwb.icephys.CurrentClampSeries'>")
                response_names.append(i)
            #print(stim_names)
            
            
            for index in range(len(stim_names)): 
                
                this_stim = nwbfile.get_stimulus(stim_names[index])
                this_response = nwbfile.get_acquisition(response_names[index])
                
                stim_units = this_stim.conversion
                response_units = this_response.unit
                              
                if this_stim.unit == 'amperes' and this_response.unit == 'volts' and len(this_stim.data[()]) == len(this_response.data[()]):
                    
                    stim_mag = round_to_sigfig(this_stim.conversion)
                    res_mag = round_to_sigfig(this_response.conversion)
        
                    longest_time = 1000*len(this_stim.data[()])/this_stim.rate #the maximum time in milliseconds
                    times = np.linspace(0, longest_time, len(this_stim.data[()]))
                    
                    volts, current, times = crop_on_voltage_flatline(this_response.data[()], this_stim.data[()], times)
                    
                    stim_res_combo.append([times, current*stim_mag/(1E-9), volts*res_mag/(1E-3)])

                    
    return stim_res_combo
   

folder_path = "D:\\Neuro_Sci\\morph_ephys_trans_stuff\\fine_and_dandi_ephys\\000020\\sub-809076486\\"  # change this to your path

files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

gaming = give_me_the_stuff(folder_path, files)

for entry in gaming:
    #plt.plot(entry[0], entry[1], label = 'nA')
    #plt.plot(entry[0], entry[2], label = 'mV')
    #plt.legend()
    #plt.show()
    
    # Dummy data (replace with your actual data)
    time = entry[0]
    voltage = entry[2]
    current = entry[1]
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    # Plot voltage
    color1 = 'tab:red'
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Voltage (mV)', color=color1)
    ax1.plot(time, voltage, color=color1, label="Voltage")
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # Create second y-axis for current
    ax2 = ax1.twinx()
    color2 = 'tab:blue'
    ax2.set_ylabel('Current (nA)', color=color2)
    ax2.plot(time, current, color=color2, label="Current")
    ax2.tick_params(axis='y', labelcolor=color2)
    
    fig.tight_layout()
    plt.title("Voltage and Current Over Time")
    plt.show()

