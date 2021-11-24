# %% IMPORTS AND SIGNAL LOADING
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal as sg
import pickle

data_path = 'signals/'
signal_name = 'ecg.csv'

signal = pd.read_csv(os.path.join(data_path,signal_name))
signal_chunk = signal[14000:14500].values.flatten()

plt.figure()
plt.plot(signal_chunk)
plt.show()


# %% ADDING WHITE NOISE

noise = np.random.random(len(signal_chunk)) 
signal_max = max(signal_chunk)
noise_scaled = noise*signal_max
noised_signal = signal_chunk + noise_scaled

plt.figure()
plt.plot(noised_signal)
plt.show()


# %% CREATING MANY REALIZATION
NUMS = [100, 1000,10000,30000]
averages = [signal_chunk]
for NUM_REAL in NUMS:
    buff = []
    for i in range(NUM_REAL):
        noise = np.random.random(len(signal_chunk)) 
        signal_max = max(signal_chunk)
        noise_scaled = noise*signal_max
        noised_signal = signal_chunk + noise_scaled
        buff.append(noised_signal)
        
    
    summary_signal = np.zeros(len(signal_chunk))
    
    for i in range(len(buff)):
        summary_signal+= buff[i]
    
    summary_signal/= NUM_REAL
    averages.append(summary_signal)

# %%
f, ax = plt.subplots(5,1)
ax[0].plot(signal_chunk)
ax[1].plot(noised_signal)
ax[2].plot(averages[1])
ax[3].plot(averages[2])
ax[4].plot(averages[3])

# %% SAVING DATA
with open(os.path.join(data_path,'realizations.p'), 'wb') as f:
    pickle.dump(averages,f)


