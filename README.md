# Repository for "Optimizing Neural Network-based Quantification for NMR Metabolomics"
# -------------------------------------------------------------------------
# Below are two demos. The first demonstrates generating synthetic mixture spectra from simulated metabolite spectra, and the second demonstrates using a transformer neural network for metabolite quantification in synthetic spectra.
# -------------------------------------------------------------------------

# The following demo demonstrates generating synthetic metabolite mixture NMR spectra for testing a neural network trained for metabolite quantification. Spectra are generated using a uniform concentration distribution ranging from 0.05 to 20 mM.

## First load some dependencies and required functions/variables for data generation

#### Load dependencies


```python
import numpy as np
import os
import nmrglue as ng
import random
import matplotlib.pyplot as plt
```

#### Define function for varying line-broadening


```python
def exponential_apodization(input_array, apodization_frequency):
    length = len(v)
    time = np.arange(length)
    decay_factor = np.exp(-time * apodization_frequency * 2 * np.pi / length)
    return input_array * decay_factor
```

#### Read in simulated spectra of 44 metabolites downloaded from HMDB, and create a ppm scale


```python
# Load spectra of simulated standards to be analytes
filenames = os.listdir('/DL-NMR/HMDBSpectraFiles/87Met')  # Get a list of filenames in the directory
filenames = filenames[:43]
os.chdir("/DL-NMR/HMDBSpectraFiles/87Met")  # Switch to appropriate folders and load data

standardsSpec =[]  # initialize variable to hold spectra of standards
standardsDictionary = []  # initialize variable to hold dictionaries of spectra of standards
for i in np.arange(len(filenames)):
    dic, data = ng.jcampdx.read(filenames[i])
    standardsSpec.append(data)
    standardsDictionary.append(dic)
    
    
# Create ppm scale for plotting ppm rather than just data point numbers for x-axis
ppm_all = (np.linspace(12.5, -2.5, 65536))
ppm = ppm_all[10000:56000]
```

#### Define quantitative reference signal and interference signals


```python
# Load quantitative reference signal
os.chdir("/DL-NMR/HMDBSpectraFiles")
vd, v = ng.jcampdx.read('HMDB0000176_142710_predicted_H_400.jdx') # maleic acid, our quantitative reference
v = ng.process.proc_base.cs(v, 28305)  # Shift maleic acid peak to 0 ppm to simulate a TSP peak (need to find good simulated TSP peak)


# Define signal for a generic singlet for use as interfering signal (using acetic acid singlet from HMDB)
os.chdir("/DL-NMR/HMDBSpectraFiles/87Met")  # Switch to appropriate folders and load data
vXd, vX = ng.jcampdx.read('HMDB0000042_5433_predicted_H_400.jdx') # acetic acid, for adding random singlets
Singlet = vX[vX != 0]  # make variable of acetic acid without zeros
```

#### Make single glucose spectrum from spectra of two glucose anomers


```python
# Load glucose spectra
AlphaGlucoseDictionary, AlphaGlucoseSpec = ng.jcampdx.read('HMDB0003345_135690_predicted_H_400.jdx')
BetaGlucoseDictionary, BetaGlucoseSpec = ng.jcampdx.read('HMDB0000122_142311_predicted_H_400.jdx')
## Combine anomers
GlucoseSpec = 0.36*AlphaGlucoseSpec + 0.64*BetaGlucoseSpec
## Combine all spectra into list
standardsSpec.append(GlucoseSpec)
standardsSpec = np.array(standardsSpec) 
```

#### Display the individual spectra


```python
## Loop to plot the simluated HMDB spectra offset on the y-axis 

# Set plot to interactive
%matplotlib notebook

# Set figure size
plt.figure(figsize=(8, 25)) 

# Plot all 44 simulated spectra
n = 0
for i in np.arange(44):
    plt.plot(ppm_all, standardsSpec[i] + n)
    n += standardsSpec[i].max()
    
# Set layout
plt.tight_layout()

# Plot pseudo-TSP-d4 peak
plt.plot(ppm_all, v - 2.5)

# Ensure ppm axis is oriented correctly
plt.xlim([ppm_all[0],ppm_all[-1]])
```
![SimulatedMetaboliteSpectraOffset](https://github.com/user-attachments/assets/912abaa2-e939-400f-8abb-d8866e76ef92)




## Generate a testing dataset with 44 metabolites. Incoporating experimental variations in the form of linebroadening, noise, peakshift, and baseline shift, and incorporating random singlets and multiplets. Generate 25 spectra containting all 44 metabolites, and 25 spectra with a 50% chance to leave out any metabolite.


```python
## Generate training and testing dataset 
## Use linear combinations with physically inspired modifications (line-broadening, noise, baseline shift, and peak shift) 

## Switch to folder where datasets will be saved
os.chdir('/DL-NMR/GeneratedDataAndVariables')

# Create some empty lists to contain spectra/concentrations as they are generated, and set some variables for the data generation function
spectra = []
conc = []
max_shift = 15
iterations = 25

# Set a seed for reproducible data generation
np.random.seed(2)
random.seed(2)

## Loop to generate synthetic data without leaving out any metabolites
for i in range(iterations):
    xdata = []
    ydata = []
    
    # Loop through and scale each standard, potentially leaving each out. Do the same for metabolite concentrations. Combine all spectra and metabolite profiles into separate lists. 
    # Sum all spectra into one synthetic mixture and add maleic acid as reference. Create random seeds and random variables throughout.
    for ii in np.arange(44):
        ShiftDirection = np.random.uniform(-1,1) 
        shifted = ng.process.proc_base.cs(standardsSpec[ii],np.random.choice([0,ShiftDirection*max_shift]))  #shift peaks
        Concentration = np.random.uniform(0.005,20)
        scaled = Concentration*shifted #scale the peak shifted spectra
        xdata.append(scaled)  #add all shifted, scaled spectra to list
        ydata.append(Concentration)  #add all concentrations to list
    xdata = np.array(xdata).sum(axis=0) + v*0.3*(9/2)  #sum all 10 metabolite spectra and add psuedo-TSP-d4 as reference
    ydata = np.array(ydata)
    
    # Inverse FFT (real-valued), apply line-broadening, and FFT (real-valued)
    linebroad = np.random.uniform(0,1)
    xdata = ng.proc_base.irft(xdata)
    xdata = exponential_apodization(xdata, linebroad)
    xdata = ng.proc_base.rft(xdata)
    
    # Select signal region and add random noise
    noise = np.random.uniform(0.3,1.15)*0.003975
    xdata = xdata[10000:56000]+np.random.normal(0, noise,size = 46000)
    
    # Shift baseline up or down
    xdata = xdata + np.random.uniform(-0.1,0.1)
    
    
    # Define the intensity distribution for interfering singlets
    InterferenceRange = 150*np.random.gamma(0.2, 0.02, size = 30)
    for i in np.arange(30):
        InterferenceRange[i] += np.random.uniform(0.005, 0.5) + np.random.choice([0, np.random.uniform(0.005, 0.02)])
    
    # Randomly add three singlet signals
    for i in np.arange(3):
        Intensity = np.random.choice(InterferenceRange)
        Sig1 = Singlet*Intensity
        Sig1 = random.choice([np.zeros(2995),Sig1])
        Placement = random.choice(np.arange(46000-2995))
        xdata[Placement:Placement+2995] = xdata[Placement:Placement+2995] + Sig1
    
    # Append spectrum and metabolite profile variables to list
    spectra.append(xdata)
    conc.append(ydata) 
        
    


## Loop to generate synthetic data with 50% chance to leave out out any metabolite
for i in range(iterations):
    xdata = []
    ydata = []
    
    # Loop through and scale each standard, potentially leaving each out. Do the same for metabolite concentrations. Combine all spectra and metabolite profiles into separate lists. 
    # Sum all spectra into one synthetic mixture and add maleic acid as reference. Create random seeds and random variables throughout.
    for ii in np.arange(44):
        ShiftDirection = np.random.uniform(-1,1) 
        shifted = ng.process.proc_base.cs(standardsSpec[ii],np.random.choice([0,ShiftDirection*max_shift]))  #shift peaks
        Concentration = np.random.uniform(0.005,20)
        ConcDecision = np.random.choice([0,1])
        scaled = Concentration*ConcDecision*shifted #scale the peak shifted spectra
        c = Concentration*ConcDecision  #define concentration
        xdata.append(scaled)  #add all shifted, scaled spectra to list
        ydata.append(c)  #add all concentrations to list
    xdata = np.array(xdata).sum(axis=0) + v*0.3*(9/2)  #sum all 10 metabolite spectra and add psuedo-TSP-d4 as reference
    ydata = np.array(ydata)
    
    # Inverse FFT (real-valued), apply line-broadening, and FFT (real-valued)
    linebroad = np.random.uniform(0,1)
    xdata = ng.proc_base.irft(xdata)
    xdata = exponential_apodization(xdata, linebroad)
    xdata = ng.proc_base.rft(xdata)
    
    # Select signal region and add random noise
    noise = np.random.uniform(0.3,1.15)*0.003975
    xdata = xdata[10000:56000]+np.random.normal(0, noise, size = 46000)
    
    # Shift baseline up or down
    xdata = xdata + np.random.uniform(-0.1,0.1)
    
    # Define the intensity distribution for interfering singlets
    InterferenceRange = 150*np.random.gamma(0.2, 0.02, size = 30)
    for i in np.arange(30):
        InterferenceRange[i] += np.random.uniform(0.005, 0.5) + np.random.choice([0, np.random.uniform(0.005, 0.02)])
    # Randomly add three singlet signals
    for i in np.arange(3):
        Intensity = np.random.choice(InterferenceRange)
        Sig1 = Singlet*Intensity
        Sig1 = random.choice([np.zeros(2995),Sig1])
        Placement = random.choice(np.arange(46000-2995))
        xdata[Placement:Placement+2995] = xdata[Placement:Placement+2995] + Sig1
    
     # Append spectrum and metabolite profile variables to list
    spectra.append(xdata)
    conc.append(ydata)         
        
        
        
# This is the overall max from the training dataset (this is used to scale in the same way as done prior to model training)
OvMax = 231.66897046506165


# Scale all spectra by the 
spectra = np.array(spectra)/OvMax


# Save the arrays to files, and save max value seen in training/testing datasets
np.save('Dataset44_Uniform_Test_Spec.npy', spectra)
np.save('Dataset44_Uniform_Test_Conc.npy', conc)
```

## Display examples of synthetic spectra generated using HMDB simulated spectra


```python
## Loop to plot the first 4 spectra generated

# Enable inline plotting
%matplotlib inline

# Plot 4 generated spectra
for i in np.arange(4):
    plt.figure(figsize=(10, 5)) 
    plt.plot(ppm, spectra[i])
    plt.xlim([ppm[0],ppm[-1]])  # Ensure ppm axis is oriented correctly
    plt.show()
# -------------------------------------------------------------------------
```

![output_17_3](https://github.com/user-attachments/assets/d3fa9499-4aa4-4d8c-986d-7f1aaa1e43ba)
![output_17_2](https://github.com/user-attachments/assets/0d0c8fd0-12ea-4d31-b7a4-4d0096e18e3a)
![output_17_1](https://github.com/user-attachments/assets/22213841-3ae7-4c16-895c-56c5160dd0bd)
![output_17_0](https://github.com/user-attachments/assets/77f42f1b-d1ca-4058-be93-2c2bdd3a84bb)
