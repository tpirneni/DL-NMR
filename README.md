# Repository for "Optimizing Neural Network-based Quantification for NMR Metabolomics"
# ----------------------------------------------------------------
# Below are two demos. The first demonstrates generating synthetic mixture spectra from simulated metabolite spectra, and the second demonstrates using a transformer neural network for metabolite quantification in synthetic spectra.
# ----------------------------------------------------------------

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


# ----------------------------------------------------------------

# Test a transformer neural network for metabolite quantification in synthetic NMR spectra 

## First load dependencies, the model, the parameters, and the testing spectra (synthesized in the data generation demo).

#### Import dependencies


```python
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import copy
```

#### Define the transformer model


```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class Transformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout=0.1):
        super(Transformer, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.decoder = nn.Linear(23552, 44)

    def forward(self, x):
        # Binning
        batch_size, seq_length = x.size()
        num_bins = seq_length // self.input_dim
        x = x.view(batch_size, num_bins, self.input_dim)  # (batch_size, num_bins, input_dim)
        
        # Embedding
        x = self.embedding(x)  # (batch_size, num_bins, d_model)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Transformer Encoder
        x = x.permute(1, 0, 2)  # (num_bins, batch_size, d_model)
        x = self.transformer_encoder(x)  # (num_bins, batch_size, d_model)
        x = x.permute(1, 0, 2)  # (batch_size, num_bins, d_model)
        
        # Reconstruct original sequence
        x = x.reshape(batch_size, num_bins * d_model)
        
        # Decoding
        x = self.decoder(x)  # (batch_size, output_dim)
        
        return x

# Parameters
input_dim = 1000  # Size of each bin
d_model = 512     # Embedding dimension
nhead = 1         # Number of attention heads
num_encoder_layers = 1  # Number of transformer encoder layers
dim_feedforward = 2048  # Feedforward dimension
dropout = 0.0     # Dropout rate
```

#### Define model loading function


```python
def train_or_load_model(model, train_loader, test_loader, num_epochs, save_path):
    train_losses = []
    test_losses = []
    is_model_trained = False  # Initialize flag

    if os.path.isfile(save_path):
        print("Loading pretrained model from {}".format(save_path))
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = optim.Adam(model.parameters())  
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        print("No pretrained model found.")
       
    return train_losses, test_losses, is_model_trained  # Return the losses and flag
```

#### Load testing dataset


```python
# Switch to directory containing datasets
os.chdir('/DL-NMR/TestSpectra')

# Load tesing dataset
spectraTest = np.load(f'Dataset44_Uniform_Test_Spec.npy')
concTest = np.load(f'Dataset44_Uniform_Test_Conc.npy')
```

#### Load the model and parameters determined in training


```python
# Switch to directory for saving model parameters
os.chdir('/DL-NMR/SavedParamsAndTrainingMetrics')

# Define the path where you saved your model parameters
save_path = 'Transformer_44met_UniformDist_TrainingAndValidation_ForManuscript_1000ep_Params.pt'

# Load the entire dictionary from the saved file
checkpoint = torch.load(save_path)

# Instantiate the model
NMR_Transformer = Transformer(input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout)

# Load the model's state dictionary from the loaded dictionary
NMR_Transformer.load_state_dict(checkpoint['model_state_dict'])
```




    <All keys matched successfully>



## Apply the trained model to the testing spectra

#### Compute MAPE on the first 25 spectra in the testing set


```python
## Compute absolute percent error statistics on validation set

APEs = []
MAPEs = []
Predictions = []

for i in np.arange(25):
    GroundTruth = concTest[i]
    NMR_Transformer.eval()
    Prediction = NMR_Transformer(torch.tensor(spectraTest[i], dtype=torch.float32).unsqueeze(0))

    # Reformat
    Prediction = Prediction.detach().numpy()

     # Calculate absolute percent error (APE) for each metabolite
    APE = np.abs((GroundTruth - Prediction) / GroundTruth) * 100 

    # Calculate mean absolute percent error (MAPE)
    MAPE = np.mean(APE)
    
    Predictions.append(Prediction)
    APEs.append(APE)
    MAPEs.append(MAPE)

print('Overall MAPE: ',np.array(MAPEs).mean().round(2))
```

    Overall MAPE:  1.47


#### Display ground truth and predicted concentrations for several samples


```python
sample = 0

for i in np.arange (44):
    print("Ground Truth: ", concTest[sample][i].round(2), 
          ";  Prediction: ", Predictions[sample][0][i].round(2), 
          ";  Precent Error: " ,APEs[sample][0][i].round(2))
```

    Ground Truth:  18.63 ;  Prediction:  18.6 ;  Precent Error:  0.18
    Ground Truth:  8.41 ;  Prediction:  8.41 ;  Precent Error:  0.01
    Ground Truth:  13.98 ;  Prediction:  14.05 ;  Precent Error:  0.52
    Ground Truth:  5.34 ;  Prediction:  5.32 ;  Precent Error:  0.33
    Ground Truth:  13.66 ;  Prediction:  13.73 ;  Precent Error:  0.47
    Ground Truth:  3.69 ;  Prediction:  3.77 ;  Precent Error:  2.17
    Ground Truth:  10.93 ;  Prediction:  10.84 ;  Precent Error:  0.8
    Ground Truth:  1.6 ;  Prediction:  1.54 ;  Precent Error:  3.41
    Ground Truth:  5.76 ;  Prediction:  5.72 ;  Precent Error:  0.68
    Ground Truth:  2.55 ;  Prediction:  2.5 ;  Precent Error:  1.88
    Ground Truth:  4.44 ;  Prediction:  4.43 ;  Precent Error:  0.27
    Ground Truth:  7.0 ;  Prediction:  6.96 ;  Precent Error:  0.61
    Ground Truth:  9.38 ;  Prediction:  9.39 ;  Precent Error:  0.2
    Ground Truth:  10.11 ;  Prediction:  10.15 ;  Precent Error:  0.41
    Ground Truth:  19.82 ;  Prediction:  19.82 ;  Precent Error:  0.01
    Ground Truth:  14.02 ;  Prediction:  14.0 ;  Precent Error:  0.11
    Ground Truth:  4.35 ;  Prediction:  4.36 ;  Precent Error:  0.31
    Ground Truth:  11.35 ;  Prediction:  11.28 ;  Precent Error:  0.59
    Ground Truth:  7.08 ;  Prediction:  7.02 ;  Precent Error:  0.82
    Ground Truth:  19.08 ;  Prediction:  18.99 ;  Precent Error:  0.43
    Ground Truth:  18.96 ;  Prediction:  18.96 ;  Precent Error:  0.01
    Ground Truth:  8.13 ;  Prediction:  8.19 ;  Precent Error:  0.75
    Ground Truth:  8.63 ;  Prediction:  8.73 ;  Precent Error:  1.13
    Ground Truth:  19.41 ;  Prediction:  19.45 ;  Precent Error:  0.2
    Ground Truth:  14.13 ;  Prediction:  14.15 ;  Precent Error:  0.15
    Ground Truth:  5.86 ;  Prediction:  5.89 ;  Precent Error:  0.37
    Ground Truth:  15.24 ;  Prediction:  15.21 ;  Precent Error:  0.19
    Ground Truth:  8.83 ;  Prediction:  8.82 ;  Precent Error:  0.1
    Ground Truth:  11.07 ;  Prediction:  11.17 ;  Precent Error:  0.88
    Ground Truth:  16.64 ;  Prediction:  16.69 ;  Precent Error:  0.29
    Ground Truth:  14.57 ;  Prediction:  14.61 ;  Precent Error:  0.27
    Ground Truth:  5.39 ;  Prediction:  5.41 ;  Precent Error:  0.27
    Ground Truth:  7.27 ;  Prediction:  7.32 ;  Precent Error:  0.63
    Ground Truth:  9.94 ;  Prediction:  9.9 ;  Precent Error:  0.43
    Ground Truth:  8.24 ;  Prediction:  8.22 ;  Precent Error:  0.16
    Ground Truth:  2.26 ;  Prediction:  2.3 ;  Precent Error:  1.62
    Ground Truth:  12.94 ;  Prediction:  12.99 ;  Precent Error:  0.36
    Ground Truth:  4.3 ;  Prediction:  4.27 ;  Precent Error:  0.64
    Ground Truth:  11.85 ;  Prediction:  11.93 ;  Precent Error:  0.7
    Ground Truth:  17.63 ;  Prediction:  17.59 ;  Precent Error:  0.24
    Ground Truth:  9.54 ;  Prediction:  9.57 ;  Precent Error:  0.31
    Ground Truth:  15.97 ;  Prediction:  16.0 ;  Precent Error:  0.19
    Ground Truth:  4.32 ;  Prediction:  4.35 ;  Precent Error:  0.72
    Ground Truth:  6.93 ;  Prediction:  7.02 ;  Precent Error:  1.34



```python
sample = 1

for i in np.arange (44):
    print("Ground Truth: ", concTest[sample][i].round(2), 
          ";  Prediction: ", Predictions[sample][0][i].round(2), 
          ";  Precent Error: " ,APEs[sample][0][i].round(2))
```

    Ground Truth:  8.55 ;  Prediction:  8.49 ;  Precent Error:  0.76
    Ground Truth:  5.44 ;  Prediction:  5.37 ;  Precent Error:  1.34
    Ground Truth:  16.68 ;  Prediction:  16.77 ;  Precent Error:  0.53
    Ground Truth:  14.05 ;  Prediction:  14.13 ;  Precent Error:  0.56
    Ground Truth:  9.13 ;  Prediction:  9.22 ;  Precent Error:  1.01
    Ground Truth:  13.26 ;  Prediction:  13.33 ;  Precent Error:  0.48
    Ground Truth:  7.53 ;  Prediction:  7.44 ;  Precent Error:  1.28
    Ground Truth:  12.52 ;  Prediction:  12.48 ;  Precent Error:  0.39
    Ground Truth:  17.71 ;  Prediction:  17.73 ;  Precent Error:  0.1
    Ground Truth:  4.58 ;  Prediction:  4.56 ;  Precent Error:  0.35
    Ground Truth:  8.54 ;  Prediction:  8.53 ;  Precent Error:  0.08
    Ground Truth:  8.02 ;  Prediction:  8.01 ;  Precent Error:  0.12
    Ground Truth:  9.55 ;  Prediction:  9.61 ;  Precent Error:  0.58
    Ground Truth:  9.83 ;  Prediction:  9.85 ;  Precent Error:  0.23
    Ground Truth:  3.59 ;  Prediction:  3.57 ;  Precent Error:  0.51
    Ground Truth:  9.36 ;  Prediction:  9.31 ;  Precent Error:  0.52
    Ground Truth:  7.33 ;  Prediction:  7.3 ;  Precent Error:  0.44
    Ground Truth:  15.29 ;  Prediction:  15.21 ;  Precent Error:  0.5
    Ground Truth:  4.06 ;  Prediction:  4.03 ;  Precent Error:  0.7
    Ground Truth:  6.52 ;  Prediction:  6.47 ;  Precent Error:  0.7
    Ground Truth:  3.75 ;  Prediction:  3.76 ;  Precent Error:  0.22
    Ground Truth:  17.05 ;  Prediction:  17.14 ;  Precent Error:  0.51
    Ground Truth:  13.16 ;  Prediction:  13.23 ;  Precent Error:  0.51
    Ground Truth:  7.93 ;  Prediction:  7.89 ;  Precent Error:  0.43
    Ground Truth:  4.93 ;  Prediction:  4.92 ;  Precent Error:  0.34
    Ground Truth:  6.03 ;  Prediction:  6.09 ;  Precent Error:  1.02
    Ground Truth:  17.69 ;  Prediction:  17.7 ;  Precent Error:  0.07
    Ground Truth:  10.91 ;  Prediction:  11.01 ;  Precent Error:  0.92
    Ground Truth:  4.76 ;  Prediction:  4.77 ;  Precent Error:  0.26
    Ground Truth:  3.0 ;  Prediction:  3.08 ;  Precent Error:  2.89
    Ground Truth:  8.04 ;  Prediction:  8.05 ;  Precent Error:  0.17
    Ground Truth:  17.46 ;  Prediction:  17.46 ;  Precent Error:  0.01
    Ground Truth:  3.6 ;  Prediction:  3.58 ;  Precent Error:  0.57
    Ground Truth:  16.8 ;  Prediction:  16.87 ;  Precent Error:  0.42
    Ground Truth:  1.32 ;  Prediction:  1.3 ;  Precent Error:  1.44
    Ground Truth:  6.37 ;  Prediction:  6.39 ;  Precent Error:  0.25
    Ground Truth:  4.35 ;  Prediction:  4.34 ;  Precent Error:  0.2
    Ground Truth:  7.55 ;  Prediction:  7.51 ;  Precent Error:  0.59
    Ground Truth:  19.05 ;  Prediction:  19.1 ;  Precent Error:  0.25
    Ground Truth:  16.09 ;  Prediction:  16.09 ;  Precent Error:  0.01
    Ground Truth:  8.69 ;  Prediction:  8.75 ;  Precent Error:  0.66
    Ground Truth:  8.51 ;  Prediction:  8.54 ;  Precent Error:  0.34
    Ground Truth:  0.89 ;  Prediction:  0.9 ;  Precent Error:  0.61
    Ground Truth:  18.31 ;  Prediction:  18.43 ;  Precent Error:  0.61



```python
sample = 2

for i in np.arange (44):
    print("Ground Truth: ", concTest[sample][i].round(2), 
          ";  Prediction: ", Predictions[sample][0][i].round(2), 
          ";  Precent Error: " ,APEs[sample][0][i].round(2))
```

    Ground Truth:  8.09 ;  Prediction:  8.0 ;  Precent Error:  1.22
    Ground Truth:  19.94 ;  Prediction:  19.88 ;  Precent Error:  0.31
    Ground Truth:  15.14 ;  Prediction:  15.22 ;  Precent Error:  0.55
    Ground Truth:  2.71 ;  Prediction:  2.65 ;  Precent Error:  2.09
    Ground Truth:  9.28 ;  Prediction:  9.31 ;  Precent Error:  0.28
    Ground Truth:  16.98 ;  Prediction:  17.1 ;  Precent Error:  0.69
    Ground Truth:  3.91 ;  Prediction:  3.81 ;  Precent Error:  2.78
    Ground Truth:  10.04 ;  Prediction:  10.03 ;  Precent Error:  0.09
    Ground Truth:  3.71 ;  Prediction:  3.7 ;  Precent Error:  0.25
    Ground Truth:  12.04 ;  Prediction:  12.1 ;  Precent Error:  0.55
    Ground Truth:  2.76 ;  Prediction:  2.78 ;  Precent Error:  0.62
    Ground Truth:  1.76 ;  Prediction:  1.78 ;  Precent Error:  0.89
    Ground Truth:  4.19 ;  Prediction:  3.87 ;  Precent Error:  7.5
    Ground Truth:  12.78 ;  Prediction:  12.76 ;  Precent Error:  0.16
    Ground Truth:  15.76 ;  Prediction:  15.81 ;  Precent Error:  0.35
    Ground Truth:  8.33 ;  Prediction:  8.3 ;  Precent Error:  0.3
    Ground Truth:  11.22 ;  Prediction:  11.22 ;  Precent Error:  0.03
    Ground Truth:  17.31 ;  Prediction:  17.54 ;  Precent Error:  1.33
    Ground Truth:  3.37 ;  Prediction:  3.32 ;  Precent Error:  1.5
    Ground Truth:  16.96 ;  Prediction:  16.89 ;  Precent Error:  0.42
    Ground Truth:  14.4 ;  Prediction:  14.49 ;  Precent Error:  0.61
    Ground Truth:  1.04 ;  Prediction:  1.06 ;  Precent Error:  2.41
    Ground Truth:  5.84 ;  Prediction:  5.9 ;  Precent Error:  1.09
    Ground Truth:  7.08 ;  Prediction:  7.06 ;  Precent Error:  0.24
    Ground Truth:  9.73 ;  Prediction:  9.69 ;  Precent Error:  0.36
    Ground Truth:  10.4 ;  Prediction:  10.55 ;  Precent Error:  1.42
    Ground Truth:  8.32 ;  Prediction:  8.35 ;  Precent Error:  0.27
    Ground Truth:  10.06 ;  Prediction:  10.09 ;  Precent Error:  0.28
    Ground Truth:  19.65 ;  Prediction:  19.79 ;  Precent Error:  0.72
    Ground Truth:  12.88 ;  Prediction:  12.92 ;  Precent Error:  0.29
    Ground Truth:  7.32 ;  Prediction:  7.35 ;  Precent Error:  0.39
    Ground Truth:  12.77 ;  Prediction:  12.8 ;  Precent Error:  0.22
    Ground Truth:  9.78 ;  Prediction:  9.88 ;  Precent Error:  1.0
    Ground Truth:  18.96 ;  Prediction:  18.98 ;  Precent Error:  0.13
    Ground Truth:  19.52 ;  Prediction:  19.58 ;  Precent Error:  0.29
    Ground Truth:  9.31 ;  Prediction:  9.37 ;  Precent Error:  0.66
    Ground Truth:  12.63 ;  Prediction:  12.67 ;  Precent Error:  0.31
    Ground Truth:  18.77 ;  Prediction:  18.72 ;  Precent Error:  0.3
    Ground Truth:  13.37 ;  Prediction:  13.43 ;  Precent Error:  0.41
    Ground Truth:  4.19 ;  Prediction:  4.2 ;  Precent Error:  0.24
    Ground Truth:  10.99 ;  Prediction:  10.97 ;  Precent Error:  0.17
    Ground Truth:  4.98 ;  Prediction:  4.97 ;  Precent Error:  0.14
    Ground Truth:  10.3 ;  Prediction:  10.27 ;  Precent Error:  0.31
    Ground Truth:  10.04 ;  Prediction:  10.15 ;  Precent Error:  1.1



