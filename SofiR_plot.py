# file name: SofiR_plot.py
#
# This script is used to plot the results of the SofiR emissive power
#
import os # functions for interacting with the operating system
import pandas as pd # dataframes similar to R
import matplotlib.pyplot as plt # a plotting library
import math # mathematical functions
from scipy.stats import linregress # functions for linear regression
import numpy as np # arrays and matrices
#
# Change the drectory, to the folder of the script
#
script_dir = os.path.dirname(os.path.abspath(__file__))
print(f"script_dir: {script_dir}")
os.chdir(script_dir)
#
# Physical Costants
#
C1 = 0.59552197e-16  # First radiation constant in W*m^2
C2 = 0.01438769 # Second radiation constant in m*K
C3 = 2898.756 # Wien's displacement constant in um*K
sigma = 5.67e-8  # Stefan-Boltzmann constant in W/m^2/K^4
emissivity = 0.90  # Emissivity of the ceramic element
#
# Geometric parameters 
#
R = 0.06025 # m
theta = 0.52 # rad
L = 0.12 # m
S = 2*theta*R*L # radiating surface aerea
#
# This function calculates the efficiency given the power
#
def efficiency(power):
    """
    Calculate the efficiency of the ceramic element 
    plus the reflector
    """
    #
    T = poly_func(power) # Temperature in K
    efficiency = emissivity*sigma*S*T**4 / power
    return efficiency
#
# This function calculates the total emissive power of the filament
#
def peak_lambda(power):
    """
    Calculate wavelength at maximum power (Wien's displacement law).
    """
    #
    T = poly_func(power) + 273.15 # Temperature in K
    peak_lambda = C3 / T  # in um
    return peak_lambda
#
# Read experimental data 
#
data = {
    'voltage_V': [42,58,75,91,107,114,133,151,174,198,220],
    'temperature_C': [194.1,267.6,329,378.7,422.4,442.5,495.4,543,588,640.7,680.7],
    'power_W': [60.6,95.7,129.2,169.1,202,218.3,265.1,312.1,359.4,399.3,465.1],
}
mydata = pd.DataFrame(data)
#
# Perform linear regression between Power and Voltage
#
x = mydata['power_W']
y = mydata['voltage_V']
slope_V, intercept_V, r_value, p_value, std_err = linregress(x, y)
pred_V = slope_V * x + intercept_V
#
# Plot data and regression line
#
plt.figure(figsize=(8, 5))
plt.scatter(x, y, color='blue', label='Data')
plt.plot(x, pred_V, color='red', label=f'Fit: y = {slope_V:.2f}x + {intercept_V:.2f}')
plt.xlabel('Power (W)')
plt.ylabel('Votage (V)')
plt.title('Linear Regression: Power vs. Voltage')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('Power_Voltage_Sofi.jpg', dpi=300)
plt.close() # Close the current figure
#
# Perform polynomial regression between Power and Temperature
#
x = mydata['power_W']
y = mydata['temperature_C'] 
degree = 2
coeffs = np.polyfit(x, y, degree)
poly_func = np.poly1d(coeffs)
x_fit = np.linspace(min(x), max(x), 500)
y_fit = poly_func(x_fit)
#
# Plot data and regression line
#
plt.figure(figsize=(8, 5))
plt.scatter(x, y, color='blue', label='Data')
plt.plot(x_fit, y_fit, color='red', label=f'Polynomial Fit (deg={degree})')
plt.xlabel('Power (W)')
plt.ylabel('Temperature (C)')
plt.title('Polynomial Regression: Power vs. Temperature')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('Power_Temperature_Sofi.jpg', dpi=300)
plt.close() # Close the current figure
#
# Assign an array of power values
#
power_max = max(mydata['power_W']) # Maximum power
power_W = [value for value in range(60,power_max,10)] # W
print(f"Power (W): {power_W}")
#
# Calculate temperature an voltage for each power value
#
temperature_K = [0] * len(power_W)  # Initialize an array to store temperature values
temperature_C = [0] * len(power_W)  # Initialize an array to store temperature values
voltage_V = [0] * len(power_W)  # Initialize an array to store voltage values
peak_lambda_um = [0] * len(power_W)  # Initialize an array to store peak wavelength values
efficiency_values = [0] * len(power_W)  # Initialize an array to store efficiency values
#
for k in range(len(power_W)):
    watt = power_W[k]
    temperature_C[k] = poly_func(watt) # Temperature in Celsius
    temperature_K[k] = temperature_C[k] + 273.15 # Temperature in K
    voltage_V[k] = slope_V * watt + intercept_V # Voltage in V
    peak_lambda_um[k] = peak_lambda(watt) # Peak wavelength in um
    efficiency_values[k] = efficiency(watt) # Efficiency
print(f"Temperature (K): {temperature_K}")
print(f"Voltage (V): {voltage_V}")  
print(f"Peak Wavelength (um): {peak_lambda_um}")        
#
# Plot wavelength at peak emissive power as a function of voltage
#   
fig, ax = plt.subplots()
ax.set_xlabel('voltage (V)')
ax.set_ylabel('wavelength (um)')
ax.set_title('Wavelength at peak emissive power')
ax.plot( voltage_V, peak_lambda_um, '-', label='Experimental Data')
ax.grid(True)
plt.tight_layout() # Adjust layout for better spacing
plt.savefig('wavelength_peak_power_Sofi.jpg', dpi=300, bbox_inches='tight') # Save the figure to a file 
plt.close() # Close the current figure
#
# Plot efficiency as a function of power
#   
fig, ax = plt.subplots()
ax.set_xlabel('Voltage (V)')
ax.set_ylabel('efficiency')
ax.set_title('Wavelength at peak emissive power')
ax.plot( voltage_V, efficiency_values, '-', label='Experimental Data')
ax.grid(True)
plt.tight_layout() # Adjust layout for better spacing
plt.savefig('efficiency_Sofi.jpg', dpi=300, bbox_inches='tight') # Save the figure to a file 
plt.close() # Close the current figure

