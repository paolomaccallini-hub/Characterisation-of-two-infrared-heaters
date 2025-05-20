# file name: Turbo_plot.py
#
# This script is used to plot the results of the Turbo emissive power
#
import os # it provides functions for interacting with the operating system
import pandas as pd # it allows to use dataframes similar to R
import matplotlib.pyplot as plt # it is a plotting library
import math # it provides mathematical functions
from scipy.stats import linregress # it provides functions for linear regression
import numpy as np # it works with arrays and matrices
#
# Physical Costants
#
C1 = 0.59552197e-16  # First radiation constant in W*m^2
C2 = 0.01438769 # Second radiation constant in m*K
C3 = 2898.756 # Wien's displacement constant in um*K
sigma = 5.67e-8  # Stefan-Boltzmann constant in W/m^2/K^4
emissivity = 0.95  # Emissivity of the filament
efficiency = 0.95  # Efficiency of the filament
#
# Costants for the filament
#
N = 25  # Number of coils
D = 0.01  # Diameter of the filament in meters
p = 0.004  # Pitch of the filament in meters
w = 0.0027  # width of the filament in meters
#
# Change the drectory, to the folder of the script
#
script_dir = os.path.dirname(os.path.abspath(__file__))
print(f"script_dir: {script_dir}")
os.chdir(script_dir)
#
# This function calculates the temperature of the filament from the power
#
def filament_temperature(power):
    """
    Calculate the filament temperature based on the power input.
    The formula used is derived from the Stefan-Boltzmann law.
    """
    #
    filament_surface = N*w*((math.pi*D)**2 + p**2)**0.5
    temperature_C = (power * efficiency / (emissivity * sigma * filament_surface) )** 0.25 - 273.15  # Convert to Celsius
    return temperature_C
#
# This function calculates the total emissive power of the filament
#
def filament_total_emissive_power(power):
    """
    Calculate the filament total emissive power based on the power input.
    """
    #
    filament_surface = N*w*((math.pi*D)**2 + p**2)**0.5
    total_emissive_powe = power * efficiency/filament_surface
    return total_emissive_powe
#
# This function calculates the total emissive power of the filament
#
def filament_peak_lambda(power):
    """
    Calculate wavelength at maximum power (Wien's displacement law).
    """
    #
    peak_lambda = C3 / (filament_temperature(power) + 273.15)  # in um
    return peak_lambda
#
# This function gives the hemispherical spectral emissive power of the filament, given the temperature
#
def Plank_law(T):
    """
    Calculate the filament hemispherical spectral emissive power based on the temperature in Kelvin.
    The formula used is the Plank law.
    """
    #
    spectral_emissive_power = [0] * len(wavelength)  # Initialize an array to store temperature values
    for k in range(len(wavelength)):
        lamb = wavelength[k] * 1e-6  # Convert um to m
        num = 2*math.pi*C1
        den = (math.exp(C2/(lamb*T)) - 1)*lamb**5
        spectral_emissive_power[k] = emissivity*num/den
    return spectral_emissive_power
#
# Assign an array of power values
#
power = [value for value in range(300,800,10)] # W
wavelength = [value/100 for value in range(10,500,1)] # um
print(f"power: {power}")
#
# Calculate filament temperature, total emissive power, and peak wavelength for each power value 
#
temperature = [0] * len(power)  # Initialize an array to store temperature values
total_emissive_power = [0] * len(power)  # Initialize an array to store total emissive power values
peak_lambda = [0] * len(power)  # Initialize an array to store peak wavelength values
for k in range(len(power)):
    watt = power[k]
    temperature[k] = filament_temperature(watt)
    total_emissive_power[k] = filament_total_emissive_power(watt)
    peak_lambda[k] = filament_peak_lambda(watt)
    print(f"Power: {watt} W, Temperature: {temperature[k]:.2f} C, Total Emissive Power: {total_emissive_power[k]:.2f} W/m^2, Wavelength at peak power: {peak_lambda[k]:.2f} um")
#
# Plot temperature of the filament as a function of power
#
fig, ax = plt.subplots()
ax.set_xlabel('Power (W)')
ax.set_ylabel('Temperature (C)')
ax.set_title('Filament temperature')
ax.plot( power, temperature, '-', label='Experimental Data')
ax.grid(True)
plt.tight_layout() # Adjust layout for better spacing
plt.savefig('filament_temperature.jpg', dpi=300, bbox_inches='tight') # Save the figure to a file 
plt.close() # Close the current figure
#
# Plot total emissive power of the filament as a function of power
#
fig, ax = plt.subplots()
ax.set_xlabel('Power (W)')
ax.set_ylabel('total emissive power (W/m^2)')
ax.set_title('Total emissive power of the filament')
ax.plot( power, total_emissive_power, '-', label='Experimental Data')
ax.grid(True)
plt.tight_layout() # Adjust layout for better spacing
plt.savefig('filament_emissive_power.jpg', dpi=300, bbox_inches='tight') # Save the figure to a file 
plt.close() # Close the current figure
#
# Plot wavelength at peak emissive power of the filament as a function of power
#   
fig, ax = plt.subplots()
ax.set_xlabel('Power (W)')
ax.set_ylabel('wavelength (um)')
ax.set_title('Wavelength at peak emissive power')
ax.plot( power, peak_lambda, '-', label='Experimental Data')
ax.grid(True)
plt.tight_layout() # Adjust layout for better spacing
plt.savefig('wavelength_peak_power.jpg', dpi=300, bbox_inches='tight') # Save the figure to a file 
plt.close() # Close the current figure
#
# Plot spectral emissive power of the filament for different values of temperature
#
fig, ax = plt.subplots()
temperatures_K = [1000, 1200, 1400, 1600, 1800]  # in Kelvin
k = 0
max_value = [0] * len(temperatures_K)  # Initialize an array to store max emissive power values
max_index = [0] * len(temperatures_K)  # Initialize an array to store max emissive power index values
for T in temperatures_K:
    spectral_emissive_power = Plank_law(T)
    ax.plot(wavelength, spectral_emissive_power, label=f'T = {T} K')
    max_value[k] = max(spectral_emissive_power)
    max_index[k] = spectral_emissive_power.index(max_value[k])
    max_index[k] = wavelength[max_index[k]]
    k = k + 1
#
ax.plot(max_index, max_value, '-o', label='Max Emissive Power') # Plot the max emissive power
ax.set_xlabel('Wavelength (um)')
ax.set_ylabel('Spectral Emissive Power (W/m^2/um)')
ax.set_title('Spectral Emissive Power of the Filament')
ax.grid(True) # Add grid for better readability
ax.legend() # Add a legend to label the curves
plt.tight_layout()
plt.savefig('filament_spectral_emissive_power.jpg', dpi=300)
plt.close() # Close the current figure
#
# Read experimental data 
#
data = {
    'power_W': [303, 315, 350, 335, 405, 426, 449, 403, 457, 658, 611, 617, 631, 595, 689],
    'voltage_V': [123, 127, 134, 147, 150, 153, 156, 161, 162, 186, 205, 206, 208, 214, 222]
}
#
mydata = pd.DataFrame(data)
#
# Perform linear regression between Power and Voltage
#
x = mydata['power_W']
y = mydata['voltage_V']
slope, intercept, r_value, p_value, std_err = linregress(x, y)
print(f"Linear regression results: slope = {slope}, intercept = {intercept}, r_value = {r_value}, p_value = {p_value}, std_err = {std_err}")
y_pred = slope * x + intercept
#
# Plot data and regression line
#
plt.figure(figsize=(8, 5))
plt.scatter(x, y, color='blue', label='Data')
plt.plot(x, y_pred, color='red', label=f'Fit: y = {slope:.2f}x + {intercept:.2f}')
plt.xlabel('Power (W)')
plt.ylabel('Votage (V)')
plt.title('Linear Regression: Power vs. Voltage')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('Power_Voltage.jpg', dpi=300)
plt.close() # Close the current figure
#
# Calculate voltage as a function of power via the linear regression
#
power = np.array([value for value in range(300, 800, 10)])
voltage = slope * power + intercept
#
# Plot temperature of the filament as a function of voltage
#
fig, ax = plt.subplots()
ax.set_xlabel('Voltage (V)')
ax.set_ylabel('Temperature (C)')
ax.set_title('Filament temperature')
ax.plot( voltage, temperature, '-', label='Experimental Data')
ax.grid(True)
plt.tight_layout() # Adjust layout for better spacing
plt.savefig('filament_temperature_voltage.jpg', dpi=300, bbox_inches='tight') # Save the figure to a file 
plt.close() # Close the current figure
#
# Plot wavelength at peak emissive power of the filament as a function of voltage
#   
fig, ax = plt.subplots()
ax.set_xlabel('Voltage (V)')
ax.set_ylabel('wavelength (um)')
ax.set_title('Wavelength at peak emissive power')
ax.plot( voltage, peak_lambda, '-', label='Experimental Data')
ax.grid(True)
plt.tight_layout() # Adjust layout for better spacing
plt.savefig('wavelength_peak_power_voltage.jpg', dpi=300, bbox_inches='tight') # Save the figure to a file 
plt.close() # Close the current figure