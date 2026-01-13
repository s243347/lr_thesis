# %% Load data using MultiDimArray

from MultiDimArray import MultiDimArray
import matplotlib.pyplot as plt
import os

filepath = "data_invitro_rats\\Ecoli\\Ecoli_on_metal_slide_UTI89_785nm"

mda = MultiDimArray(filepath)
wavenumbers = mda.axesCoords[2]
#print(mda.data)           # Show the loaded data array
print(mda.shape)          # Show the shape of the array
#print(mda.axesLabels)     # Show axis labels
#print(mda.currentMap())   # Show the current map slice

# %% Plot first signal
# Get the first signal (Y=0, X=0)
signal = mda.data[0, 0, :]

plt.plot(wavenumbers, signal)
plt.xlabel("Wavenumber (cm$^{-1}$)")
plt.ylabel("Intensity")
plt.title("First Signal (Y=0, X=0)")
plt.show()

# %% Plot all signals in the same plot
plt.figure(figsize=(10, 6))
for y in range(mda.shape[0]):
    for x in range(mda.shape[1]):
        signal = mda.data[y, x, :]
        plt.plot(wavenumbers, signal, label=f"Y={y}, X={x}")

plt.xlabel("Wavenumber (cm$^{-1}$)")
plt.ylabel("Intensity")
plt.title("All Signals E. coli")
plt.show()

# %%
mda = MultiDimArray(filepath)
plt.figure(figsize=(10, 6))
for y in range(mda.shape[0]):
    for x in range(mda.shape[1]):
        if mda.data[y, x, 0] < 16:
            signal = mda.data[y, x, :]
            plt.plot(wavenumbers, signal, label=f"Y={y}, X={x}")
# %% Substract by polynomial approximation
import numpy as np

def plot_polynomial_approximation(signal, degree):
    x = np.arange(len(signal))
    # Fit polynomial
    if np.isnan(signal):
        valid_indices = np.where(~np.isnan(signal))[0]
        last_valid = valid_indices[-1]
        signal = signal[:last_valid + 1]
        x = x[:last_valid + 1]
    coeffs = np.polyfit(x, signal, 2)
    poly_approx = np.polyval(coeffs, x)
    residual = signal - poly_approx

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(x, signal, label='Original Signal')
    plt.plot(x, poly_approx, label=f'Polynomial (deg={degree})', linestyle='--')
    plt.legend()
    plt.title('Signal and Polynomial Approximation')

    plt.subplot(2, 1, 2)
    plt.plot(x, residual, label='Residual (Signal - Poly)')
    plt.legend()
    plt.title('Residual')

    plt.tight_layout()
    plt.show()

plot_polynomial_approximation(mda.data[2, 2, :], degree=3)
# %%
signal = mda.data[2, 2, :]
valid_indices = np.where(~np.isnan(signal))[0]
last_valid = valid_indices[-1]
trimmed_signal = signal[:last_valid + 1]
trimmed_x = x[:last_valid + 1]
coeffs = np.polyfit(trimmed_x, trimmed_signal, 2)
poly_approx = np.polyval(coeffs, trimmed_x)
plt.plot(trimmed_x, poly_approx)
   
# %%
