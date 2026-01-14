# %% Import packages
import time
from MultiDimArray import MultiDimArray
import matplotlib.pyplot as plt
import os
import numpy as np


def pol_approx_subs(signal, degree):
    x = np.arange(len(signal))
    # Fit polynomial
    if np.any(np.isnan(signal)):
        valid_indices = np.where(~np.isnan(signal))[0]
        last_valid = valid_indices[-1]
        signal = signal[:last_valid + 1]
        x = x[:last_valid + 1]
    coeffs = np.polyfit(x, signal, 2)
    poly_approx = np.polyval(coeffs, x)
    residual = signal - poly_approx

    return x, signal, poly_approx, residual

# %% Load Ecoli data

filepath = "data_invitro_rats\\Ecoli\\Ecoli_on_metal_slide_UTI89_785nm"

mda = MultiDimArray(filepath)
wavenumbers = mda.axesCoords[2]
plt.figure(figsize=(10, 6))
for y in range(mda.shape[0]):
    for x in range(mda.shape[1]):
        signal = mda.data[y, x, :]
        plt.plot(wavenumbers, signal, label=f"Y={y}, X={x}")

plt.xlabel("Wavenumber (cm$^{-1}$)")
plt.ylabel("Intensity")
plt.title("E. coli")
plt.show()

plt.figure(figsize=(10, 6))
for y in range(mda.shape[0]):
    for x in range(mda.shape[1]):
        if mda.data[y, x, 0] < 16:
            signal = mda.data[y, x, :]
            x = np.arange(len(signal))
            x, signal, poly_approx, residual = pol_approx_subs(signal, degree = 3)  
            plt.plot(x, residual, label=f"Y={y}, X={x}")

plt.xlabel("Wavenumber (cm$^{-1}$)")
plt.ylabel("Intensity")
plt.title("E. coli w/ Poly Subtracted")
plt.show()

# %% Load Assumptomatic Ecoli data

filepath = "data_invitro_rats\\Assymptomatic_Ecoli\\Assymptomatic_Ecoli_1_785nm"

mda = MultiDimArray(filepath)
wavenumbers = mda.axesCoords[2]
plt.figure(figsize=(10, 6))
for y in range(mda.shape[0]):
    for x in range(mda.shape[1]):
        signal = mda.data[y, x, :]
        plt.plot(wavenumbers, signal, label=f"Y={y}, X={x}")
plt.xlabel("Wavenumber (cm$^{-1}$)")
plt.ylabel("Intensity")
plt.title("Assymptomatic E. coli")
plt.show()

plt.figure(figsize=(10, 6))
for y in range(mda.shape[0]):
    for x in range(mda.shape[1]):
        if mda.data[y, x, 0] < 16:
            signal = mda.data[y, x, :]
            x = np.arange(len(signal))
            x, signal, poly_approx, residual = pol_approx_subs(signal, degree = 3)  
            plt.plot(x, residual, label=f"Y={y}, X={x}")

plt.xlabel("Wavenumber (cm$^{-1}$)")
plt.ylabel("Intensity")
plt.title("Assymptomatic E. coli w/ Poly Subtracted")
plt.show()

# %% Load Protius data

filepath = "data_invitro_rats\\Protius\\Protius_agar_plate_785nm"

mda = MultiDimArray(filepath)
wavenumbers = mda.axesCoords[2]
plt.figure(figsize=(10, 6))
for y in range(mda.shape[0]):
    for x in range(mda.shape[1]):
        signal = mda.data[y, x, :]
        plt.plot(wavenumbers, signal, label=f"Y={y}, X={x}")
plt.xlabel("Wavenumber (cm$^{-1}$)")
plt.ylabel("Intensity")
plt.title("Protius")
plt.show()

plt.figure(figsize=(10, 6))
for y in range(mda.shape[0]):
    for x in range(mda.shape[1]):
        signal = mda.data[y, x, :]
        x = np.arange(len(signal))
        x, signal, poly_approx, residual = pol_approx_subs(signal, degree = 3)  
        plt.plot(x, residual, label=f"Y={y}, X={x}")

plt.xlabel("Wavenumber (cm$^{-1}$)")
plt.ylabel("Intensity")
plt.title("Protius w/ Poly Subtracted")
plt.show()

# %% Load S. Saprofyticus data

filepath = "data_invitro_rats\\Ssaprofyticus\\ssaprofyticus_scan_785nm"

mda = MultiDimArray(filepath)
wavenumbers = mda.axesCoords[2]
plt.figure(figsize=(10, 6))
for y in range(mda.shape[0]): 
    for x in range(mda.shape[1]):
        signal = mda.data[y, x, :]
        if np.any(signal < 30):
            plt.plot(wavenumbers, signal, label=f"Y={y}, X={x}")
plt.xlabel("Wavenumber (cm$^{-1}$)")
plt.ylabel("Intensity")
plt.title("S. Saprofyticus")
plt.show()

plt.figure(figsize=(10, 6))
for y in range(mda.shape[0]):
    for x in range(mda.shape[1]):
        signal = mda.data[y, x, :]
        if np.any(signal < 30):
            x = np.arange(len(signal))
            x, signal, poly_approx, residual = pol_approx_subs(signal, degree = 3)  
            plt.plot(x, residual, label=f"Y={y}, X={x}")

plt.xlabel("Wavenumber (cm$^{-1}$)")
plt.ylabel("Intensity")
plt.title("S. Saprofyticus w/ Poly Subtracted")
plt.show()

# %%
signal = mda.data[9, 1, :]
plt.figure()
plt.plot(np.arange(len(signal)), signal, label=f"Y={y}, X={x}")
la = ~ (np.any(signal) < 30)
print (la)
# %%
count = 0
for y in range(mda.shape[0]):
    for x in range(mda.shape[1]):
        signal = mda.data[y, x, :]
        if np.any(signal > 10):
            print(x,y)
            count += 1
print(count)
# %%
print(signal)
# %%
