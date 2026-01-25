# %% Import packages
import time
from MultiDimArray import MultiDimArray
import matplotlib.pyplot as plt
import os
import numpy as np


filepath_ecoli = "data_invitro_rats\\Ecoli\\Ecoli_on_metal_slide_UTI89_785nm"
filepath_assymptomatic_ecoli = "data_invitro_rats\\Assymptomatic_Ecoli\\Assymptomatic_Ecoli_1_785nm"
filepath_protius = "data_invitro_rats\\Protius\\Protius_agar_plate_785nm"
filepath_saprophyticus = "data_invitro_rats\\Ssaprofyticus\\ssaprofyticus_scan_785nm"
# %% Preprocessing #01: Remove polynomial baseline

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

# Ecoli 

mda = MultiDimArray(filepath_ecoli)
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

# Assumptomatic Ecoli

mda = MultiDimArray(filepath_assymptomatic_ecoli)
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
        signal = mda.data[y, x, :]
        x = np.arange(len(signal))
        x, signal, poly_approx, residual = pol_approx_subs(signal, degree = 6)  
        plt.plot(x, residual, label=f"Y={y}, X={x}")

plt.xlabel("Wavenumber (cm$^{-1}$)")
plt.ylabel("Intensity")
plt.title("Assymptomatic E. coli w/ Poly Subtracted")
plt.show()

# Protius 

mda = MultiDimArray(filepath_protius)
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

# S. Saprofyticus

mda = MultiDimArray(filepath_saprophyticus)
wavenumbers = mda.axesCoords[2]
plt.figure(figsize=(10, 6))
for y in range(mda.shape[0]): 
    for x in range(mda.shape[1]):
        signal = mda.data[y, x, :]
        if ~ np.any(signal > 30):
            plt.plot(wavenumbers, signal, label=f"Y={y}, X={x}")
plt.xlabel("Wavenumber (cm$^{-1}$)")
plt.ylabel("Intensity")
plt.title("S. Saprofyticus")
plt.show()

plt.figure(figsize=(10, 6))
for y in range(mda.shape[0]):
    for x in range(mda.shape[1]):
        signal = mda.data[y, x, :]
        if ~ np.any(signal > 30):
            x = np.arange(len(signal))
            x, signal, poly_approx, residual = pol_approx_subs(signal, degree = 3)  
            plt.plot(x, residual, label=f"Y={y}, X={x}")
plt.xlabel("Wavenumber (cm$^{-1}$)")
plt.ylabel("Intensity")
plt.title("S. Saprofyticus w/ Poly Subtracted")
plt.show()

# %% Preprocessing #02: baseline correction
from pybaselines import Baseline

mda = MultiDimArray(filepath_protius)
wavenumbers = mda.axesCoords[2]
plt.figure(figsize=(10, 6))
for y in range(mda.shape[0]):
    for x in range(mda.shape[1]):
        signal = mda.data[y, x, :]
        if np.all(signal == 0):
            continue
        if np.any(np.isnan(signal)):
            valid_indices = np.where(~np.isnan(signal))[0]
            last_valid = valid_indices[-1]
            signal = signal[:last_valid + 1]
        plt.subplot(1,2,1)
        plt.plot(np.arange(len(signal)), signal, label=f"Y={y}, X={x}")
        baseline_fitter = Baseline(signal)
        bkg, _ = baseline_fitter.modpoly(signal, poly_order=3)
        #signal_corrected = signal - bkg
        plt.subplot(1,2,2)
        plt.plot(np.arange(len(bkg)), bkg, label=f"Y={y}, X={x}")

plt.xlabel("Wavenumber (cm$^{-1}$)")
plt.ylabel("Intensity")
plt.title("E. coli w/ Baseline Corrected")
plt.show()

# %% Preprocessing #03: Compare polynom with baseline methods
from pybaselines import Baseline

mda_ecoli = MultiDimArray(filepath_ecoli)
mda_assymptomatic_ecoli = MultiDimArray(filepath_assymptomatic_ecoli)
mda_protius = MultiDimArray(filepath_protius)
mda_saprophyticus = MultiDimArray(filepath_saprophyticus)

mdas = mda_ecoli, mda_assymptomatic_ecoli, mda_protius, mda_saprophyticus

fig, axs = plt.subplots(4, 3) 
plt.tight_layout()

for mda in mdas:
    for y in range(mda.shape[0]):
        for x in range(mda.shape[1]):
            signal = mda.data[y, x, :]
            if np.all(signal == 0) or np.any(signal > 30):
                continue
            x = np.arange(len(signal))
            x, signal, poly_approx, residual = pol_approx_subs(signal, degree = 3)  
            plt.subplot(4, 3, 3* (mdas.index(mda) + 1) - 2)
            plt.plot(x, signal, label=f"Y={y}, X={x}")
            plt.subplot(4, 3, 3* (mdas.index(mda) + 1) - 1)
            plt.plot(x, residual, label=f"Y={y}, X={x}")

            baseline_fitter = Baseline(len(signal))
            bkg, params = baseline_fitter.modpoly(signal, poly_order=3)
            signal_corrected = signal - bkg
            plt.subplot(4, 3, 3* (mdas.index(mda) + 1) )
            plt.plot(x, signal_corrected, label=f"Y={y}, X={x}")

for i in np.arange(12):
    if i < 9 :
        plt.subplot(4, 3, i+1)
        plt.xticks([])

axs[0, 0].set_title("Raw Signal", fontsize=8)
axs[0, 1].set_title("Manual Poynomial Substraction", fontsize=8)
axs[0, 2].set_title("Pybaselines Baseline Substraction", fontsize=8)

# %% Preprocessing #04: Whittaker smoothing 

from pybaselines import Baseline
mda_ecoli = MultiDimArray(filepath_ecoli)
mda_assymptomatic_ecoli = MultiDimArray(filepath_assymptomatic_ecoli)
mda_protius = MultiDimArray(filepath_protius)
mda_saprophyticus = MultiDimArray(filepath_saprophyticus)

mdas = mda_ecoli, mda_assymptomatic_ecoli, mda_protius, mda_saprophyticus

mda_names = ['E. coli', 'Assymptomatic E. coli', 'Protius', 'S. Saprofyticus']
for mda in mdas:
    avg_diffs_per_mda = []
    for y in range(mda.shape[0]):
        for x in range(mda.shape[1]):
            signal = mda.data[y, x, :]
            x = np.arange(len(signal))
            if np.any(np.isnan(signal)):
                valid_indices = np.where(~np.isnan(signal))[0]
                last_valid = valid_indices[-1]
                signal = signal[:last_valid + 1]
                x = x[:last_valid + 1]
            baseline_fitter = Baseline(x)
            lams = np.logspace(3, 9, 10)
            baselines = []
            for lam in lams:
                z, _ = baseline_fitter.asls(signal, lam=lam, p=0.01)
                baselines.append(z)
                diffs = [
                    np.linalg.norm(baselines[i+1] - baselines[i]) / np.linalg.norm(baselines[i])
                    for i in range(len(baselines)-1)
                ]
            avg_diffs_per_mda.append(diffs)

    # Plot the average differences per mda
plt.figure(figsize=(8, 6))
for idx, diffs in enumerate(avg_diffs_per_mda):
    avg = np.mean(diffs)
    plt.plot(diffs, marker='o', label=f'{mda_names[idx]} (avg={avg:.2e})')
    for i, diff in enumerate(diffs):
        plt.annotate(f'{lams[i]:.1e}', (i, diff), textcoords="offset points", xytext=(0,5), ha='center', fontsize=7)
plt.xlabel('Step')
plt.ylabel('Relative Change')
plt.title('Consecutive Baseline Changes per MDA\n(annotated with lowest $\\lambda$)')
plt.legend()
plt.tight_layout()
plt.show()

# %%

from pybaselines import Baseline

mda_ecoli = MultiDimArray(filepath_ecoli)
mda_assymptomatic_ecoli = MultiDimArray(filepath_assymptomatic_ecoli)
mda_protius = MultiDimArray(filepath_protius)
mda_saprophyticus = MultiDimArray(filepath_saprophyticus)

mdas = mda_ecoli, mda_assymptomatic_ecoli, mda_protius, mda_saprophyticus

lam = 1e6
p = 0.01

fig, axs = plt.subplots(4, 2) 
plt.tight_layout()

for mda in mdas:
    for y in range(mda.shape[0]):
        for x in range(mda.shape[1]):
            signal = mda.data[y, x, :]
            x = np.arange(len(signal))
            if np.all(signal == 0) or np.any(signal > 30):
                continue
            if np.any(np.isnan(signal)):
                valid_indices = np.where(~np.isnan(signal))[0]
                last_valid = valid_indices[-1]
                signal = signal[:last_valid + 1]
                x = x[:last_valid + 1]

            baseline_fitter = Baseline(x)
            baseline, params = baseline_fitter.asls(signal, lam=lam, p=p)
            signal_corrected = signal - baseline

            plt.subplot(4, 2, 2* (mdas.index(mda)+ 1) - 1)
            plt.plot(x, signal, label=f"Y={y}, X={x}")

            plt.subplot(4, 2, 2* (mdas.index(mda)+ 1) )
            plt.plot(x, signal_corrected, label=f"Y={y}, X={x}")

for i in np.arange(8):
    if i < 6 :
        plt.subplot(4, 2, i+1)
        plt.xticks([])

axs[0, 0].set_title("Raw Signal", fontsize=8)
axs[0, 1].set_title(f"Whittaker Smoothing lam={lam}", fontsize=8)
# %% Apply low-pass filter on top of that 

# Compute and plot mean PSD of all signals in E. coli dataset
mda = MultiDimArray(filepath_assymptomatic_ecoli)
psd_list = []
for y in range(mda.shape[0]):
    for x in range(mda.shape[1]):
        signal = mda.data[y, x, :]
        if np.any(np.isnan(signal)):
            valid_indices = np.where(~np.isnan(signal))[0]
            last_valid = valid_indices[-1]
            signal = signal[:last_valid + 1]
        x_d = signal - np.mean(signal)  # remove DC
        X = np.fft.rfft(x_d)
        P = np.abs(X)**2
        psd_list.append(P)

# Find common frequency axis (assuming all signals same length)
f = np.fft.rfftfreq(len(x_d), d=1)  # d = Δν = 1 cm⁻¹
psd_array = np.vstack(psd_list)
mean_psd = np.mean(psd_array, axis=0)

# Plot mean PSD
plt.figure(figsize=(8, 4))
plt.semilogy(f, mean_psd)
plt.xlabel('Frequency [cm$^{-1}$]')
plt.ylabel('Mean PSD [a.u.]')
plt.title('Mean Power Spectral Density (Assymptomatic E. coli)')')
plt.grid(True)
plt.tight_layout()
plt.show()
# %%

from scipy.signal import butter, filtfilt

def lowpass_filter(signal, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)



mda = MultiDimArray(filepath_ecoli)
signal = mda.data[2, 8, :]
x = np.arange(len(signal))
if np.any(np.isnan(signal)):
    valid_indices = np.where(~np.isnan(signal))[0]
    last_valid = valid_indices[-1]
    signal = signal[:last_valid + 1]
    x = x[:last_valid + 1]


# Example usage:
filtered = lowpass_filter(signal, cutoff=0.01, fs=1)

plt.figure()
plt.plot(x, signal, label='Original Signal')
plt.plot(x, filtered, label='Filtered Signal')
plt.legend()

# %% Signal normalization

def minmax_normalize(signal):
    min_val = np.min(signal)
    max_val = np.max(signal)
    if max_val == min_val:
        return signal  # Avoid division by zero
    return (signal - min_val) / (max_val - min_val)


mda_ecoli = MultiDimArray(filepath_ecoli)
mda_assymptomatic_ecoli = MultiDimArray(filepath_assymptomatic_ecoli)
mda_protius = MultiDimArray(filepath_protius)
mda_saprophyticus = MultiDimArray(filepath_saprophyticus)

mdas = mda_ecoli, mda_assymptomatic_ecoli, mda_protius, mda_saprophyticus

for mda in mdas:
    plt.figure(figsize=(10, 5))
    for y in range(mda.shape[0]):
        for x in range(mda.shape[1]):
            signal = mda.data[y, x, :]
            if np.all(signal == 0) or np.any(signal > 30):
                continue
            if np.any(np.isnan(signal)):
                valid_indices = np.where(~np.isnan(signal))[0]
                last_valid = valid_indices[-1]
                signal = signal[:last_valid + 1]
            
                baseline_fitter = Baseline(np.arange(len(signal)))
                baseline, params = baseline_fitter.asls(signal, lam=lam, p=p)
                signal_corrected = signal - baseline
                filtered = lowpass_filter(signal_corrected, cutoff=0.05, fs=1)
                normalized = minmax_normalize(filtered)

                plt.subplot(1,2,1)
                plt.plot(np.arange(len(signal)), signal, label=f"Y={y}, X={x}")
                baseline_fitter = Baseline(signal)
                bkg, _ = baseline_fitter.modpoly(signal, poly_order=3)
                #signal_corrected = signal - bkg
                plt.subplot(1,2,2)
                plt.plot(np.arange(len(normalized)), filtered, label=f"Y={y}, X={x}")
# %%
plt.figure(figsize=(6, 6))
i = 0
for mda in mdas:
    i += 1
    for y in range(mda.shape[0]):
        for x in range(mda.shape[1]):
            signal = mda.data[y, x, :]
            if np.all(signal == 0) or np.any(signal > 30):
                continue
            if np.any(np.isnan(signal)):
                valid_indices = np.where(~np.isnan(signal))[0]
                last_valid = valid_indices[-1]
                signal = signal[:last_valid + 1]
            
                baseline_fitter = Baseline(np.arange(len(signal)))
                baseline, params = baseline_fitter.asls(signal, lam=lam, p=p)
                signal_corrected = signal - baseline
                filtered = lowpass_filter(signal_corrected, cutoff=0.05, fs=1)
                normalized = minmax_normalize(filtered)

                plt.subplot(4,1,i)
                if i < 4:
                    plt.xticks([])
                plt.plot(np.arange(len(normalized)), signal, label=f"Y={y}, X={x}")
# %%
