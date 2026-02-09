
"""
Preprocessing pipeline for spectral data analysis.
Steps:
1. Load data from files
2. Baseline correction (AsLS smoothing)
3. Power Spectral Density (PSD) analysis
4. Low-pass filtering
5. Signal normalization
6. Visualization
"""

import time
import os
import numpy as np
import matplotlib.pyplot as plt
from MultiDimArray import MultiDimArray
from pybaselines import Baseline
from scipy.signal import butter, filtfilt

# ---------------------- Configuration ----------------------
FILEPATHS = {
'E. coli': "data_invitro_rats\\Ecoli\\Ecoli_on_metal_slide_UTI89_785nm",
'Assymptomatic E. coli': "data_invitro_rats\\Assymptomatic_Ecoli\\Assymptomatic_Ecoli_1_785nm",
'Protius': "data_invitro_rats\\Protius\\Protius_agar_plate_785nm",
'S. Saprofyticus': "data_invitro_rats\\Ssaprofyticus\\ssaprofyticus_scan_785nm"
}
LAM = 1e6
P = 0.01

# ---------------------- Utility Functions ----------------------
def lowpass_filter(signal, cutoff, fs, order=5):
    """Apply a low-pass Butterworth filter to a signal."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)

def normalize_signal(signal):
    """Normalize a signal to zero mean and unit variance."""
    signal = np.asarray(signal)
    mean = np.mean(signal)
    std = np.std(signal)
    if std == 0:
        return signal - mean
    return (signal - mean) / std

def remove_nans(signal):
    """Trim signal to exclude trailing NaNs."""
    if np.any(np.isnan(signal)):
        valid_indices = np.where(~np.isnan(signal))[0]
        last_valid = valid_indices[-1]
        return signal[:last_valid + 1]
    return signal

# ---------------------- Processing Steps ----------------------
def load_datasets():
    """Load all datasets as MultiDimArray objects."""
    return {name: MultiDimArray(path) for name, path in FILEPATHS.items()}

def baseline_sweep(mda, lams):
    """Sweep baseline lambda values and compute relative changes."""
    avg_diffs = []
    for y in range(mda.shape[0]):
        for x in range(mda.shape[1]):
            signal = remove_nans(mda.data[y, x, :])
            x_axis = np.arange(len(signal))
            baseline_fitter = Baseline(x_axis)
            baselines = []
            for lam in lams:
                z, _ = baseline_fitter.asls(signal, lam=lam, p=0.01)
                baselines.append(z)
            diffs = [
                np.linalg.norm(baselines[i+1] - baselines[i]) / np.linalg.norm(baselines[i])
                for i in range(len(baselines)-1)
            ]
            avg_diffs.append(diffs)
    return avg_diffs

def plot_baseline_sweep(avg_diffs_per_mda, mda_names, lams):
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

def plot_raw_and_corrected(mdas, lam, p):
    fig, axs = plt.subplots(4, 2)
    plt.tight_layout()
    for idx, mda in enumerate(mdas):
        for y in range(mda.shape[0]):
            for x in range(mda.shape[1]):
                signal = mda.data[y, x, :]
                x_axis = np.arange(len(signal))
                if np.all(signal == 0) or np.any(signal > 30):
                    continue
                signal = remove_nans(signal)
                x_axis = x_axis[:len(signal)]
                baseline_fitter = Baseline(x_axis)
                baseline, _ = baseline_fitter.asls(signal, lam=lam, p=p)
                signal_corrected = signal - baseline
                plt.subplot(4, 2, 2* (idx+1) - 1)
                plt.plot(x_axis, signal, label=f"Y={y}, X={x}")
                plt.subplot(4, 2, 2* (idx+1))
                plt.plot(x_axis, signal_corrected, label=f"Y={y}, X={x}")
    for i in np.arange(8):
        if i < 6:
            plt.subplot(4, 2, i+1)
            plt.xticks([])
    axs[0, 0].set_title("Raw Signal", fontsize=8)
    axs[0, 1].set_title(f"Whittaker Smoothing lam={lam}", fontsize=8)
    plt.show()

def plot_mean_psd(mda):
    psd_list = []
    for y in range(mda.shape[0]):
        for x in range(mda.shape[1]):
            signal = remove_nans(mda.data[y, x, :])
            x_d = signal - np.mean(signal)
            X = np.fft.rfft(x_d)
            P = np.abs(X)**2
            psd_list.append(P)
    f = np.fft.rfftfreq(len(x_d), d=1)
    psd_array = np.vstack(psd_list)
    mean_psd = np.mean(psd_array, axis=0)
    plt.figure(figsize=(8, 4))
    plt.semilogy(f, mean_psd)
    plt.xlabel('Frequency [cm$^{-1}$]')
    plt.ylabel('Mean PSD [a.u.]')
    plt.title('Mean Power Spectral Density (Assymptomatic E. coli)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_lowpass_example(mda, y=2, x=8, cutoff=0.01, fs=1):
    signal = mda.data[y, x, :]
    x_axis = np.arange(len(signal))
    signal = remove_nans(signal)
    x_axis = x_axis[:len(signal)]
    filtered = lowpass_filter(signal, cutoff=cutoff, fs=fs)
    plt.figure()
    plt.plot(x_axis, signal, label='Original Signal')
    plt.plot(x_axis, filtered, label='Filtered Signal')
    plt.legend()
    plt.show()

def plot_normalized_signals(mdas, lam, p, cutoff=0.05, fs=1):
    for mda in mdas:
        plt.figure(figsize=(10, 5))
        for y in range(mda.shape[0]):
            for x in range(mda.shape[1]):
                signal = mda.data[y, x, :]
                if np.all(signal == 0) or np.any(signal > 30):
                    continue
                signal = remove_nans(signal)
                baseline_fitter = Baseline(np.arange(len(signal)))
                baseline, _ = baseline_fitter.asls(signal, lam=lam, p=p)
                signal_corrected = signal - baseline
                filtered = lowpass_filter(signal_corrected, cutoff=cutoff, fs=fs)
                normalized = normalize_signal(filtered)
                plt.subplot(1,2,1)
                plt.plot(np.arange(len(signal)), signal, label=f"Y={y}, X={x}")
                plt.subplot(1,2,2)
                plt.plot(np.arange(len(normalized)), filtered, label=f"Y={y}, X={x}")
        plt.show()

def plot_all_normalized(mdas, lam, p, cutoff=0.05, fs=1):
    plt.figure(figsize=(6, 6))
    for i, mda in enumerate(mdas):
        for y in range(mda.shape[0]):
            for x in range(mda.shape[1]):
                signal = mda.data[y, x, :]
                if np.all(signal == 0) or np.any(signal > 30):
                    continue
                signal = remove_nans(signal)
                baseline_fitter = Baseline(np.arange(len(signal)))
                baseline, _ = baseline_fitter.asls(signal, lam=lam, p=p)
                signal_corrected = signal - baseline
                filtered = lowpass_filter(signal_corrected, cutoff=cutoff, fs=fs)
                normalized = normalize_signal(filtered)
                plt.subplot(4,1,i+1)
                if i < 3:
                    plt.xticks([])
                plt.plot(np.arange(len(normalized)), normalized, label=f"Y={y}, X={x}")
    plt.show()

# ---------------------- Main Workflow ----------------------
def main():
    """Run the full preprocessing and visualization pipeline."""
    # Load data
    datasets = load_datasets()
    mda_names = list(datasets.keys())
    mdas = list(datasets.values())

    # 1. Baseline sweep and plot
    lams = np.logspace(3, 9, 10)
    avg_diffs_per_mda = [np.mean(baseline_sweep(mda, lams), axis=0) for mda in mdas]
    #plot_baseline_sweep(avg_diffs_per_mda, mda_names, lams)

    # 2. Plot raw and baseline-corrected signals
    #plot_raw_and_corrected(mdas, LAM, P)

    # 3. PSD analysis (example: Assymptomatic E. coli)
    #plot_mean_psd(datasets['Assymptomatic E. coli'])

    # 4. Low-pass filter example (E. coli)
    #plot_lowpass_example(datasets['E. coli'])

    # 5. Normalization and filtered signals
    #plot_normalized_signals(mdas, LAM, P)

    # 6. All normalized signals together
    plot_all_normalized(mdas, LAM, P)


# Run main if this script is executed directly
if __name__ == "__main__":
    main()