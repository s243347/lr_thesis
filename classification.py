# %% Imports

import time
from MultiDimArray import MultiDimArray
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report, confusion_matrix
from pybaselines import Baseline

def preprocess_signals(data, threshold=30):
    """
    Preprocess 3D spectral data and record (y, x) positions.
    Returns a tuple: (list of cleaned signals, list of (y, x) positions)
    """
    lam = 1e6
    p = 0.01

    cleaned_signals = []
    positions = []
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            signal = data[y, x, :]
            if np.all(signal == 0) or np.any(signal > threshold):
                continue
            if np.any(np.isnan(signal)):
                valid_indices = np.where(~np.isnan(signal))[0]
                last_valid = valid_indices[-1]
                signal = signal[:last_valid + 1]
        
            baseline_fitter = Baseline(np.arange(len(signal)))
            baseline, params = baseline_fitter.asls(signal, lam=lam, p=p)
            signal_corrected = signal - baseline

            cleaned_signals.append(signal_corrected)
            positions.append((y, x))

    return cleaned_signals, positions

# Load data
filepath_ecoli = "data_invitro_rats\\Ecoli\\Ecoli_on_metal_slide_UTI89_785nm"
mda_ecoli = MultiDimArray(filepath_ecoli)

filepath_assymptomatic_ecoli = "data_invitro_rats\\Assymptomatic_Ecoli\\Assymptomatic_Ecoli_1_785nm"
mda_assymptomatic_ecoli = MultiDimArray(filepath_assymptomatic_ecoli)

filepath_protius = "data_invitro_rats\\Protius\\Protius_agar_plate_785nm"
mda_protius = MultiDimArray(filepath_protius)

filepath_saprophyticus = "data_invitro_rats\\Ssaprofyticus\\ssaprofyticus_scan_785nm"
mda_saprophyticus = MultiDimArray(filepath_saprophyticus)

# Preprocess and collect signals and labels
all_signals = []
all_positions = []
all_labels = []

for mda, label in zip(
    [mda_ecoli, mda_assymptomatic_ecoli, mda_protius, mda_saprophyticus],
    ["Ecoli", "Assymptomatic_Ecoli", "Protius", "Saprophyticus"]):
    signals, positions = preprocess_signals(mda.data, threshold=30)
    all_signals.extend(signals)
    all_positions.extend(positions)
    all_labels.extend([label] * len(signals))


# Convert to numpy arrays for PCA and classification
X = np.array([np.asarray(s) for s in all_signals])
y = np.array(all_labels)

# %% PCA

pca = PCA(n_components=6)
X_pca = pca.fit_transform(X)

# 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
for label in np.unique(y):
    idx = y == label
    ax.scatter(X_pca[idx, 0], X_pca[idx, 1], X_pca[idx, 2], label=label)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.legend()
ax.set_title('PCA (3 Principal Components)')
plt.show()
# %%
# Classification: SVM
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, shuffle=True)
clf = SVC(kernel='rbf', C=0.1, degree=3, gamma=0.1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("split finished")
print("Kernel: rbf")
print("C: 0.1")
print("Degree: 3")
print("Gamma: 0.1")
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Plot confusion matrix

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(ax=ax, cmap='Blues', colorbar=True)
plt.title('Confusion Matrix')
plt.show()


# %%
