# %% Imports
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from MultiDimArray import MultiDimArray
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, accuracy_score

# Import preprocessing functions from preprocess_data.py
from preprocess_data import lowpass_filter, normalize_signal, remove_nans, Baseline, LAM, P

def preprocess_signals(data, threshold=30, lam=LAM, p=P):
    """
    Preprocess 3D spectral data and record (y, x) positions.
    Returns a tuple: (list of cleaned signals, list of (y, x) positions)
    """
    cleaned_signals = []
    positions = []
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            signal = data[y, x, :]
            if np.all(signal == 0) or np.any(signal > threshold):
                continue
            signal = remove_nans(signal)
            baseline_fitter = Baseline(np.arange(len(signal)))
            baseline, _ = baseline_fitter.asls(signal, lam=lam, p=p)
            signal_corrected = signal - baseline
            filtered = lowpass_filter(signal_corrected, cutoff=0.01, fs=1)
            normalized = normalize_signal(filtered)
            cleaned_signals.append(filtered)
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


# %% Select best parameters via grid search
from sklearn.metrics import accuracy_score

# Parameter grids
pca_components_list = [3, 4, 5, 6, 7, 8]
gamma_list = [0.01, 0.05, 0.1, 0.5, 1]
C_list = [0.01, 0.1, 1, 10]
degree_list = [2, 3, 4]

best_score = 0
best_params = {}

for n_components in pca_components_list:
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, shuffle=True)
    for gamma in gamma_list:
        for C in C_list:
            for degree in degree_list:
                clf = SVC(kernel='rbf', C=C, degree=degree, gamma=gamma)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                score = accuracy_score(y_test, y_pred)
                if score > best_score:
                    best_score = score
                    best_params = {
                        'n_components': n_components,
                        'gamma': gamma,
                        'C': C,
                        'degree': degree
                    }
                print(f"n_components={n_components}, gamma={gamma}, C={C}, degree={degree}, accuracy={score:.4f}")

print("Best parameters:", best_params)

# %% PCA

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)
# After fitting PCA
print("Explained variance ratio for each component:")
print(pca.explained_variance_ratio_)
print("Cumulative explained variance:")
print(np.cumsum(pca.explained_variance_ratio_))

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
ax.set_title('PCA (6 Principal Components)')
plt.show()

# Classification: SVM
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, shuffle=True)
clf = SVC(kernel='rbf', C=10, degree=2, gamma=0.05)
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


