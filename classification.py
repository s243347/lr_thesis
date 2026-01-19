# %% Load MultiDimArray class and data

import time
from MultiDimArray import MultiDimArray
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

filepath_ecoli = "data_invitro_rats\\Ecoli\\Ecoli_on_metal_slide_UTI89_785nm"
mda_ecoli = MultiDimArray(filepath_ecoli)

filepath_assymptomatic_ecoli = "data_invitro_rats\\Assymptomatic_Ecoli\\Assymptomatic_Ecoli_1_785nm"
mda_assymptomatic_ecoli = MultiDimArray(filepath_assymptomatic_ecoli)

filepath = "data_invitro_rats\\Protius\\Protius_agar_plate_785nm"
mda_protius = MultiDimArray(filepath)

filepath_saprophyticus = "data_invitro_rats\\Ssaprophyticus\\S_Saprophyticus_on_metal_slide_785nm"
mda_saprophyticus = MultiDimArray(filepath_saprophyticus)       


X = []
y = []

# E. coli
X.append(mda_ecoli.data)
y += ['Ecoli'] * mda_ecoli.data.shape[2]

# Assymptomatic E. coli
X.append(mda_assymptomatic_ecoli.data)
y += ['Assymptomatic_Ecoli'] * mda_assymptomatic_ecoli.data.shape[2]
# Protius
X.append(mda_protius.data)
y += ['Protius'] * mda_protius.data.shape[2]

# Saprophyticus
X.append(mda_saprophyticus.data)
y += ['Saprophyticus'] * mda_saprophyticus.data.shape[2]
X = np.vstack(X)
y = np.array(y)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot PCA
plt.figure(figsize=(8,6))
for label in np.unique(y):
    plt.scatter(X_pca[y == label, 0], X_pca[y == label, 1], label=label)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.title('PCA of Spectra')
plt.show()

# Classification (SVM example)
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# %%
