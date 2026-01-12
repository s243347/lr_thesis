from MultiDimArray import MultiDimArray

import os

filepath = "data_invitro_rats\\Ecoli\\Ecoli_on_metal_slide_UTI89_785nm"

mda = MultiDimArray(filepath)
#print(mda.data)           # Show the loaded data array
print(mda.shape)          # Show the shape of the array
#print(mda.axesLabels)     # Show axis labels
#print(mda.currentMap())   # Show the current map slice