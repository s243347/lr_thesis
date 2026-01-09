from MultiDimArray import MultiDimArray

# Assuming your data file is "E_coli_data.bin" and metadata is "E_coli_data.json" in the folder "E. coli"
folder = "E. coli"
filename = "E_coli_data.bin"
filepath = os.path.join(folder, filename)

mda = MultiDimArray(filepath)
print(mda.data)           # Show the loaded data array
print(mda.shape)          # Show the shape of the array
print(mda.axesLabels)     # Show axis labels
print(mda.currentMap())   # Show the current map slice