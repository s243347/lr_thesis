# MultiDimArray Project

## Overview
The `MultiDimArray` project provides a class designed to handle multi-dimensional arrays efficiently. The `MultiDimArray` class includes various methods for initializing the array, loading and saving metadata, managing data, and performing operations such as cropping and retrieving values based on specified indices.

## Features
- **Initialization**: Create a `MultiDimArray` instance from a file or a list.
- **Metadata Management**: Load and save metadata in JSON format.
- **Data Operations**: Perform operations like cropping the array and retrieving specific values based on indices.
- **Properties**: Access various properties such as `name`, `shape`, `axesCoords`, `data`, `axesLabels`, `axesUnits`, `axisIndex`, `currentIndexes`, `plotDim`, and `mapDims`.

## Installation
To use the `MultiDimArray` class, ensure you have Python and NumPy installed. You can install NumPy using pip:

```
pip install numpy
```

## Usage
1. **Creating an Instance**: You can create an instance of `MultiDimArray` by providing a filename or by using the `from_list` class method.
   
   ```python
   import numpy as np
   from MultiDimArray import MultiDimArray

   # Example of creating from a list
   data = np.random.rand(10, 10)
   axes_coords = [np.linspace(0, 1, 10), np.linspace(0, 1, 10)]
   axes_labels = ['X', 'Y']
   axes_units = ['m', 'm']
   multi_dim_array = MultiDimArray.from_list(data, axes_coords, axes_labels, axes_units, 'example')
   ```

2. **Accessing Properties**: You can access various properties of the `MultiDimArray` instance.
   
   ```python
   print(multi_dim_array.shape)
   print(multi_dim_array.axesLabels)
   ```

3. **Saving Data**: Save the data and metadata to a file.
   
   ```python
   multi_dim_array.save_data('output_file.dat')
   ```

4. **Cropping Data**: Crop the data based on specified axis ranges.
   
   ```python
   multi_dim_array.crop((0.2, 0.8), 0)  # Crop along the first axis
   ```

## License
This project is licensed under the MIT License. See the LICENSE file for more details.