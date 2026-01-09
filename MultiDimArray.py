# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 23:20:27 2024

@author: amkut
"""

import os
import numpy as np
import json


class MultiDimArray:
    def __init__(self, fname=None):
        if (fname):
            self.__fname = os.path.splitext(fname)
            self.__shape = None
            self.__axesCoords = None
            self.__data = None
            self.__axesLabels = None
            self.__axesUnits = None
            self.__axisIndex = None
            self.__currentIndexes = None
            self.__plotDim = None
            self.__mapDims = None
    
            self.__load_metadata()
            self.__load_data()
        
    @classmethod
    def from_list(cls, dt, axs, lbls, unts, name):
        instance = cls()
        instance.init_with_list(dt, axs, lbls, unts, name)
        return instance
    
    def init_with_list(self, dt, axs, lbls, unts, name):
        self.__data = dt
        self.__shape = list(dt.shape)
        self.__axesCoords = axs
        self.__axesLabels = lbls
        self.__axesUnits = unts
        self.__updateMultidimIndexes()
        self.__fname = name
        
    @property
    def name(self):
        return self.__fname
    
    @name.setter
    def name(self, nm):
        self.__fname = nm
    
    @property
    def shape(self):
        """MultiDimArray shape"""
        return self.__shape
    
    @property
    def axesCoords(self):
        return self.__axesCoords
    
    @property
    def data(self):
        return self.__data
    
    @data.setter
    def data(self, dt):
        if (dt.size == self.data.size):
            self.__data = dt
    
    @property
    def axesLabels(self):
        return self.__axesLabels
    
    @property
    def axesUnits(self):
        return self.__axesUnits
    
    @property
    def axisIndex(self):
        return self.__axisIndex
    
    @property
    def currentIndexes(self):
        return self.__currentIndexes
    
    @currentIndexes.setter
    def currentIndexes(self, indexArray):
        if (np.issubdtype(indexArray, int)):
            if (len(indexArray) == len(self._shape)):
                self.currentIndex = indexArray
                
    @property
    def plotDim(self):
        return self.__plotDim
    
    @plotDim.setter
    def plotDim(self, dim):
        if isinstance(dim, int):
            if (dim >= 0) and (dim < self.ndim()):
                self.__plotDim = dim
                
    @property
    def mapDims(self):
        return self.__mapDims
    
    @mapDims.setter
    def mapDims(self,dims):
        if (np.issubdtype(dims.dtype, int)):
            if len(np.unique(dims)) == 2:
                if (np.any(dims) >= 0) and (np.any(dims < self.ndim())):
                    self.__mapDims = dims 
                    
    def __save_metadata(self, fname):
        jsonData = { 'Shape'        : self.__shape,
                     'AxesCoords'   : self.__axesCoords,
                     'Labels'       : self.__axesLabels,
                     'Units'        : self.__axesUnits
            }
        if (fname[1] == 'peak'):
            spectraDimensions = 'peakDimensions'
        else:
            spectraDimensions = 'spectraDimensions'
            
        metadata = {spectraDimensions : jsonData}
        
        with open(fname[0] + ".json", 'w') as f:
            json.dump(metadata, f)
    
    def __load_metadata(self):
        with open(self.__fname[0] + ".json", 'r') as f:
            metadata = json.load(f)
            spectraDimensions = 'spectraDimensions'
            if (self.__fname[1] == 'peak'):
                spectraDimensions = 'peakDimensions'
            self.__shape = metadata[spectraDimensions]['Shape']
            self.__axesCoords = metadata[spectraDimensions]['AxesCoords']
            self.__axesLabels = metadata[spectraDimensions]['Labels']
            self.__axesUnits = metadata[spectraDimensions]['Units']
            self.__updateMultidimIndexes()
            
    def __updateMultidimIndexes(self):
        self.__axisIndex = dict(zip(self.__axesLabels, range(len(self.__axesLabels))))
        self.__currentIndexes = np.zeros(len(self.shape), int)
        if 'Wavenumber' in self.axisIndex:
            self.__plotDim = np.int32(self.axisIndex['Wavenumber'])
        else:
            self.__plotDim = np.int32(self.ndim()-1);
        self.__mapDims = np.zeros(2, int);
        
        if 'X' in self.axisIndex:
            self.__mapDims[0] = self.axisIndex['X']
        else:
            self.__mapDims[0] = 0;  
            
        if 'Y' in self.axisIndex:
            self.__mapDims[1] = self.axisIndex['Y']
        else:
            self.__mapDims[1] = self.mapDims[0] + 1;

    def __load_data(self):
        with open(self.__fname[0] + self.__fname[1], 'rb') as f:
            buf = f.read()
        if len(buf) < 4 * np.prod(self.shape):
            print("Incomplete file, adding zeros to the end")
            buf += b'\0' * (4 * np.prod(self.shape) - len(buf))
        self.__data = np.frombuffer(buf, dtype=np.single, count=np.prod(self.shape))
        self.__data = self.data.reshape(self.shape)
        
    def save_data(self, fileName):        
        with open(fileName, 'wb') as f:
            f.write(self.__data.tobytes())
            fname = os.path.splitext(fileName)
            self.__save_metadata(fname)
    
    def currentMap(self):
        ndim = self.ndim()
        index = list(slice(None) for _ in range(ndim))
        for aindex in range(ndim):
            if (aindex != self.mapDims[0]) & (aindex != self.mapDims[1]):
                index[aindex] = self.currentIndexes[aindex]
        if self.mapDims[0] < self.mapDims[1] :
            return self.data[tuple(index)]
        else:
            return np.transpose(self.data[tuple(index)])
    
    def currentSpectrum(self):
        ndim = self.ndim()
        index = list(slice(None) for _ in range(ndim))
        for aindex in range(ndim):
            if (aindex != self.plotDim):
                index[aindex] = self.currentIndexes[aindex]
        return self.data[tuple(index)]
       
    def setCurrentIndex(self, index, indexVal):
        if (isinstance(index, np.int32)):
            if (index >=0) and (index < self.ndim()):
                self.__currentIndexes[index] = indexVal
                
                             
    def ndim(self):
        return len(self.shape)
    
    def crop(self, axisRange, axisIndex):
        Wmin, Wmax = axisRange
        W = np.asanyarray(self.axesCoords[axisIndex])
        indices = np.where((W >= Wmin) & (W <= Wmax))[0]
        W_cropped = np.take(W, indices, axis=0)
        self.axesCoords[axisIndex] = W_cropped.tolist()
        self.shape[axisIndex] = len(self.axesCoords[axisIndex])
        self.__data = np.take(self.__data, indices, axis=axisIndex)

    def getValIndex(self, val, dimIndex):
        if self.shape[dimIndex] > 1:
            x = self.axesCoords[dimIndex]
            index = (np.abs(x - val)).argmin()
            return index
        else:
            return 0