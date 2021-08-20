# -*- coding: utf-8 -*-

from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
def normalizar(x, y):
    scalerX = MinMaxScaler(copy=True, feature_range=(0.15, 0.85))
    scalerX.fit(x)
    dataNormalizadoX=scalerX.transform(x)
    scalerY = MinMaxScaler(copy=True, feature_range=(0.15, 0.85))
    scalerY.fit(y)
    dataNormalizadoY=scalerY.transform(y)
    
    return scalerX,scalerY,dataNormalizadoX,dataNormalizadoY
        
def normalizarLinear(x, a, b):
     
    listMin = []
    listMax = []
    
    for column in x:
        listMin.append(min(x[column]))
        listMax.append(max(x[column]))
        x[column] = (((b - a) * ((x[column]- min(x[column])) / (max(x[column]) - min(x[column])))) + a)
    return x,listMin,listMax

def desnormalizarLinear(data, maximo, minimo, a, b):
    i=0            
    for column in data:
        data[column] = (((data[column] - a) * ((maximo[i] - minimo[i]) / (b - a))) + minimo[i])
        i=i+1
    return data
    
def desnormalizar(scaler,dataNormalizado):    
    dataDesnormalizado=scaler.inverse_transform(dataNormalizado)
    return dataDesnormalizado

def dividir(data,quantidadeX,quantidadeY):
    data=pd.DataFrame(data)
    x=data.drop(data.iloc[:, quantidadeX:(quantidadeY+quantidadeX)], axis=1)
    y=data.drop(data.iloc[:, 0:quantidadeX], axis=1)    
    return x,y

def normEstatistic(X):
    x_Norm=(X-np.mean(X))/ np.std(X)
    return x_Norm

def desnormEstatistic(X_Norm,X):
    x_Desnorm=(np.std(X)*X_Norm)+(np.mean(X))
    return x_Desnorm

