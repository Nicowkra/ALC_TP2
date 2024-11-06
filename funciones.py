# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 10:08:27 2024

@author: soler
"""
import numpy as np
import pandas as pd
from numpy import linalg as LA
import scipy.linalg as sc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from IPython.display import display, HTML

# =============================================================================
# FUNCIONES PARA CONSIGNA 2
# =============================================================================

## creamos las matrices que nos piden
A1 = np.array([
    [0.186, 0.521, 0.014, 0.320, 0.134],
    [0.240, 0.073, 0.219, 0.013, 0.327],
    [0.098, 0.120, 0.311, 0.302, 0.208],
    [0.173, 0.030, 0.133, 0.140, 0.074],
    [0.303, 0.256, 0.323, 0.225, 0.257]
])


A2 = np.array([
    [0.186, 0.521, 0.014, 0.320, 0.134],
    [0.240, 0.073, 0.219, 0.013, 0.327],
    [0.098, 0.120, 0.311, 0.302, 0.208],
    [0.173, 0.030, 0.133, 0.140, 0.074],
    [0.003, 0.256, 0.323, 0.225, 0.257]
])

## Esta funcion calcula:
###    posicion i del vector: ||A^i||_2
def vectorDeA_n(n, matrizOriginal):
    res = []
    potenciaDeA = matrizOriginal
    for i in range(n):
        potenciaDeA = potenciaDeA @ matrizOriginal 
        norma = np.linalg.norm(potenciaDeA, ord = 2)
        res.append(norma)
    return res


##invocamos la funcion vectorDeA_n y realizamos el grafico de puntos de cada posicion i del vector 
def graficarVector(matriz, tamanio):

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(np.linspace(0,tamanio-1,tamanio), vectorDeA_n(tamanio, matriz), s=10, color='aquamarine', edgecolor='k', alpha=0.7)

    # Personalización del gráfico
    ax.set_title("Norma 2 de A1 a Potencias Sucesivas", fontsize=14, fontweight='bold')
    ax.set_xlabel("Índice de la potencia", fontsize=12)
    ax.set_ylabel("Valor de la Norma 2", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.show()

# =============================================================================
# FUNCIONES PARA CONSIGNA 3
# =============================================================================

def vectorEstrella(vector):
    # Conjugar y trasponer al mismo tiempo
    vector_hermitiano = np.conjugate(vector.T)
    return vector_hermitiano


def metodoPotencia(A, v, k):

    # Calculamos autovector asociado al autovalor de mayor modulo
    for i in range(k):
        v = A @ v
        # Lo normalizamos para que no siga creciendo y converja
        v = v / np.linalg.norm(v)

    # Calculamos autovalor de mayor modulo
    vEstrella = vectorEstrella(v)
    aval = (vEstrella @ A @ v) / (vEstrella @ v)

    return aval,v

def monteCarlo(A):
    vectores = []
    DE = 0
    for i in range(0,250,1):
        vect = np.random.randint(0,1000,size = A.shape[0])
        res,avect = metodoPotencia(A,vect,8)
        vectores.append(res)
    promedio = np.mean(vectores)
    
    for j in vectores:
        a = np.sqrt((j-np.mean(vectores))**2)
        DE = DE + (pow(a,2))
    desvioEstandar = np.sqrt(DE/len(vectores))
    return promedio, desvioEstandar

# =============================================================================
# FUNCIONES PARA CONSIGNA 10
# =============================================================================

def calculoACP(data):
    d, n = data.shape
    m = np.mean(data.values, axis=1)  # Asegúrate de que m sea un array de numpy

    X = data.values - np.tile(m.reshape((len(m), 1)), (1, n))  # Usa data.values para obtener el array de numpy
    Mcov = np.dot(X, X.T) / n  # Covariance Matrix
    
    D, V = np.linalg.eigh(Mcov)
    
    # Ordenamos los autovalores de mayor a menor
    idx = np.argsort(-D)
    D = D[idx]
    V = V[:, idx]
    
    return D, V, X, m

D, V, X, m = calculoACP(A1)