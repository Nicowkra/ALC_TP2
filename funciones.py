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
# FUNCIONES TP 1                                     
# =============================================================================
def inversaLU(A):
    L, U, P, cant_op = calcularLU(A)
    filas, columnas = L.shape
    Inv = np.zeros((filas, columnas))  # Inicializa una matriz de ceros
    id = np.eye(filas)  # Crea una matriz identidad

    for i in range(columnas):
        y = sc.solve_triangular(L, id[:, i], lower=True)  # Resuelve L * y = e_i
        x = sc.solve_triangular(U, y)  # Resuelve U * x = y
        Inv[:, i] = x  # Almacena la columna en Inv

    return Inv

def calcularLU(A):
    cant_op = 0
    m, n = A.shape
    Ac = np.zeros_like(A)
    Ad = A.copy()
    P = np.eye(m)
    
    if m != n:
        print('Matriz no cuadrada')
        return None, None, None, 0

    for j in range(n):
        pivote = Ad[j, j]
        if pivote == 0:
            for k in range(j + 1, m):
                if Ad[k, j] != 0:
                    Ad[[j, k]], P[[j, k]], Ac[[j, k], :j] = Ad[[k, j]], P[[k, j]], Ac[[k, j], :j]
                    break
            pivote = Ad[j, j]
            if pivote == 0:
                print('No se puede continuar: todos los pivotes en la columna son cero.')
                return None, None, None, 0

        for i in range(j + 1, m):
            k = Ad[i, j] / pivote if pivote != 0 else 0
            Ad[i] -= k * Ad[j]
            Ac[i, j] = k
            cant_op += 1

    L = np.tril(Ac, -1) + np.eye(m)
    U = Ad
    return L, U, P, cant_op

# =============================================================================
# FUNCIONES PARA CONSIGNA 4
# =============================================================================

def suma_iterativa(A, n):
    k = A.shape[0]
    sucesion = np.identity(k)
    B = A.copy()
    for i in range(n):
        sucesion += A
        A = A @ B
    return sucesion

def calcular_error_aproximacion(A, Id, n):
    inversa_real = inversaLU(Id - A)
    sucesion = suma_iterativa(A, n)
    error = np.linalg.norm(sucesion - inversa_real, ord=2)
    return error

def graficar_sucesion(A, n_values, title):
    fig, axes = plt.subplots(1, len(n_values), figsize=(15, 5))
    Id = np.eye(A.shape[0])
    for i, n in enumerate(n_values):
        sucesion = suma_iterativa(A, n)
        axes[i].imshow(sucesion, cmap="viridis")
        axes[i].set_title(f'Suma de Potencias, n={n}')
    fig.suptitle(title)
    plt.show()

def graficar_error(A, Id, n_max):
    errores = [calcular_error_aproximacion(A, Id, n) for n in range(1, n_max + 1)]
    plt.plot(range(1, n_max + 1), errores, marker='o')
    plt.xlabel('Número de términos (n)')
    plt.ylabel('Error de Aproximación')
    plt.title('Convergencia del Error de Aproximación')
    plt.yscale('log')  # Escala logarítmica para observar mejor la convergencia
    plt.show()

# =============================================================================
# EJECUCIÓN DE LAS CONSIGNAS
# =============================================================================

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

# Id para calcular las aproximaciones
#Id = np.identity(A1.shape[0])

# (a) Graficar la serie de potencias para n = 10 y n = 100
#n_values = [10, 100]
#graficar_sucesion(A1, n_values, 'Sucesión de Potencias para A1')
#graficar_sucesion(A2, n_values, 'Sucesión de Potencias para A2')

# (d) Calcular y graficar el error de aproximación para diferentes valores de n
#graficar_error(A1, Id, n_max=100)
#graficar_error(A2, Id, n_max=100)
# =============================================================================
# FUNCIONES PARA CONSIGNA 10
# =============================================================================
def centrarDatos(matriz):
    # Centramos los datos
    d, n = matriz.shape
    m = np.mean(matriz.values, axis=1)  # Asegúrate de que m sea un array de numpy

    X = matriz.values - np.tile(m.reshape((len(m), 1)), (1, n))  # Usa data.values para obtener el array de numpy
    Mcov = np.dot(X, X.T) / n  # Covariance Matrix
    return X, Mcov
