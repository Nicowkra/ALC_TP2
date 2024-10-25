# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 10:08:27 2024

@author: soler
"""
import numpy as np
import scipy

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


def vectorEstrella(vector):
    # Conjugar y trasponer al mismo tiempo
    vector_hermitiano = np.conjugate(vector_columna.T)
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

    return aval, v


def metodoPotenciaAutovalor(A, q, k):  # segun visto en la teorica
    for i in range(k):
        z_k = A @ q(k-1)
        q_k = = z_k / scipy.linalg.norm(z_k, ord=2)
        lamda_k = vector*(q_k) @ A @ q_k


def vectorConNormaDePotencias(matriz, n):
    vector = []
    for i in range(n):
        aux = metodoPotencia(matriz, i)
        norma = scipy.linalg.norm(matriz, ord=2)
        vector.append(norma)
    return vector



