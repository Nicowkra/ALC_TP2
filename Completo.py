import numpy as np
import pandas as pd
from numpy import linalg as LA
valor = 0.00000000000000001
A1 = np.array([[0.186,0.521,0.014,0.32,0.134],[0.24,0.073,0.219,0.013,0.327],[0.098,0.12,0.311,0.302,0.208],[0.173,0.03,0.133,0.14,0.074],[0.303,0.256,0.323,0.225,0.257]])
A2 = np.array([[0.186,0.521,0.014,0.32,0.134],[0.24,0.073,0.219,0.013,0.327],[0.098,0.12,0.311,0.302,0.208],[0.173,0.03,0.133,0.14,0.074],[0.003,0.256,0.323,0.225,0.257]])


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
        res,v = metodoPotencia(A,vect,250)
        vectores.append(res)
    promedio = np.mean(vectores)
    
    for j in vectores:
        a = np.sqrt((j-np.mean(vectores))**2)
        DE = DE + (pow(a,2))
    desvioEstandar = np.sqrt(DE/len(vectores))
    return promedio, desvioEstandar

def creaTabla(A1,A2):
    prom_a1 ,DE_a1 = monteCarlo(A1) 
    prom_a2 ,DE_a2 = monteCarlo(A2) 
    tabla = pd.DataFrame({"Promedio":[prom_a1,prom_a2],"Desvio Estandar":[DE_a1,DE_a2]})
    tabla.index = ["A1","A2"]
    
    
    
matriz = pd.read_excel("matriz.xlsx", sheet_name ="LAC_IOT_2011",)
Nic_col = []    
Pry_col = []
for i in range(1,41): #Crea la lista de columnas a filtrar
    Nic_col.append('NICs'+str(i))
    Pry_col.append('PRYs'+str(i))
    
Pry = matriz[matriz["Country_iso3"] == "PRY"] # Crea la tabla con filas de PRY
Nic = matriz[matriz["Country_iso3"] == "NIC"] # Crea la tabla con filas de NIC
# Crea matrices intra-regionales
Pry_int= Pry.loc[:,Pry_col] 
Nic_int = Nic.loc[:,Nic_col] 


vect_Pry = np.random.randint(0,1000,size = Pry_int.shape[0]) #Creo vector aleatorio
aval_Pry,avect_Pry = metodoPotencia(Pry_int.to_numpy(),vect_Pry,250) #Uso metodo de potencia

vect_Nic = np.random.randint(0,1000,size = Nic_int.shape[0])
aval_Nic,avect_Nic = metodoPotencia(Nic_int.to_numpy(),vect_Nic,250)

tabla = pd.DataFrame({"Pry":[aval_Pry],"Nic":[aval_Nic]})
tabla.index = ["Autovalor"]

n = Pry_int.shape[0]
Id = np.identity(n)
e = np.ones(n)
En = Id - 1/n * (np.atleast_2d(e).T @ np.atleast_2d(e))
C = ((En @ Pry_int.to_numpy()).T @ (En @ Pry_int.to_numpy()))/(40-1) 

def create(n):
     x = np.random.normal(size=n)
     x -= x.mean()
     return x / np.linalg.norm(x)


def Hotelling(A,v,k,e):
    for i in range(k):
        v_prev = v
        v = A @ v
        v = v / np.linalg.norm(v)
        res = v.T - v_prev
        if np.linalg.norm(res,2) < (1-e):
            return v
    print("No")
vect = create(n)        
vectH = Hotelling(C,vect,1000,0.0000001)
print(vectH)
vectHEstrella = vectorEstrella(vectH)
autoValH = (vectHEstrella @ C @ vectH)/ (vectHEstrella @ vectH)
print(autoValH)


C2 = C - autoValH * (vectH @ vectHEstrella)
vect2 = create(n)
vectH2 = Hotelling(C2,vect2,1000,0.0000001)
print(vectH2)
vectHEstrella2 = vectorEstrella(vectH2)
autoValH2 = (vectHEstrella2 @ C2 @ vectH2)/ (vectHEstrella2 @ vectH2)