import numpy as np
import pandas as pd
from numpy import linalg as LA
valor = 1e-10
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
    
    
    
matriz = pd.read_excel("C:/Users/nicow/Desktop/Facultad/ALC/ALC_TP2/matriz.xlsx", sheet_name ="LAC_IOT_2011",)
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
En = Id - 1/n * np.ones((n,n))
C = ((En @ Pry_int.to_numpy()).T @ (En @ Pry_int.to_numpy()))/(40-1) 

def create(n):
     x = np.random.normal(size=n)
     x -= x.mean()
     return x / np.linalg.norm(x)


def Hotelling(A,v,valor,c):
    i = 0
    while True:
        i += 1
        v_prev = v
        vEstrella_prev = vectorEstrella(v_prev)
        aval_prev = (vEstrella_prev @ A @ v_prev) / (vEstrella_prev @ v_prev)
        
        v = (A @ v)/ np.linalg.norm(A @ v)

        vEstrella = vectorEstrella(v)
        aval = (vEstrella @ A @ v) / (vEstrella @ v)
        res = vEstrella - v_prev

        
        if (aval >=0) and (np.linalg.norm(res,2) < valor):
            return aval,v 
            
        elif (aval < 0) and ((abs(aval)-abs(aval_prev)) < valor):

            return aval,v
            
vect = create(n)        
avalH, vectH = Hotelling(C,vect,valor,"H1")
eival, eivect = LA.eig(C)
print(vectH)
print(eivect[:,0])
print(avalH)
print(eival[0])
tabla = pd.DataFrame({"Aproximado":[vectH,avalH],"Real":[eivect[:,0],eival[0]]})
tabla.index = ["Autovector","Autovalor"]
tabla

print("------------------")
vectHEstrella = vectorEstrella(vectH)
C2 = C - avalH * (vectH @ vectHEstrella)
vect2 = create(n)
avalH2, vectH2 = Hotelling(C2,vect2,valor,"H2")
eival2,eivect2 = LA.eig(C2)
print(vectH2)
print(eivect2[:,0])
vectHEstrella2 = vectorEstrella(vectH2)
autoValH2 = (vectHEstrella2 @ C2 @ vectH2)/ (vectHEstrella2 @ vectH2)
print(autoValH2)
print(eival2[0])