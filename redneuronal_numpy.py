from scipy import stats
import numpy as np
import math
import matplotlib.pyplot as plt
import random

sigmoid = (
    lambda x: 1 / (1 + np.exp(-x)),
    lambda x: x * (1 - x)
)

def derivada_relu(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x

relu = (
    lambda x: x * (x > 0),
    lambda x: derivada_relu(x)
)

def circulo(num_datos=100, R=1, minimo=0, maximo=1, latitud=0, longitud=0):
    pi = np.pi

    r = R * np.sqrt(stats.truncnorm.rvs(minimo, maximo, size=num_datos)) * 10
    theta = stats.truncnorm.rvs(minimo, maximo, size=num_datos) * 2 * pi * 10

    x = np.cos(theta) * r
    y = np.sin(theta) * r

    x = np.round(x + longitud, 3)
    y = np.round(y + latitud, 3)

    df = np.column_stack([x, y])
    return df

class capa():
    def __init__(self, n_neuronas_capa_anterior, n_neuronas, funcion_act):
        self.funcion_act = funcion_act
        self.b = np.round(np.random.rand(1, n_neuronas) * 2 - 1, 3)
        self.W = np.round(np.random.rand(n_neuronas_capa_anterior, n_neuronas) * 2 - 1, 3)

def mse(Ypredich, Yreal):
    x = (np.array(Ypredich) - np.array(Yreal)) ** 2
    x = np.mean(x)
    y = np.array(Ypredich) - np.array(Yreal)
    return (x, y)

def entrenamiento(epoch, X, Y, red_neuronal, lr=0.01):
    output = [X]
    for num_capa in range(len(red_neuronal)):
        z = output[-1] @ red_neuronal[num_capa].W + red_neuronal[num_capa].b
        a = red_neuronal[num_capa].funcion_act[0](z)
        output.append(a)

    back = list(range(len(output)-1))
    back.reverse()
    delta = []

    for capa in back:
        a = output[capa+1]

        if capa == back[0]:
            x = mse(a, Y)[1] * red_neuronal[capa].funcion_act[1](a)
            delta.append(x)
        else:
            x = delta[-1] @ W_temp * red_neuronal[capa].funcion_act[1](a)
            delta.append(x)

        W_temp = red_neuronal[capa].W.transpose()
        red_neuronal[capa].b = red_neuronal[capa].b - np.mean(delta[-1], axis=0, keepdims=True) * lr
        red_neuronal[capa].W = red_neuronal[capa].W - output[capa].transpose() @ delta[-1] * lr

    return output[-1]

N = 250

datos_brasilia = circulo(num_datos=N, R=1.5, latitud=-15.7801, longitud=-47.9292)
datos_kazajistan = circulo(num_datos=N, R=1, latitud=48.0196, longitud=66.9237)
X = np.concatenate([datos_brasilia, datos_kazajistan])
X = np.round(X, 3)
print ("x", X)

Y = [0] * N + [1] * N
Y = np.array(Y).reshape(len(Y), 1)
print(Y)

neuronas = [2, 4, 8, 1]
funciones_activacion = [relu, relu, sigmoid]
red_neuronal = []

for paso in list(range(len(neuronas) - 1)):
    x = capa(neuronas[paso], neuronas[paso + 1], funciones_activacion[paso])
    red_neuronal.append(x)

error = []
predicciones = []

for epoch in range(0, 1000):
    ronda = entrenamiento(epoch, X=X, Y=Y, red_neuronal=red_neuronal, lr=0.001)
    predicciones.append(ronda)
    temp = mse(np.round(predicciones[-1]), Y)[0]
    error.append(temp)
    print(f'Época {epoch}:')
    

print('=== Y1 ===')
print(np.round(predicciones[-1][0:N]))
print('=== Y2 ===')
print(np.round(predicciones[-1][N:N * 2]))

#    if epoch % 100 == 0:
 #       print(f'Época {epoch}:')
        #print('=== Y1 ===')
        #print(np.round(predicciones[-1][0:N]))
        #print('=== Y2 ===')
        #print(np.round(predicciones[-1][N:N * 2]))
        #print('=== Capa 1 ===')
        #print('W:')
        #print(red_neuronal[0].W)
        #print('b:')
        #print(red_neuronal[0].b)
        #print('=== Capa 2 ===')
        #print('W:')
        #print(red_neuronal[-2].W)
        #print('b:')
        #print(red_neuronal[-2].b)
        #print('=== Capa 3 ===')
        #print('W:')
        #print(red_neuronal[-1].W)
        #print('b:')
        #print(red_neuronal[-1].b)
        #print('------------------------')