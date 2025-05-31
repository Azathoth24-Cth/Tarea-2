# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import os
from fcmeans import FCM
def CargarDatos():
    """
    Carga los datos desde un archivo Excel y prepara las características y etiquetas para el modelo de aprendizaje automático.
    Returns:
        X (numpy.ndarray): Matriz de características.
        y (numpy.ndarray): Vector de etiquetas.
    """
    #se cargan los datos desde el archivo xlsx
    df = pd.read_excel('DatosConClases.xlsx', sheet_name='Hoja1')
    #Se eliminan las columnas Genero, Clase Entrega 2 y Clase Entrega 3
    df = df.drop(columns=['Genero', 'Clase Entrega 2', 'Clase Entrega 3'])
    #se toma la columa Clase Entrega 1 como la clase a predecir
    y = df['Clase Entrega 1']
    #se toma el resto de las columnas como las caracteristicas
    X = df.drop(columns=['Clase Entrega 1'])
    #se convierten las caracteristicas a un array de numpy
    X = X.to_numpy()
    #se convierten las clases a 0 y 1
    y = y.map({1: 0, 2: 1})
    #se convierten las clases a un array de numpy
    y = y.to_numpy()
    return X, y

def Punto1(X, y):
    """
    Improved K-Means clustering for classification.
    
    Parameters:
    X -- Feature matrix
    y -- True class labels
    
    Returns:
    None (plots error vs number of clusters)
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y)
    
    # Scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    #se reducen los datos a 2 dimensiones para poder graficar en 2 filas
    pca = PCA(n_components=2)
    X_reduced_test = pca.fit_transform(X_test)
    colors = np.array(['red', 'blue'])
    y_test_colored = colors[y_test]
    #primera fila
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(X_reduced_test[:, 0], X_reduced_test[:, 1], c=y_test_colored, alpha=0.5, s=50)
    plt.title('Datos de Test Reducidos a 2D')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.show()


    kmeans = KMeans(n_clusters=2,max_iter=1000, random_state=42)
    kmeans.fit(X_train)
    y_predict = kmeans.predict(X_test)
    #se reducen los datos a 2 dimensiones para poder graficar
    y_predict_colored = colors[y_predict]
    plt.subplot(1, 2, 2)
    plt.scatter(X_reduced_test[:, 0], X_reduced_test[:, 1], c=y_predict_colored, alpha=0.5, s=50)
    plt.title('Datos de Test Reducidos a 2D con K-Means Clustering')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.show()
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_predict)
    print(f'Accuracy of K-Means: {accuracy * 100:.2f}%')
def Punto2(X, y):
    """Utilice el algoritmo FC-Means para agrupar los datos de entrenamiento y encontrar los centros
    que permiten diferenciar entre las dos clases esperadas. Si considera necesario puede tener un
    número de clusters C mayor a 2, de tal manera que varios clusters se asocian a una clase esperada.
    Importante que justifique muy bien la forma como eligió el número de clusters. Estime el error
    con los datos de entrenamiento, para esto puede asignar cada dato al cluster al que tenga mayor
    grado de pertenencia. Revise los grados de pertenencia de cada dato de entrenamiento a cada
    cluster para verificar si tiene valor alto de pertenencia al cluster asignado o si hay alta
    incertidumbre, es decir si los grados de pertenencia de cada dato a todas las clases es similar."""

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y)
    # Scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    #se reducen los datos a 2 dimensiones para poder graficar en 2 filas
    pca = PCA(n_components=2)
    X_reduced_test = pca.fit_transform(X_test)
    colors = np.array(['red', 'blue'])
    y_test_colored = colors[y_test]
    #primera fila
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(X_reduced_test[:, 0], X_reduced_test[:, 1], c=y_test_colored, alpha=0.5, s=50)
    plt.title('Datos de Test Reducidos a 2D')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.show()
    # Implementación del algoritmo FC-Means
    vector_errores = []
    vector_k = np.arange(2, 11)  # Probar de 2 a 10 clusters
    for k in vector_k:
        fcm = FCM(n_clusters=k, m=2, max_iter=1000, random_state=42)
        fcm.fit(X_train)
        y_predict = fcm.predict(X_train)
        # Calcular el error como la suma de las distancias al centroide
        error = np.sum(fcm.u ** 2)  # Suma de los cuadrados de las pertenencias
        vector_errores.append(error)
    # Graficar el error vs número de clusters
    plt.figure(figsize=(10, 6))
    plt.plot(vector_k, vector_errores, marker='o')
    plt.title('Error vs Número de Clusters (FC-Means)')
    plt.xlabel('Número de Clusters')
    plt.ylabel('Error (Suma de Pertenencias al Cuadrado)')
    plt.xticks(vector_k)
    plt.grid(True)
    plt.show()
    #se elige el número de clusters como el que minimiza el error
    optimal_k = vector_k[np.argmin(vector_errores)]
    np.random.seed(42)  # Fijar la semilla para reproducibilidad
    colors = np.random.rand(optimal_k, 3)  # Generar colores aleatorios para los clusters
    print(f'Número óptimo de clusters: {optimal_k}')
    # Reentrenar el modelo con el número óptimo de clusters
    fcm = FCM(n_clusters=optimal_k, m=2, max_iter=1000, random_state=42)
    fcm.fit(X_train)
    y_predict = fcm.predict(X_test)
    #se reducen los datos a 2 dimensiones para poder graficar
    y_predict_colored = colors[y_predict]
    plt.subplot(1, 2, 2)
    plt.scatter(X_reduced_test[:, 0], X_reduced_test[:, 1], c=y_predict_colored, alpha=0.5, s=50)
    plt.title('Datos de Test Reducidos a 2D con FC-Means Clustering')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.show()
    # Calcular la precisión
    accuracy = accuracy_score(y_test, y_predict)
    print(f'Precisión del FC-Means: {accuracy * 100:.2f}%')
    return

    

X, y = CargarDatos()
#se divide el dataset en un 80% para entrenamiento y un 20% para test

#Punto1(X, y)
Punto2(X, y)