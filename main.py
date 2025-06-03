# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from fcmeans import FCM
from scipy import stats
from sklearn.metrics import f1_score, precision_score, recall_score


def SeleccionarComponentes(MatrizCaracterizacion, Etiquetas1, Etiquetas2):
    """
    Selecciona componentes con diferencias significativas entre dos grupos
    
    Args:
        MatrizCaracterizacion: Matriz de características (muestras x características)
        Etiquetas1: Etiquetas del primer grupo
        Etiquetas2: Etiquetas del segundo grupo
        
    Returns:
        MatrizCaracterizacion filtrada, índices de componentes seleccionados
        o None, None si no hay diferencias significativas
    """
    MatrizClase1 = MatrizCaracterizacion[:len(Etiquetas1)]
    MatrizClase2 = MatrizCaracterizacion[len(Etiquetas1):]
    Componentes = []
    
    for i in range(MatrizClase1.shape[1]):
        Columna_i_clase1 = MatrizClase1[:, i]
        Columna_i_clase2 = MatrizClase2[:, i]
        
        # Verificar si la columna tiene varianza cero
        if np.var(Columna_i_clase1) == 0 or np.var(Columna_i_clase2) == 0:
            continue  # Saltar esta característica
            
        try:
            # Prueba de normalidad solo si hay suficiente variabilidad
            _, p_value1 = stats.shapiro(Columna_i_clase1)
            _, p_value2 = stats.shapiro(Columna_i_clase2)
            Normal = p_value1 > 0.05 and p_value2 > 0.05
        except:
            # Si falla Shapiro (puede pasar con datos constantes), asumir no normal
            Normal = False
            
        if Normal:
            # Prueba t de Student
            _, p_value = stats.ttest_ind(Columna_i_clase1, Columna_i_clase2)
        else:
            # Prueba de Mann-Whitney
            _, p_value = stats.mannwhitneyu(Columna_i_clase1, Columna_i_clase2)
            
        if p_value < 0.05:
            Componentes.append(i)
    
    if len(Componentes) > 0:
        return MatrizCaracterizacion[:, Componentes], Componentes
    else:
        return None, None
    
def assign_clusters_to_classes(fcm_model, X, y_true):
    u = fcm_model.u
    n_clusters = fcm_model.n_clusters
    cluster_classes = []
    class_labels = np.unique(y_true)
    
    # Calcular afinidad por clase para cada cluster
    for k in range(n_clusters):
        class_scores = []
        for cls in class_labels:
            # Ponderación que considera pertenencia y balance de clases
            mask = (y_true == cls)
            class_score = np.sum(u[mask, k]) * (1 + np.sum(mask)/len(y_true))
            class_scores.append(class_score)
        
        cluster_classes.append(class_labels[np.argmax(class_scores)])
    
    # Garantizar que ambas clases estén representadas
    if len(np.unique(cluster_classes)) < len(class_labels):
        # Calcular entropía de cada cluster
        cluster_entropy = []
        for k in range(n_clusters):
            pk = u[:, k] / (np.sum(u[:, k]) + 1e-10)
            entropy = -np.sum(pk * np.log(pk + 1e-10))
            cluster_entropy.append(entropy)
        
        # Reasignar el cluster más incierto
        most_uncertain = np.argmax(cluster_entropy)
        missing_class = [c for c in class_labels if c not in cluster_classes][0]
        cluster_classes[most_uncertain] = missing_class
    
    return cluster_classes

def AnalisisDeCorrelacion(MatrizCaracterizacion):
    """
    Analiza la correlación entre las características de la matriz de caracterización.
    
    Args:
        MatrizCaracterizacion: Matriz de características (muestras x características)
        
    Returns:
        Matriz de caracteristicas sin correlación
    """
    Correlacion = np.corrcoef(MatrizCaracterizacion, rowvar=False)
    #se grafica la matriz de correlación
    plt.figure(figsize=(10, 8))
    plt.imshow(Correlacion, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.title('Matriz de Correlación')
    plt.xlabel('Características')
    plt.ylabel('Características')
    plt.show()
    # Se establece un umbral de correlación
    umbral = 0.7
    # Encontrar índices de características altamente correlacionadas
    indices_correlacionados = np.where(np.abs(Correlacion) > umbral)
    indices_a_eliminar = set()
    for i, j in zip(*indices_correlacionados):
        if i != j:  # Evitar la diagonal
            indices_a_eliminar.add(j)  # Eliminar una de las características correlacionadas
    # Eliminar las características correlacionadas
    MatrizCaracterizacion = np.delete(MatrizCaracterizacion, list(indices_a_eliminar), axis=1)
    print(f"Características eliminadas por correlación: {len(indices_a_eliminar)}")
    if MatrizCaracterizacion.shape[1] == 0:
        print("Todas las características fueron eliminadas por correlación.")
        return None
    else:
        print(f"Características restantes: {MatrizCaracterizacion.shape[1]}")
    
    return MatrizCaracterizacion

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
    #se ordena la matriz de caracteristicas y las etiquetas de acuerdo a las clases
    indices_clase1 = np.where(y == 0)[0]
    indices_clase2 = np.where(y == 1)[0]
    y_clase1 = y[indices_clase1]
    y_clase2 = y[indices_clase2]
    X = np.concatenate((X[indices_clase1], X[indices_clase2]), axis=0)
    y = np.concatenate((y[indices_clase1], y[indices_clase2]), axis=0)
    
    X, _ = SeleccionarComponentes(X, y_clase1, y_clase2)
    if X is None:
        print("No se seleccionaron componentes significativos.")
        return None, None
    
    #se analiza la correlacion entre las caracteristicas
    X = AnalisisDeCorrelacion(X)
    

    return X, y

def optimizar_parametro_modelo(model_class, param_name, valores, X, y_true, assign_clusters_to_classes, fixed_params=None):

    resultados = []

    fixed_params = fixed_params or {}

    for val in valores:
        params = {**fixed_params, param_name: val}
        try:
            modelo = model_class(**params)
            modelo.fit(X)
            cluster_classes = assign_clusters_to_classes(modelo, X, y_true)
            y_pred = np.array([cluster_classes[c] for c in modelo.predict(X)])
            
            error = np.sum([
                np.min([np.linalg.norm(x - c)**2 for c in modelo.centers])
                for x in X
            ]) / len(X)
            
            if np.isnan(error):
                print(f"NaN en error con {param_name}={val}, omitiendo.")
                continue
            
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='weighted')
            prec = precision_score(y_true, y_pred, average='weighted')
            rec = recall_score(y_true, y_pred, average='weighted')

            resultados.append({
                param_name: val,
                'modelo': modelo,
                'cluster_classes': cluster_classes,
                'error': error,
                'accuracy': acc,
                'f1_score': f1,
                'precision': prec,
                'recall': rec
            })

        except Exception as e:
            print(f"Error con {param_name}={val}: {e}")
            continue

    if not resultados:
        raise RuntimeError(f"No se pudo ajustar ningún modelo con los valores dados para '{param_name}'.")

    # Selección del mejor modelo según función de puntuación
    def score(r): return r['f1_score']# + r['accuracy']+r['recall'] - r['error']
    mejor_resultado = max(resultados, key=score)
    mejor_valor = mejor_resultado[param_name]

    # Graficar métricas vs parámetro
    xs = [r[param_name] for r in resultados]
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(xs, [r['error'] for r in resultados], 'bo-', label='Error')
    plt.plot(xs, [r['accuracy'] for r in resultados], 'go-', label='Accuracy')
    plt.axvline(x=mejor_valor, color='r', linestyle='--')
    plt.xlabel(param_name)
    plt.title('Error / Accuracy vs ' + param_name)
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(xs, [r['f1_score'] for r in resultados], 'mo-', label='F1-score')
    plt.plot(xs, [r['precision'] for r in resultados], 'co-', label='Precision')
    plt.axvline(x=mejor_valor, color='r', linestyle='--')
    plt.xlabel(param_name)
    plt.title('F1 / Precision vs ' + param_name)
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(xs, [r['recall'] for r in resultados], 'yo-', label='Recall')
    plt.axvline(x=mejor_valor, color='r', linestyle='--')
    plt.xlabel(param_name)
    plt.title('Recall vs ' + param_name)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return mejor_resultado

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
    #Se hallan los centros de los clusters
    centers = kmeans.cluster_centers_
    centers_reduced = pca.transform(centers)
    plt.subplot(1, 2, 2)
    plt.scatter(X_reduced_test[:, 0], X_reduced_test[:, 1], c=y_predict_colored, alpha=0.5, s=50)
    #Se grafican los centros de los clusters
    plt.scatter(centers_reduced[:, 0], centers_reduced[:, 1], c='black', marker='X', s=200, label='Centros de Clusters')
    plt.title('Datos de Test Reducidos a 2D con K-Means Clustering')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.show()
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_predict)
    print(f'Accuracy of K-Means: {accuracy * 100:.2f}%')
    return

def Punto2(X, y, AplicarPCA=False):
    
    # 1. Preparación de datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42, stratify=y,
        shuffle=True)
    # Escalado de características
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    if AplicarPCA:
        # Reducción de dimensionalidad para visualización
        pca = PCA(n_components=2)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        X_reduced_train = X_train
        X_reduced_test = X_test
    else:
        # Reducción de dimensionalidad para visualización
        pca = PCA(n_components=2)
        X_reduced_train = pca.fit_transform(X_train)
        X_reduced_test = pca.transform(X_test)
        
    # Visualización de datos originales
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(X_reduced_test[:, 0], X_reduced_test[:, 1], c=y_test, cmap='viridis', alpha=0.6)
    plt.title('Datos Originales (Test)')
    plt.colorbar(label='Clase')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.grid(True)
    plt.show()
    #se seleciona el mejor número de clusters
    optimal_k = optimizar_parametro_modelo(
        FCM, 'n_clusters',
        range(1, 20,1), X_train, y_train, assign_clusters_to_classes,
        fixed_params={'m': 2, 'max_iter': 500, 'random_state': 42}
    )
    optimal_k = optimal_k['n_clusters']
    optimal_m = optimizar_parametro_modelo(
        FCM, 'm',
        np.arange(1.1, 3.1, 0.1),
        X_train, y_train, assign_clusters_to_classes,
        fixed_params={'n_clusters': optimal_k, 'max_iter': 500, 'random_state': 42}
    )
    optimal_m = optimal_m['m']
    # 6. Entrenamiento final con los mejores parámetros (k y m)
    final_fcm = FCM(n_clusters=optimal_k, m=optimal_m, max_iter=1000, random_state=42)
    final_fcm.fit(X_train)
    final_cluster_classes = assign_clusters_to_classes(final_fcm, X_train, y_train)
    
    y_test_pred = np.array([final_cluster_classes[c] for c in final_fcm.predict(X_test)])
    
    # Cálculo de todas las métricas en test
    test_error = np.sum([np.min([np.linalg.norm(x - c)**2 for c in final_fcm.centers]) for x in X_test]) / len(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    test_precision = precision_score(y_test, y_test_pred, average='weighted')
    test_recall = recall_score(y_test, y_test_pred, average='weighted')

    # 7. Visualización final (se mantiene igual, pero con más métricas en el título)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_reduced_test[:, 0], X_reduced_test[:, 1], 
                         c=y_test_pred, cmap='viridis', alpha=0.6)
    if AplicarPCA==False:
        centers_reduced = pca.transform(final_fcm.centers)
    else:
        centers_reduced = final_fcm.centers
    plt.scatter(centers_reduced[:, 0], centers_reduced[:, 1], 
                marker='X', s=1, c='red', label='Centroides')
    plt.title(f'Resultado Final (k={optimal_k}, m={optimal_m:.2f})\nAccuracy: {test_accuracy:.2%} | F1: {test_f1:.2f}\nPrecision: {test_precision:.2f} | Recall: {test_recall:.2f}')
    plt.colorbar(scatter, label='Clase Predicha')
    plt.legend()
    plt.show()
    
    print("\nMétricas en Test:")
    print(f"- Error: {test_error:.4f}")
    print(f"- Accuracy: {test_accuracy:.2%}")
    print(f"- F1-score: {test_f1:.4f}")
    print(f"- Precision: {test_precision:.4f}")
    print(f"- Recall: {test_recall:.4f}")
    print(f"\nAsignación clusters-clases: {final_cluster_classes}")
    
    return final_fcm, final_cluster_classes
    

X, y = CargarDatos()
#se divide el dataset en un 80% para entrenamiento y un 20% para test

#Punto1(X, y)
Punto2(X, y)
Punto2(X, y, AplicarPCA=True)