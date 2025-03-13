# Importar las librerías necesarias
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Cargar los datos (puedes usar un dataset de clientes, por ejemplo, datos de compras)
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/customer_segmentation.csv"
data = pd.read_csv(url)

# Ver las primeras filas del dataset
print(data.head())

# Preprocesamiento de los datos
# Eliminar columnas no necesarias o que no contienen información relevante para el clustering
data_clean = data.drop(columns=["CustomerID", "Name"])

# Normalizar los datos para que todas las características estén en la misma escala
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_clean)

# Aplicar el algoritmo K-means para segmentar a los clientes en 4 grupos
kmeans = KMeans(n_clusters=4, random_state=42)
data['Cluster'] = kmeans.fit_predict(data_scaled)

# Visualización de los clusters generados por K-means (usando las dos primeras características para simplificar)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data_clean.iloc[:, 0], y=data_clean.iloc[:, 1], hue=data['Cluster'], palette='Set2', s=100, alpha=0.6)
plt.title('Customer Segmentation Using K-Means Clustering')
plt.xlabel(data_clean.columns[0])
plt.ylabel(data_clean.columns[1])
plt.legend(title='Cluster')
plt.show()

# Resumen de los clientes en cada uno de los clusters
cluster_summary = data.groupby("Cluster").mean()
print(cluster_summary)

# Opcional: Analizar la cantidad de clientes por segmento
cluster_counts = data['Cluster'].value_counts()
print(cluster_counts)

# Guardar los resultados para su análisis posterior
data.to_csv("customer_segmentation_results.csv", index=False)


""" Explicacion:
Cargar los datos:

El código carga un conjunto de datos de ejemplo que contiene información sobre los clientes. En este caso, se utiliza un conjunto de datos de segmentación de clientes que incluye detalles como el gasto en productos, frecuencia de compra, etc.
Preprocesamiento:

Se eliminan las columnas que no son relevantes para el análisis de segmentación (como el CustomerID o Name).
Se normalizan los datos con StandardScaler para asegurarnos de que todas las características estén en la misma escala, lo que es importante cuando se utiliza un algoritmo como K-means.
Aplicar K-means:

Se aplica el algoritmo de K-means clustering para segmentar a los clientes en 4 grupos (esto es arbitrario; en la práctica, puedes usar el método del codo o la silueta para determinar el número óptimo de clusters).
El modelo asigna cada cliente a un cluster, y se agrega una columna en el dataframe original llamada Cluster que contiene el número de cluster al que pertenece cada cliente.
Visualización:

Se crea un scatter plot para visualizar los clusters usando las dos primeras características de los datos (para simplificar la visualización). Los puntos se colorean según el cluster al que pertenecen, lo que ayuda a visualizar la segmentación.
Resumen de los clusters:

Se realiza un análisis descriptivo para calcular las características promedio de los clientes en cada cluster. Esto te ayuda a entender las diferencias entre los segmentos y qué los define.
Análisis de la cantidad de clientes por cluster:

Se calcula cuántos clientes hay en cada cluster para entender mejor la distribución de los segmentos.
Guardar los resultados:

Finalmente, se guarda el dataframe con la columna Cluster añadida, lo que te permite realizar un análisis posterior o compartir los resultados.
Resultado:
Visualización de Clusters: El gráfico de dispersión muestra cómo se agrupan los clientes en 4 segmentos (clusters). Esto te da una idea visual de cómo se dividen los clientes según sus comportamientos de compra.
Resumen de Clusters: El resumen te permite entender las características promedio de cada grupo de clientes.
Número de Clientes por Segmento: Te da una idea de cuántos clientes hay en cada segmento, lo cual es útil para tomar decisiones de marketing.
Key Takeaways:
Segmentación de Clientes: Utilizando K-means clustering, se pueden crear grupos de clientes con comportamientos similares, lo cual es útil para marketing personalizado.
Preprocesamiento y Escalado: Es esencial normalizar los datos antes de aplicar algoritmos de clustering.
Visualización: La visualización interactiva y clara te permite comprender mejor los segmentos y las diferencias entre ellos.

"""
