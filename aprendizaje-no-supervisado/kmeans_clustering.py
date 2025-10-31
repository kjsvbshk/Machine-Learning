import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 1. Preparación de los Datos (Sin Etiquetar)
# Datos de ejemplo 2D (ej. Ingreso y Edad)
X = np.array([
    [1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]
])

# 2. Creación y Entrenamiento del Modelo
# Se eligen 2 clústeres (K=2)
modelo_no_supervisado = KMeans(n_clusters=2, n_init='auto', random_state=42)

# El modelo aprende a encontrar patrones y agrupar los datos
modelo_no_supervisado.fit(X)

# 3. Asignación de Etiquetas de Clúster
etiquetas_cluster = modelo_no_supervisado.labels_

# 4. Visualización (Opcional, para entender el resultado)
plt.scatter(X[:, 0], X[:, 1], c=etiquetas_cluster, cmap='viridis')
plt.title("K-Means Clustering (K=2)")
plt.show()

# Salida de ejemplo de las etiquetas asignadas
print(f"Etiquetas de clúster asignadas: {etiquetas_cluster}")

