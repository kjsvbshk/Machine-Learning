import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# 1. Datos Etiquetados y No Etiquetados
X_etiquetado = np.array([1, 2, 3, 4]).reshape(-1, 1) # Pequeña cantidad
Y_etiquetado = np.array([0, 0, 1, 1])
X_no_etiquetado = np.array([5, 6, 7]).reshape(-1, 1) # Gran cantidad

# 2. Entrenamiento Inicial con Datos Etiquetados
modelo_inicial = LogisticRegression()
modelo_inicial.fit(X_etiquetado, Y_etiquetado)

# 3. Generación de Pseudoetiquetas
# El modelo inicial predice las etiquetas para los datos no etiquetados
pseudolabels = modelo_inicial.predict(X_no_etiquetado)
print(f"Pseudoetiquetas generadas: {pseudolabels}")

# 4. Autoentrenamiento (Combinación de datos)
X_nuevo = np.vstack([X_etiquetado, X_no_etiquetado])
Y_nuevo = np.concatenate([Y_etiquetado, pseudolabels])

# 5. Re-entrenamiento del Modelo con los datos expandidos
modelo_semi_supervisado = LogisticRegression()
modelo_semi_supervisado.fit(X_nuevo, Y_nuevo)

# Salida de ejemplo de la nueva clase predicha para el valor 8
prediccion_final = modelo_semi_supervisado.predict(np.array([[8]]))
print(f"Predicción final después del semi-supervisado (para el valor 8): {prediccion_final}")

# 6. Visualización del Self-Training
plt.figure(figsize=(12, 5))

# Subplot 1: Modelo Inicial
plt.subplot(1, 2, 1)
plt.scatter(X_etiquetado[Y_etiquetado == 0], [0]*sum(Y_etiquetado == 0), 
           color='blue', s=150, label='Clase 0 (Etiquetado)', marker='o')
plt.scatter(X_etiquetado[Y_etiquetado == 1], [0]*sum(Y_etiquetado == 1), 
           color='red', s=150, label='Clase 1 (Etiquetado)', marker='o')
plt.scatter(X_no_etiquetado, [0]*len(X_no_etiquetado), 
           color='gray', s=150, label='No Etiquetado', marker='s')
plt.xlabel('Valor X')
plt.title('Antes: Solo Datos Etiquetados')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yticks([])

# Subplot 2: Después del Self-Training
plt.subplot(1, 2, 2)
plt.scatter(X_etiquetado[Y_etiquetado == 0], [0]*sum(Y_etiquetado == 0), 
           color='blue', s=150, label='Clase 0 (Original)', marker='o')
plt.scatter(X_etiquetado[Y_etiquetado == 1], [0]*sum(Y_etiquetado == 1), 
           color='red', s=150, label='Clase 1 (Original)', marker='o')
plt.scatter(X_no_etiquetado[pseudolabels == 0], [0]*sum(pseudolabels == 0), 
           color='blue', s=150, label='Clase 0 (Pseudoetiqueta)', marker='s', alpha=0.6)
plt.scatter(X_no_etiquetado[pseudolabels == 1], [0]*sum(pseudolabels == 1), 
           color='red', s=150, label='Clase 1 (Pseudoetiqueta)', marker='s', alpha=0.6)
plt.xlabel('Valor X')
plt.title('Después: Con Pseudoetiquetas')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yticks([])

plt.tight_layout()
plt.show()

