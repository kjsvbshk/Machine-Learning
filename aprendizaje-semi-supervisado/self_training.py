import numpy as np
from sklearn.linear_model import LogisticRegression

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

