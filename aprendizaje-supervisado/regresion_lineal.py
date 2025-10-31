import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 1. Preparación de los Datos (Etiquetados)
# X: Horas de estudio (variable independiente)
X = np.array([5, 15, 25, 35, 45, 55]).reshape(-1, 1)
# Y: Puntuación obtenida (variable dependiente - la etiqueta)
Y = np.array([5, 20, 14, 32, 22, 38])

# 2. División de Datos (Entrenamiento y Prueba)
X_entrenamiento, X_prueba, Y_entrenamiento, Y_prueba = train_test_split(
    X, Y, test_size=0.3, random_state=42
)

# 3. Creación y Entrenamiento del Modelo
modelo_supervisado = LinearRegression()
# El modelo aprende a mapear X a Y usando los datos etiquetados
modelo_supervisado.fit(X_entrenamiento, Y_entrenamiento)

# 4. Predicción
prediccion = modelo_supervisado.predict(X_prueba)

# Salida de ejemplo
print(f"Predicciones: {np.round(prediccion)}")
print(f"Valores reales: {Y_prueba}")

