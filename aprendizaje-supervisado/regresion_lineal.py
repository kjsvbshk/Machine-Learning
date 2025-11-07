import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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

# 5. Visualización de la Regresión Lineal
plt.figure(figsize=(10, 6))
plt.scatter(X_entrenamiento, Y_entrenamiento, color='blue', label='Datos de Entrenamiento', s=100)
plt.scatter(X_prueba, Y_prueba, color='green', label='Datos de Prueba', s=100)
plt.scatter(X_prueba, prediccion, color='red', marker='x', s=100, label='Predicciones')

# Línea de regresión
X_linea = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
Y_linea = modelo_supervisado.predict(X_linea)
plt.plot(X_linea, Y_linea, color='red', linestyle='--', linewidth=2, label='Línea de Regresión')

plt.xlabel('Horas de Estudio')
plt.ylabel('Puntuación')
plt.title('Regresión Lineal: Horas de Estudio vs Puntuación')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

