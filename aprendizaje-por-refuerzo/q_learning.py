import numpy as np

# 1. Inicialización de la Tabla Q
# 3 estados (filas) y 2 acciones (columnas: A y B), inicializadas a cero.
Q_tabla = np.zeros((3, 2))
alpha = 0.1  # Tasa de aprendizaje
gamma = 0.9  # Factor de descuento

# 2. Simulación de un paso de Aprendizaje
# El agente está en el Estado 1 (fila 1), toma la Acción B (columna 1) y recibe una recompensa.
estado_actual = 1
accion_tomada = 1
recompensa = 10
siguiente_estado = 2

# El agente encuentra el valor Q máximo para el siguiente estado
max_Q_siguiente = np.max(Q_tabla[siguiente_estado, :])

# 3. Actualización de la Tabla Q (Ecuación de Bellman)
# Q(s,a) = Q(s,a) + alpha * [recompensa + gamma * max(Q(s',a')) - Q(s,a)]
valor_Q_actualizado = Q_tabla[estado_actual, accion_tomada] + alpha * (
    recompensa + gamma * max_Q_siguiente - Q_tabla[estado_actual, accion_tomada]
)

# Asignar el nuevo valor
Q_tabla[estado_actual, accion_tomada] = valor_Q_actualizado

# Salida de ejemplo
print("Tabla Q inicial:\n", np.zeros((3, 2)))
print("Tabla Q después de un paso de aprendizaje:\n", np.round(Q_tabla, 2))

