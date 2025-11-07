import numpy as np
import matplotlib.pyplot as plt

# 1. Inicialización de la Tabla Q
# 3 estados (filas) y 2 acciones (columnas: A y B), inicializadas a cero.
Q_tabla_inicial = np.zeros((3, 2))
Q_tabla = Q_tabla_inicial.copy()
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
print("Tabla Q inicial:\n", Q_tabla_inicial)
print("Tabla Q después de un paso de aprendizaje:\n", np.round(Q_tabla, 2))

# 4. Visualización de la Tabla Q
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Tabla Q Inicial
im1 = ax1.imshow(Q_tabla_inicial, cmap='Blues', aspect='auto')
ax1.set_xticks([0, 1])
ax1.set_xticklabels(['Acción A', 'Acción B'])
ax1.set_yticks([0, 1, 2])
ax1.set_yticklabels(['Estado 0', 'Estado 1', 'Estado 2'])
ax1.set_title('Tabla Q Inicial')
# Agregar valores en las celdas
for i in range(3):
    for j in range(2):
        text = ax1.text(j, i, f'{Q_tabla_inicial[i, j]:.2f}',
                       ha="center", va="center", color="black", fontsize=12)
plt.colorbar(im1, ax=ax1)

# Tabla Q Actualizada
im2 = ax2.imshow(Q_tabla, cmap='Greens', aspect='auto')
ax2.set_xticks([0, 1])
ax2.set_xticklabels(['Acción A', 'Acción B'])
ax2.set_yticks([0, 1, 2])
ax2.set_yticklabels(['Estado 0', 'Estado 1', 'Estado 2'])
ax2.set_title('Tabla Q Después de Actualización')
# Agregar valores en las celdas
for i in range(3):
    for j in range(2):
        color = "red" if (i == estado_actual and j == accion_tomada) else "black"
        text = ax2.text(j, i, f'{Q_tabla[i, j]:.2f}',
                       ha="center", va="center", color=color, fontsize=12, fontweight='bold' if color == 'red' else 'normal')
plt.colorbar(im2, ax=ax2)

plt.tight_layout()
plt.show()

