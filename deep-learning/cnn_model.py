import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
import numpy as np

# 1. Definición de la Arquitectura CNN
# Se simula la entrada de una imagen a color de 32x32 píxeles
input_shape = (32, 32, 3)

modelo_cnn = Sequential([
    # Capa Convolucional: Aprende las características (bordes, texturas)
    Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    
    # Capa de MaxPooling: Reduce la dimensionalidad y hace el modelo más robusto
    MaxPooling2D((2, 2)),
    
    # Se repite el patrón de convolución y pooling
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    # Aplanamiento: Transforma la salida 2D a un vector 1D
    Flatten(),
    
    # Capa Densa (Fully Connected): Realiza la clasificación final
    Dense(10, activation='softmax') # 10 clases (ej. dígitos del 0 al 9)
])

# 2. Compilación del Modelo
# Esto prepara el modelo para el entrenamiento real
modelo_cnn.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])

# 3. Resumen de la Arquitectura
modelo_cnn.summary()

# 4. Visualización de la Arquitectura y Datos de Ejemplo
fig = plt.figure(figsize=(14, 5))

# Subplot 1: Imagen de entrada simulada
ax1 = plt.subplot(1, 3, 1)
imagen_ejemplo = np.random.rand(32, 32, 3)  # Imagen aleatoria de ejemplo
ax1.imshow(imagen_ejemplo)
ax1.set_title('Imagen de Entrada\n(32x32x3 píxeles)', fontsize=12, fontweight='bold')
ax1.axis('off')

# Subplot 2: Arquitectura de la CNN (diagrama simplificado)
ax2 = plt.subplot(1, 3, 2)
ax2.text(0.5, 0.9, 'Arquitectura CNN', ha='center', fontsize=14, fontweight='bold')
ax2.text(0.5, 0.75, '🖼️ Input: 32×32×3', ha='center', fontsize=10)
ax2.text(0.5, 0.65, '↓', ha='center', fontsize=12)
ax2.text(0.5, 0.55, '🔍 Conv2D (32 filtros 3×3)', ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightblue'))
ax2.text(0.5, 0.45, '⬇️ MaxPooling (2×2)', ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightgreen'))
ax2.text(0.5, 0.35, '🔍 Conv2D (64 filtros 3×3)', ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightblue'))
ax2.text(0.5, 0.25, '⬇️ MaxPooling (2×2)', ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightgreen'))
ax2.text(0.5, 0.15, '📊 Flatten + Dense (10)', ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightyellow'))
ax2.text(0.5, 0.05, '✅ Softmax (Clasificación)', ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightcoral'))
ax2.axis('off')

# Subplot 3: Salida - Probabilidades de clasificación simuladas
ax3 = plt.subplot(1, 3, 3)
clases = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
probabilidades = np.random.rand(10)
probabilidades = probabilidades / probabilidades.sum()  # Normalizar
colores = ['red' if p == max(probabilidades) else 'skyblue' for p in probabilidades]
ax3.barh(clases, probabilidades, color=colores)
ax3.set_xlabel('Probabilidad', fontsize=10)
ax3.set_ylabel('Clase', fontsize=10)
ax3.set_title('Salida del Modelo\n(Predicción)', fontsize=12, fontweight='bold')
ax3.set_xlim([0, 1])

plt.tight_layout()
plt.show()

