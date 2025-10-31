import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

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

