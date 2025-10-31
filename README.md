# Modelos de Machine Learning

Este proyecto contiene ejemplos de diferentes tipos de aprendizaje automático implementados en Python.

## Estructura del Proyecto

```
machine-learning/
├── aprendizaje-supervisado/
│   └── regresion_lineal.py          # Regresión Lineal
├── aprendizaje-no-supervisado/
│   └── kmeans_clustering.py         # K-Means Clustering
├── aprendizaje-por-refuerzo/
│   └── q_learning.py                # Q-Learning Tabular
├── aprendizaje-semi-supervisado/
│   └── self_training.py             # Self-Training
├── deep-learning/
│   └── cnn_model.py                 # Red Neuronal Convolucional (CNN)
├── requirements.txt                  # Dependencias del proyecto
└── README.md                         # Este archivo
```

## Configuración del Entorno

### 1. Activar el entorno virtual

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
.\venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### 2. Instalar dependencias (ya instaladas)

```bash
pip install -r requirements.txt
```

## Descripción de los Modelos

### 1. Aprendizaje Supervisado - Regresión Lineal
**Archivo:** `aprendizaje-supervisado/regresion_lineal.py`

Predice valores continuos basándose en datos etiquetados. En este ejemplo, se predice la puntuación de un estudiante según las horas de estudio.

**Ejecutar:**
```bash
python aprendizaje-supervisado/regresion_lineal.py
```

### 2. Aprendizaje No Supervisado - K-Means Clustering
**Archivo:** `aprendizaje-no-supervisado/kmeans_clustering.py`

Agrupa datos similares sin etiquetas previas. El algoritmo encuentra patrones naturales en los datos.

**Ejecutar:**
```bash
python aprendizaje-no-supervisado/kmeans_clustering.py
```

### 3. Aprendizaje por Refuerzo - Q-Learning
**Archivo:** `aprendizaje-por-refuerzo/q_learning.py`

Simula el aprendizaje mediante interacción con el ambiente, usando una Tabla Q para aprender la política óptima.

**Ejecutar:**
```bash
python aprendizaje-por-refuerzo/q_learning.py
```

### 4. Aprendizaje Semi-Supervisado - Self-Training
**Archivo:** `aprendizaje-semi-supervisado/self_training.py`

Combina datos etiquetados y no etiquetados para mejorar el modelo, generando pseudoetiquetas.

**Ejecutar:**
```bash
python aprendizaje-semi-supervisado/self_training.py
```

### 5. Deep Learning - CNN
**Archivo:** `deep-learning/cnn_model.py`

Define una Red Neuronal Convolucional para procesamiento de imágenes.

**Ejecutar:**
```bash
python deep-learning/cnn_model.py
```

## Dependencias

- **numpy**: Operaciones numéricas y álgebra lineal
- **scikit-learn**: Algoritmos de machine learning
- **matplotlib**: Visualización de datos
- **tensorflow**: Deep learning y redes neuronales

## Notas

- El entorno virtual ya está configurado y las dependencias instaladas
- Cada modelo es independiente y puede ejecutarse por separado
- Los ejemplos son educativos y demuestran conceptos fundamentales de ML

