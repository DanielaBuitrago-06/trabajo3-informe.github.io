---
title: "Visión por Computador II – Proyecto Final"
layout: default
nav_order: 1
---

# **Proyecto Final — Clasificación de Neumonía en Rayos X usando Descriptores Clásicos**

**Curso:** Visión por Computador II – 3009228  
**Semestre:** 2025-02  
**Facultad de Minas, Universidad Nacional de Colombia**  
**Departamento de Ciencias de la Computación y de la Decisión**

---

## **Descripción del proyecto**

Este proyecto implementa un sistema completo para la clasificación de neumonía en imágenes de rayos X de tórax utilizando descriptores clásicos de forma y textura, junto con algoritmos de machine learning tradicionales y deep learning.

El proyecto está dividido en tres partes principales:

1. **Análisis y Preprocesamiento**: Realiza análisis exploratorio del dataset de rayos X, visualiza la distribución de clases y dimensiones, e implementa un pipeline de preprocesamiento con normalización de tamaño y ecualización de contraste (CLAHE).

2. **Extracción de Descriptores**: Extrae descriptores clásicos de forma y textura:
   - **Forma**: HOG, Momentos de Hu, Descriptores de Contorno, Descriptores de Fourier
   - **Textura**: LBP, GLCM, Filtros de Gabor, Estadísticas de Primer Orden

3. **Clasificación**: Implementa y compara múltiples algoritmos:
   - **Métodos Clásicos**: SVM (Linear, RBF, Polynomial), Random Forest, k-NN, Regresión Logística
   - **Deep Learning**: CNN con PyTorch

### **Dataset**

El proyecto utiliza el dataset [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) de Kaggle, que contiene:
- **5,840 imágenes** de rayos X de tórax
- **2 clases**: NORMAL (1,575) y PNEUMONIA (4,265)
- División en train/test/val

---

## **Acceso al informe completo**
👉 [Ver Informe Final]({{ site.baseurl }}/informe.html)

---

## **Tecnologías y Herramientas**

- **Python 3.10+**
- **OpenCV**: Procesamiento de imágenes
- **scikit-image**: Extracción de descriptores (HOG, LBP, GLCM, Gabor)
- **scikit-learn**: Algoritmos de machine learning
- **PyTorch**: Redes neuronales convolucionales
- **Jupyter Notebooks**: Análisis interactivo

## **Resultados Principales**

El sistema logra:
- Extracción de **26,338 características** por imagen
- Comparación de **6 algoritmos** de clasificación diferentes
- Evaluación mediante métricas: Accuracy, Precision, Recall, F1-Score, ROC AUC
- Validación cruzada para robustez de resultados

## **Créditos**

**Desarrollado por:** Daniela Buitrago  
**Curso:** Visión por Computador II – 3009228  
**Universidad Nacional de Colombia – Facultad de Minas (2025-02)**
