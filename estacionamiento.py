import cv2
import numpy as np
import os

# Tamaño deseado para las imágenes
desired_size = (100, 100)  # Por ejemplo, 100x100 píxeles

# Ruta a la carpeta de entrenamiento
training_folder = "training"

# Listas para almacenar características y etiquetas
features = []
labels = []

# Recorrer las imágenes en la carpeta de entrenamiento
for filename in os.listdir(training_folder):
    if filename.endswith(".png"):
        # Cargar la imagen
        img_path = os.path.join(training_folder, filename)
        img = cv2.imread(img_path)

        # Redimensionar la imagen al tamaño deseado
        img_resized = cv2.resize(img, desired_size)

        # Preprocesamiento si es necesario (cambio de color, etc.)
        # ...

        # Extraer características (por ejemplo, HOG)
        hog = cv2.HOGDescriptor()
        features_vector = hog.compute(img_resized)
        features.append(features_vector.flatten())  # Añadir las características a la lista

        # Obtener la etiqueta del nombre de archivo (1.png, 2.png, etc.)
        label = int(filename.split(".")[0])
        labels.append(label)  # Añadir la etiqueta a la lista

# Convertir listas a arrays numpy
features = np.array(features)
labels = np.array(labels)

# Inicializar y entrenar el clasificador SVM
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.train(features, cv2.ml.ROW_SAMPLE, labels)

# Guardar el modelo entrenado
svm.save("svm_model.xml")
