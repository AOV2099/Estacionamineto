import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Tamaño deseado para las imágenes
desired_size = (100, 100)  # Por ejemplo, 100x100 píxeles

# Ruta a la carpeta de entrenamiento
training_folder = "training"

# Listas para almacenar características y etiquetas
features = []
labels = []

def get_hog():
    winSize = (20,20)
    blockSize=(8,8)
    blockStride = (4,4)
    cellSize=(8,8)
    nbins = 9
    derivAperture = 1
    winSigma = 2.
    histrogramType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlavels = 64
    signedGradient = True 
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histrogramType,L2HysThreshold,gammaCorrection,nlavels,signedGradient)
    return hog

def train_svm():
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
            hog = get_hog()
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

def detect_car(image_path):
    # Cargar el modelo SVM entrenado
    svm = cv2.ml.SVM_load("svm_model.xml")

    # Cargar la imagen a evaluar
    img = cv2.imread(image_path)

    # Redimensionar la imagen al tamaño deseado
    img_resized = cv2.resize(img, desired_size)

    # Extraer características (por ejemplo, HOG)
    hog = get_hog()
    features_vector = hog.compute(img_resized)

    # Predecir si la imagen es un coche o no
    _, result = svm.predict(features_vector.reshape(1, -1))

    return result

def train_knn():
    # Recorrer las imágenes en la carpeta de entrenamiento
    features = []
    labels = []

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
            hog = get_hog()
            features_vector = hog.compute(img_resized)
            features.append(features_vector.flatten())  # Añadir las características a la lista

            # Obtener la etiqueta del nombre de archivo (1.png, 2.png, etc.)
            label = int(filename.split(".")[0])
            labels.append(label)  # Añadir la etiqueta a la lista

    # Convertir listas a arrays numpy
    features = np.array(features)
    labels = np.array(labels)

    # Inicializar y entrenar el clasificador KNN
    knn = cv2.ml.KNearest_create()
    knn.train(features, cv2.ml.ROW_SAMPLE, labels)

    # Guardar el modelo entrenado (opcional)
    knn.save("knn_model.xml")

def detect_car_with_knn(image_path):
    # Cargar el modelo KNN entrenado
    knn = cv2.ml.KNearest_load("knn_model.xml")

    # Cargar la imagen a evaluar
    img = cv2.imread(image_path)

    # Redimensionar la imagen al tamaño deseado
    img_resized = cv2.resize(img, desired_size)

    # Extraer características (por ejemplo, HOG)
    hog = get_hog()
    features_vector = hog.compute(img_resized)

    # Realizar la predicción
    _, result, _, _ = knn.findNearest(features_vector.reshape(1, -1), k=1)

    return result

def detect_car_in_roi(roi):
    result = False
    try:
        # Cargar el modelo KNN entrenado con Sklearn
        import joblib

        # Cargar el modelo entrenado
        knn = joblib.load('knn_model_sklearn.joblib')

        # Redimensionar la región de interés (ROI) al tamaño deseado
        desired_size = (100, 100)  # Tamaño deseado para las imágenes
        img_resized = cv2.resize(roi, desired_size)

        # Preprocesamiento si es necesario (cambio de color, etc.)
        # ...

        # Extraer características (por ejemplo, HOG)
        hog = get_hog()
        features_vector = hog.compute(img_resized).flatten()

        # Realizar la predicción con el modelo KNN
        result = knn.predict([features_vector])  
    except:
        pass

    return result

def visualize_hog(image_path):
    # Cargar la imagen
    img = cv2.imread(image_path)

    # Redimensionar la imagen al tamaño deseado
    img_resized = cv2.resize(img, desired_size)

    # Extraer características (por ejemplo, HOG)
    hog = get_hog()
    features_vector = hog.compute(img_resized)

    # Visualizar las características del HOG
    plt.figure(figsize=(10, 5))
    plt.title('Histogram of Oriented Gradients (HOG)')
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Value')
    
    # Trazar las características del HOG
    plt.plot(features_vector.flatten())

    plt.show()


def train_knn_sklearn():
    # Recorrer las imágenes en la carpeta de entrenamiento
    features = []
    labels = []

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
            hog = get_hog()
            features_vector = hog.compute(img_resized)
            features.append(features_vector.flatten())  # Añadir las características a la lista

            # Obtener la etiqueta del nombre de archivo (1.png, 2.png, etc.)
            label = int(filename.split(".")[0])
            labels.append(label)  # Añadir la etiqueta a la lista

    # Convertir listas a arrays numpy
    features = np.array(features)
    labels = np.array(labels)

    # Inicializar y entrenar el clasificador KNN de Sklearn
    knn = KNeighborsClassifier(n_neighbors=30)  # Puedes ajustar el número de vecinos
    knn.fit(features, labels)

    # Guardar el modelo entrenado (opcional)
    import joblib
    joblib.dump(knn, 'knn_model_sklearn.joblib')

# Llamar a la función para entrenar el clasificador KNN con Sklearn
train_knn_sklearn()

# Entrenar el clasificador KNN
#train_knn()

# Entrenar el clasificador SVM
#train_svm()

# Ejemplo de uso de la función detect_car
#result = detect_car_with_knn("data/6.png")
#print("Es un coche" if result == 1 else "No es un coche")
#visualize_hog_with_borders("test/1.png")