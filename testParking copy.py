import cv2
from matplotlib import pyplot as plt

def mostrar_imagen_gris(ruta_imagen):
    # Cargar la imagen
    imagen = cv2.imread(ruta_imagen)

    # Convertir la imagen a escala de grises
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Mostrar la imagen en escala de grises
    plt.imshow(imagen_gris, cmap='gray')
    plt.title('Imagen en Escala de Grises')
    plt.axis('off')  # Ocultar ejes
    plt.show()


def mostrar_contornos(ruta_imagen):
    # Cargar la imagen
    imagen = cv2.imread(ruta_imagen)

    # Convertir la imagen a escala de grises
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Aplicar un umbral para obtener una imagen binaria
    _, umbral = cv2.threshold(imagen_gris, 50, 255, cv2.THRESH_BINARY)

    # Encontrar contornos en la imagen binaria
    #contornos, _ = cv2.findContours(umbral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #contornos, _ = cv2.findContours(umbral, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    contornos, hierarchy = cv2.findContours(umbral,cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)
    boxArea = []


    # Dibujar los contornos en la imagen original
    imagen_contornos = imagen.copy()
    cv2.drawContours(imagen_contornos, contornos, -1, (0, 255, 0), 2)

    # Mostrar la imagen con los contornos
    plt.imshow(cv2.cvtColor(imagen_contornos, cv2.COLOR_BGR2RGB))
    plt.title('Objetos Detectados')
    plt.axis('off')  # Ocultar ejes
    plt.show()


def mostrar_bordes_canny(ruta_imagen):
    imagen = cv2.imread(ruta_imagen)
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Aplicar suavizado para reducir el ruido
    imagen_suavizada = cv2.GaussianBlur(imagen_gris, (5, 5), 0)

    # Aplicar Canny Edge Detection
    bordes = cv2.Canny(imagen_suavizada, 50, 150)

    # Mostrar la imagen con los bordes detectados
    plt.imshow(bordes, cmap='gray')
    plt.title('Bordes detectados con Canny Edge Detection')
    plt.axis('off')
    plt.show()




mostrar_imagen_gris("test/2.png")
mostrar_contornos("test/2.png")
mostrar_bordes_canny("test/2.png")