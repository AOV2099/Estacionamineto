import cv2
from matplotlib import pyplot as plt
import numpy as np

def mostrar_contornos_convexhull(ruta_imagen):
    imagen = cv2.imread(ruta_imagen)
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    _, umbral = cv2.threshold(imagen_gris, 50, 255, cv2.THRESH_BINARY)

    contornos, _ = cv2.findContours(umbral, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)

    # Eliminar el contorno con el área más grande (imagen original)
    areas = [cv2.contourArea(c) for c in contornos]
    contornos_restantes = [c for i, c in enumerate(contornos) if i != np.argmax(areas)]

    # Dibujar los contornos y rellenarlos
    imagen_contornos = imagen.copy()
    cv2.drawContours(imagen_contornos, contornos_restantes, -1, (0, 255, 0), 2)
    cv2.fillPoly(imagen_contornos, contornos_restantes, color=(0, 255, 0))  # Relleno de contornos

    # Calcular el convex hull de cada contorno restante y rellenarlo
    for c in contornos_restantes:
        hull = cv2.convexHull(c)
        cv2.drawContours(imagen_contornos, [hull], -1, (255, 0, 0), 2)
        cv2.fillConvexPoly(imagen_contornos, hull, color=(255, 0, 0))  # Relleno de convex hull

    # Mostrar la imagen con contornos y convex hull rellenos
    plt.imshow(cv2.cvtColor(imagen_contornos, cv2.COLOR_BGR2RGB))
    plt.title('Objetos con Convex Hull Rellenos')
    plt.axis('off')
    plt.show()

mostrar_contornos_convexhull("test/1.png")
