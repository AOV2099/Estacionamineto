# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 22:49:56 2023

@author: A_209
"""

import cv2
from matplotlib import pyplot as plt
import numpy as np

def detectar_carros_por_color(ruta_imagen):
    imagen = cv2.imread(ruta_imagen)
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(imagen, cv2.COLOR_RGB2HSV)

    # Definir rangos de colores para los carros en HSV
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])

    # Crear una máscara usando el rango de colores
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Aplicar operación de cierre para mejorar la máscara
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Encontrar contornos en la máscara
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Dibujar contornos y Convex Hulls
    for c in contours:
        # Dibujar contornos
        cv2.drawContours(imagen, [c], -1, (0, 255, 0), 2)

        # Calcular y dibujar Convex Hull
        hull = cv2.convexHull(c)
        cv2.drawContours(imagen, [hull], -1, (255, 0, 0), 2)

    # Mostrar la imagen con los contornos y Convex Hull
    plt.imshow(imagen)
    plt.title('Contornos y Convex Hull de Carros por Color')
    plt.axis('off')
    plt.show()

def segmentacion_por_umbral(ruta_imagen):
    # Cargar la imagen
    imagen = cv2.imread(ruta_imagen)
    
    # Convertir la imagen a escala de grises
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Detección de bordes con Canny
    bordes = cv2.Canny(imagen_gris, 100, 200)

    # Aplicar umbralización a la imagen original usando los bordes como máscara
    _, mascara = cv2.threshold(bordes, 50, 255, cv2.THRESH_BINARY)
    
    # Segmentación por umbralización
    resultado_segmentacion = cv2.bitwise_and(imagen, imagen, mask=mascara)

    # Mostrar la imagen original, los bordes y la segmentación resultante
    plt.title('Segmentación por Umbralización')
    plt.imshow(cv2.cvtColor(resultado_segmentacion, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
def rellenar_bordes_color_plano(ruta_imagen):
    # Cargar la imagen
    imagen = cv2.imread(ruta_imagen)
    
    # Convertir la imagen a escala de grises
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Detección de bordes con Canny
    bordes = cv2.Canny(imagen_gris, 100, 200)

    # Encontrar contornos
    contornos, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Crear una máscara para rellenar los bordes
    mask = np.zeros_like(imagen)

    # Rellenar los bordes con colores planos
    cv2.fillPoly(mask, contornos, color=(0, 255, 0))  # Cambia el color según sea necesario

    # Mostrar la imagen con los bordes rellenos
    plt.imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
    plt.title('Bordes Rellenos con Color Plano')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def cerrar_bordes_y_rellenar(ruta_imagen):
    # Cargar la imagen
   imagen = cv2.imread(ruta_imagen)
   
   # Convertir la imagen a escala de grises
   imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
   
   # Detección de bordes con Canny
   bordes = cv2.Canny(imagen_gris, 100, 200)

   # Operación de dilatación para cerrar los bordes
   kernel_dilatacion = np.ones((5, 5), np.uint8)
   bordes_dilatados = cv2.dilate(bordes, kernel_dilatacion, iterations=1)

   # Operación de erosión para reducir el tamaño de los bordes
   kernel_erosion = np.ones((5, 5), np.uint8)
   bordes_erosionados = cv2.erode(bordes_dilatados, kernel_erosion, iterations=1)

   # Aplicar umbralización a la imagen original usando los bordes como máscara
   _, mascara = cv2.threshold(bordes_erosionados, 50, 255, cv2.THRESH_BINARY)

   # Segmentación por umbralización
   resultado_segmentacion = cv2.bitwise_and(imagen, imagen, mask=mascara)

   # Mostrar la imagen resultante
   plt.imshow(cv2.cvtColor(resultado_segmentacion, cv2.COLOR_BGR2RGB))
   plt.title('Segmentación por Umbralización')
   plt.axis('off')
   plt.show()


def encontrar_contornos(ruta_imagen):
    # Cargar la imagen
    imagen = cv2.imread(ruta_imagen)
    
    # Convertir la imagen a escala de grises
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Detección de bordes con Canny
    bordes = cv2.Canny(imagen_gris, 100, 200)

    # Operación de dilatación para cerrar los bordes
    kernel_dilatacion = np.ones((5, 5), np.uint8)
    bordes_dilatados = cv2.dilate(bordes, kernel_dilatacion, iterations=1)

    # Operación de erosión para reducir el tamaño de los bordes
    kernel_erosion = np.ones((5, 5), np.uint8)
    bordes_erosionados = cv2.erode(bordes_dilatados, kernel_erosion, iterations=1)

    # Aplicar umbralización a la imagen original usando los bordes como máscara
    _, mascara = cv2.threshold(bordes_erosionados, 50, 255, cv2.THRESH_BINARY)

    # Segmentación por umbralización
    resultado_segmentacion = cv2.bitwise_and(imagen, imagen, mask=mascara)

    # Encontrar contornos en la imagen resultante
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Dibujar los contornos encontrados en la imagen original
    imagen_contornos = imagen.copy()
    cv2.drawContours(imagen_contornos, contornos, -1, (0, 255, 0), 2)

    # Mostrar la imagen con los contornos encontrados
    plt.imshow(cv2.cvtColor(imagen_contornos, cv2.COLOR_BGR2RGB))
    plt.title('Contornos encontrados')
    plt.axis('off')
    plt.show()
    
def quitar_color_predominante(ruta_imagen):
    # Cargar la imagen
    imagen = cv2.imread(ruta_imagen)
    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

    # Convertir la imagen a escala de grises
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Calcular el color más predominante en la imagen
    hist = cv2.calcHist([imagen], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    color_predominante = np.unravel_index(hist.argmax(), hist.shape)

    # Definir un rango para el color predominante
    umbral = 30
    lower_color = np.array([max(0, color_predominante[0] - umbral), max(0, color_predominante[1] - umbral),
                            max(0, color_predominante[2] - umbral)])
    upper_color = np.array([min(255, color_predominante[0] + umbral), min(255, color_predominante[1] + umbral),
                            min(255, color_predominante[2] + umbral)])

    # Crear una máscara para el color predominante
    mask = cv2.inRange(imagen, lower_color, upper_color)

    # Invertir la máscara
    mask = cv2.bitwise_not(mask)

    # Aplicar la máscara a la imagen original
    resultado = cv2.bitwise_and(imagen_rgb, imagen_rgb, mask=mask)

    # Mostrar la imagen resultante
    plt.imshow(resultado)
    plt.title('Eliminación del Color Predominante')
    plt.axis('off')
    plt.show()

#detectar_carros_por_color("test/1.png")
#segmentacion_por_umbral("test1/1.png")
#rellenar_bordes_color_plano("test/2.png")
#cerrar_bordes_y_rellenar("test/2.png")
#encontrar_contornos("test/2.png")
quitar_color_predominante("test/2.png")
