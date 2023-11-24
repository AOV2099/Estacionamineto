import cv2
from matplotlib import pyplot as plt
import numpy as np
import random
from training import *

def quitar_colores_predominantes(ruta_imagen):
    # Cargar la imagen
    imagen = cv2.imread(ruta_imagen)
    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

    # Convertir la imagen a escala de grises
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Calcular el primer color más predominante en la imagen
    hist = cv2.calcHist([imagen], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    color1 = np.unravel_index(hist.argmax(), hist.shape)

    # Ocultar el primer color más predominante
    umbral = 30
    lower_color1 = np.array([max(0, color1[0] - umbral), max(0, color1[1] - umbral), max(0, color1[2] - umbral)])
    upper_color1 = np.array([min(255, color1[0] + umbral), min(255, color1[1] + umbral), min(255, color1[2] + umbral)])
    mask1 = cv2.inRange(imagen, lower_color1, upper_color1)
    mask1 = cv2.bitwise_not(mask1)

    # Calcular el segundo color más predominante
    hist[mask1 != 0] = 0
    color2 = np.unravel_index(hist.argmax(), hist.shape)

    # Ocultar el segundo color más predominante
    lower_color2 = np.array([max(0, color2[0] - umbral), max(0, color2[1] - umbral), max(0, color2[2] - umbral)])
    upper_color2 = np.array([min(255, color2[0] + umbral), min(255, color2[1] + umbral), min(255, color2[2] + umbral)])
    mask2 = cv2.inRange(imagen, lower_color2, upper_color2)
    mask2 = cv2.bitwise_not(mask2)

    # Combinar las máscaras
    combined_mask = cv2.bitwise_and(mask1, mask2)

    # Aplicar la máscara combinada a la imagen original
    resultado = cv2.bitwise_and(imagen_rgb, imagen_rgb, mask=combined_mask)

    # Mostrar la imagen resultante
    plt.imshow(resultado)
    plt.title('Eliminación de Colores Predominantes')
    plt.axis('off')
    plt.show()
    
def quitar_color_predominante(ruta_imagen):
    # Cargar la imagen
    imagen = cv2.imread(ruta_imagen)

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
    resultado = cv2.bitwise_and(imagen, imagen, mask=mask)

    # Mostrar la imagen resultante
    plt.imshow(cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB))
    plt.title('Eliminación del Color Predominante')
    plt.axis('off')
    plt.show()

    return resultado

def quitar_bits_predominante(imagen_procesada, color_primario):
    # Convertir la imagen a escala de grises
    imagen_gris = cv2.cvtColor(imagen_procesada, cv2.COLOR_BGR2GRAY)

    # Calcular el segundo color más predominante en la imagen
    hist = cv2.calcHist([imagen_procesada], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    
    # Encontrar el segundo color predominante
    hist[color_primario] = 0  # Ignorar el primer color predominante
    color_predominante = np.unravel_index(hist.argmax(), hist.shape)

    # Definir un rango para el segundo color predominante
    umbral = 30
    lower_color = np.array([max(0, color_predominante[0] - umbral), max(0, color_predominante[1] - umbral),
                            max(0, color_predominante[2] - umbral)])
    upper_color = np.array([min(255, color_predominante[0] + umbral), min(255, color_predominante[1] + umbral),
                            min(255, color_predominante[2] + umbral)])

    # Crear una máscara para el segundo color predominante
    mask = cv2.inRange(imagen_procesada, lower_color, upper_color)

    # Invertir la máscara
    mask = cv2.bitwise_not(mask)

    # Aplicar la máscara a la imagen original
    resultado = cv2.bitwise_and(imagen_procesada, imagen_procesada, mask=mask)

    # Mostrar la imagen resultante
    plt.imshow(cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB))
    plt.title('Eliminación del Segundo Color Predominante')
    plt.axis('off')
    plt.show()

def quitar_bits_cercanos(imagen_procesada, color_objetivo):
    # Definir el rango de colores cercanos al color objetivo
    umbral = 80  # Umbral de tolerancia para los colores cercanos
    lower_color = np.array([max(0, color_objetivo[0] - umbral),
                            max(0, color_objetivo[1] - umbral),
                            max(0, color_objetivo[2] - umbral)])
    upper_color = np.array([min(255, color_objetivo[0] + umbral),
                            min(255, color_objetivo[1] + umbral),
                            min(255, color_objetivo[2] + umbral)])

    # Crear una máscara para los colores cercanos al objetivo
    mask = cv2.inRange(imagen_procesada, lower_color, upper_color)

    # Aplicar la máscara para eliminar bits en el rango definido
    imagen_procesada[mask != 0] = 0

    # Mostrar la imagen resultante
    plt.imshow(imagen_procesada)
    plt.title('Eliminación de Bits Cercanos al Objetivo')
    plt.axis('off')
    plt.show()
    
    return imagen_procesada

def encontrar_objetos(imagen_procesada, ruta_imagen_original):
    imagen_original = imagen = cv2.imread(ruta_imagen_original)
    # Aplicar el procesamiento para eliminar bits y encontrar contornos
    imagen_sin_bits = quitar_bits_cercanos(imagen_procesada, (122, 118, 90))  # Utiliza la función actualizada

    # Convertir la imagen procesada a escala de grises
    imagen_gris = cv2.cvtColor(imagen_sin_bits, cv2.COLOR_BGR2GRAY)

    # Encontrar contornos en la imagen
    contornos, _ = cv2.findContours(imagen_gris, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Dibujar los contornos encontrados sobre la imagen original
    imagen_contornos = imagen_procesada.copy()
    #cv2.drawContours(imagen_contornos, contornos, -1, (0, 255, 0), 2)
    
    # Calcular y dibujar el contorno convexo
    for c in contornos:
        area = cv2.contourArea(c)
        if area > 80:  # Filtrar por área mínima
            hull = cv2.convexHull(c)
            cv2.drawContours(imagen_contornos, [hull], -1, (255, 0, 0), 2)
            
            # Rellenar el contorno convexo con un color aleatorio
            color_random = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
            cv2.fillConvexPoly(imagen_contornos, hull, color_random)
            
            # Dibujar un rectángulo alrededor del contorno+
     
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(imagen_contornos,(x,y),(x+w,y+h),(0,255,0),2)
            
            # Rellenar el rectángulo con la porción correspondiente de la imagen original
            roi = imagen_original[y:y+h, x:x+w]
            imagen_contornos[y:y+h, x:x+w] = roi
            isCar = detect_car_in_roi(roi)

            if isCar:
                mask_color = (0, 255, 0, 128)  # Verde semitransparente si es un carro
            else:
                mask_color = (0, 0, 255, 128)  # Rojo semitransparente si no es un carro
            
            mask = np.zeros_like(roi, dtype=np.uint8)
            mask[:] = mask_color[:3]  # Obtener solo los tres primeros valores de color (BGR)
            mask = cv2.addWeighted(imagen_contornos[y:y + h, x:x + w], 1, mask, mask_color[3]/255, 0)
            imagen_contornos[y:y + h, x:x + w] = mask
            
    
    #kernel = np.ones((5,5),np.uint8)
    #imagen_contornos = cv2.dilate(imagen_contornos,kernel,iterations = 1)

    # Mostrar la imagen con los contornos detectados
    plt.imshow(cv2.cvtColor(imagen_contornos, cv2.COLOR_BGR2RGB))
    plt.title('Contornos de Objetos')
    plt.axis('off')
    plt.show()

    return contornos
# Procesar la imagen y obtener la imagen procesada



imagen_procesada = quitar_color_predominante("test/2.png")
quitar_bits_cercanos(imagen_procesada, (122, 118, 90))
encontrar_objetos(imagen_procesada, "test/2.png")