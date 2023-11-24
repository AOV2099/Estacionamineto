import cv2

import numpy as np
from training import *


def mostrar_contornos_con_rectangulos_ampliados(ruta_imagen, area_minima):
    imagen = cv2.imread(ruta_imagen)
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    _, umbral = cv2.threshold(imagen_gris, 50, 255, cv2.THRESH_BINARY)

    contornos, _ = cv2.findContours(umbral, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)

    # Eliminar el contorno con el área más grande (imagen original)
    areas = [cv2.contourArea(c) for c in contornos]
    contornos_restantes = [c for i, c in enumerate(contornos) if i != np.argmax(areas)]

    # Filtrar los contornos por área mínima
    contornos_filtrados = [c for c in contornos_restantes if cv2.contourArea(c) > area_minima]

    # Dibujar rectángulos ampliados alrededor de los contornos filtrados
    imagen_contornos = imagen.copy()
    for c in contornos_filtrados:
        x, y, w, h = cv2.boundingRect(c)
        ampliacion = 2  # Ajusta el valor de ampliación según sea necesario
        x -= ampliacion
        y -= ampliacion
        w += ampliacion * 2
        h += ampliacion * 2

        # Crear la región de interés (ROI)
        roi = imagen[y:y+h, x:x+w]

        # Determinar si el área pertenece a un coche o no
        result = detect_car_in_roi(roi)

        # Color del rectángulo según el resultado
        color = (0, 255, 0) if result == 1 else (0, 0, 255)
        cv2.rectangle(imagen_contornos, (x, y), (x + w, y + h), color, 2)

    # Mostrar la imagen con los rectángulos ampliados alrededor de los contornos filtrados
    plt.imshow(cv2.cvtColor(imagen_contornos, cv2.COLOR_BGR2RGB))
    plt.title(f'Rectángulos ampliados alrededor de objetos > {area_minima}')
    plt.axis('off')
    plt.show()

mostrar_contornos_con_rectangulos_ampliados("test/1.png", area_minima=80)  # Ajusta el área mínima y la ampliación según sea necesario

