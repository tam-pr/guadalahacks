import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawings_utils # Utilidades de dibujo
mp_hands = mp.solutions.holistic # Modelo 

# 2. Captura del Entorno (OpenCV)
# Inicia la captura de video apuntando a cámara web (índice 0)

# 3. Arrancar el modelo 
# with modelo_mediapipe_configurado as modelo:
    
    # Bucle de ejecución continua
    # while la_captura_este_abierta:
        
        # A. Leer el frame actual de la cámara
        
        # B. Convertir el frame de BGR (formato por defecto de OpenCV) a RGB (formato que exige MediaPipe)
        
        # C. Pasar el frame RGB al modelo para que haga la inferencia (el tracking)
        
        # D. Extraer las coordenadas y (opcionalmente) dibujar los landmarks en el frame para ver que funcione
        
        # E. Mostrar el frame en una ventana interactiva
        
        # F. Escuchar el teclado para romper el bucle (ej. si se presiona la tecla 'q')

# 4. Limpieza (Para no trabar la cámara)
# Libera la captura de video y destruye todas las ventanas
