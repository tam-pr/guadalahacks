import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawings_utils # Utilidades de dibujo
mp_hands = mp.solutions.holistic # Modelo 

cap = cv2.VideoCapture(0) # Inicializar captura de video

# Checar si la cámara abrió correctamente
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

with mp_hands as model:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignorando fotograma vacío.")
            continue
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # OpenCV lee en BGR, lo convertimos a RGB para MediaPipe
        
        # C. Pasar el frame RGB al modelo para que haga la inferencia (el tracking)
        
        # D. Extraer las coordenadas y (opcionalmente) dibujar los landmarks en el frame para ver que funcione
        
        # E. Mostrar el frame en una ventana interactiva
        
        # F. Escuchar el teclado para romper el bucle (ej. si se presiona la tecla 'q')

# 4. Limpieza (Para no trabar la cámara)
# Libera la captura de video y destruye todas las ventanas