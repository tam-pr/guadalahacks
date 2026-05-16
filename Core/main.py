import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawings_utils # Utilidades de dibujo
mp_holistic = mp.solutions.holistic # Modelo 

cap = cv2.VideoCapture(0) # Inicializar captura de video

# Checar si la cámara abrió correctamente
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

with mp_holistic as model:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignorando fotograma vacío.")
            continue
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convertir a RGB para MediaPipe
        
        results = model.process(image_rgb) # Procesar la imagen
                
        # Extraer las coordenadas y dibujar los landmarks en el frame para ver que funcione
        
        if results.face_landmarks:
            mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        
        cv2.imshow("Detección Holistic en Tiempo Real", frame) # Mostrar el frame en una ventana interactiva
        
        # Escuchar el teclado para romper el bucle (ej. si se presiona la tecla 'q')
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break 

cap.release()  # Libera la captura de video
cv2.destroyAllWindows()  # Destruye todas las ventanas