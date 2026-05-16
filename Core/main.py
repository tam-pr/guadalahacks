import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils # Utilidades de dibujo
mp_holistic = mp.solutions.holistic # Modelo 

cap = cv2.VideoCapture(0) # Inicializar captura de video

# Checar si la cámara abrió correctamente
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignorando fotograma vacío.")
            continue
        
        # Para mejorar el rendimiento, se marca la imagen como no escribible
        frame.flags.writeable = False 
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convertir a RGB para MediaPipe
        
        results = holistic.process(image_rgb) # Procesar la imagen
        
        # Volver a marcar la imagen como escribible para dibujar sobre ella
        frame.flags.writeable = True 

                
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
        
        # Escuchar el teclado para romper el bucle
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break 

cap.release()  # Libera la captura de video
cv2.destroyAllWindows()  # Destruye todas las ventanas