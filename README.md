# SignSync
### Traducción y digitalización de Lengua de Señas Mexicana en tiempo real

---

## Descripción

**SignSync** es un sistema de traducción de Lengua de Señas Mexicana (LSM) en tiempo real completamente local.

Utiliza visión por computadora y aprendizaje profundo para reconocer gestos de la mano y convertirlos en texto y voz en vivo, sin depender de APIs externas ni servicios en la nube.

El sistema está diseñado para permitir comunicación fluida entre personas usuarias de LSM y personas no hablantes mediante una arquitectura de baja latencia.

---

## Cómo funciona

El sistema captura video en tiempo real desde una cámara web usando OpenCV en un entorno local de Python (Spyder/Anaconda).

Cada frame es procesado con MediaPipe Hands, extrayendo 21 puntos de referencia por mano, que se convierten en vectores de características basados en coordenadas normalizadas.

Estas características se agrupan en ventanas temporales de 45 frames y se envían a un modelo LSTM en PyTorch entrenado para reconocer secuencias de señas.

El modelo genera predicciones probabilísticas de clases de Lengua de Señas Mexicana, filtradas mediante umbrales de confianza, warm-up y cooldown para estabilizar resultados.

Las señas detectadas se convierten en texto continuo mediante un constructor de oraciones en tiempo real, incluyendo acciones como espacio y borrar.

El texto final se envía a un frontend en React, donde se visualiza y se convierte en voz mediante síntesis de texto a voz.

---

## Features

- Traducción de lengua de señas en tiempo real  
- Interfaz web en React  
- Comunicación en vivo con backend mediante WebSockets  
- Procesamiento de cámara en backend (no en navegador)  
- Visualización de gesto, confianza y frase en tiempo real  
- Text-to-Speech en español (es-MX)  
- Modo WORDS / SPELL sincronizado  
- Sistema de cooldown y estabilización de predicciones  
- Indicador de estado del pipeline AI  
- Streaming de video desde Python a React (base64)  

---

## Tech Stack

### Backend 
- Python  
- OpenCV  
- MediaPipe (Hands)  
- PyTorch (LSTM)  
- NumPy  
- Pickle  

### Real-Time Communication
- WebSockets  
- Python custom server (`server.py`)  

### Frontend
- React  
- JavaScript (ES6+)  
- Web Speech API  
- HTML / CSS  

### System / Runtime
- Anaconda / Spyder  
- Webcam local (OpenCV)  
- Base64 frame streaming  

---

## Modelo de IA

- Arquitectura: **LSTM apilada (2 capas)**
- Hidden size: **64**
- Input size: **126 features**
- Sequence length: **45 frames**
- Tipo: **many-to-one sequence classifier**
- Framework: **PyTorch**

## Archivos

```text

SignSync/
│
├── Core/
│   ├── dynamic_lsm_model.pth        # Modelo LSTM (señas dinámicas)
│   ├── lsm_backend.py               # Pipeline principal de IA (MediaPipe + inferencia)
│   ├── lsm_model.pkl                # Modelo para modo spelling / estático
│   ├── lsm_raw_data.csv             # Dataset usado para entrenamiento
│   ├── main.py                      # Ejecución principal en tiempo real
│   │
│   └── pipeline/                    # Scripts de entrenamiento y recolección de datos
│       ├── collect_data.py
│       ├── collect_dynamic_data.py
│       ├── train_dynamic_model.py
│       └── train_model.py
│
├── UI/
│   └── guadalahacks-ui/            # Frontend en React (Vite)
│       ├── src/                    # Código fuente de la interfaz
│       ├── public/
│       ├── index.html
│       ├── package.json
│       ├── vite.config.js
│       └── README.md
│
├── back/
│   └── Repo/
│       ├── server.py               # Servidor WebSocket (Python ↔ React)
│       ├── lsm_backend.py          # Lógica del backend de IA
│       ├── dynamic_lsm_model.pth   # Modelo LSTM entrenado
│       ├── lsm_model.pkl           # Modelo estático
│       ├── requirements.txt
│       ├── build.sh
│       └── deactivate.sh
│
├── front/
│   └── index.html                  # Frontend básico / versión de prueba
│
├── README.md
└── structure.md

