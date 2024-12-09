import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Cargar el modelo previamente entrenado
model = load_model('modelo5.keras')

# Etiquetas de las clases
class_labels = ['Apple', 'Banana', 'avocado', 'cherry', 'kiwi', 'mango', 'orange', 'pineapple', 'strawberries', 'watermelon']

# Función para preprocesar la imagen antes de pasarla al modelo
def preprocess_image(frame, target_size=(120, 120)):
    resized = cv2.resize(frame, target_size)  # Redimensionar al tamaño esperado por el modelo
    resized = resized.astype('float32') / 255.0  # Normalizar
    resized = np.expand_dims(resized, axis=0)  # Añadir dimensión batch
    return resized

# Inicializar la cámara
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # Leer un frame de la cámara
    if not ret:
        break

    # Redimensionar la imagen antes de pasarla al modelo
    input_data = preprocess_image(frame)  # Redimensionar a 120x120 y preprocesar

    # Realizar predicción
    predictions = model.predict(input_data)
    class_id = np.argmax(predictions[0])  # Obtener la clase con mayor probabilidad
    label = class_labels[class_id]
    confidence = predictions[0][class_id]

    # Mostrar la etiqueta y la confianza en el frame
    text = f"{label}: {confidence:.2f}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Mostrar el frame
    cv2.imshow('Detección de Frutas', frame)

    # Salir al presionar la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
