import os
import tkinter as tk
from tkinter import filedialog
from pdf2image import convert_from_path
from PIL import Image
import tensorflow as tf
import numpy as np
import cv2

# Cargar el modelo entrenado
model = tf.keras.models.load_model('documento_classifier_model.h5')

# Función para redimensionar una imagen a 224x224 píxeles
def resize_image(image):
    return image.resize((224, 224))

# Función para clasificar una imagen
def classify_image(image):
    image = np.asarray(image) / 255.0
    image = np.expand_dims(image, axis=0)
    predictions = model.predict(image)
    class_label = np.argmax(predictions)
    return class_label

# Función para seleccionar un archivo PDF y realizar la clasificación
def select_pdf_and_classify():
    root = tk.Tk()
    root.withdraw()  # Ocultar la ventana principal

    # Abrir el cuadro de diálogo para seleccionar un archivo PDF
    file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])

    if file_path:
        # Convertir el PDF en imágenes
        pages = convert_from_path(file_path, 500)
        img1_pillow = pages[0]

        # Redimensionar la imagen
        resized_image = resize_image(img1_pillow)

        # Clasificar la imagen
        class_label = classify_image(resized_image)

        # Definir las etiquetas de clase
        class_labels = {
            0: "BL",
            1: "CO",
            2: "Factura",
            3: "PL",
            4: "Seguro",
            5: "Otros"
        }

        # Mostrar el resultado de la clasificación
        result_label = f"El PDF es de tipo: {class_labels[class_label]}"
        print(result_label)

if __name__ == "__main__":
    select_pdf_and_classify()