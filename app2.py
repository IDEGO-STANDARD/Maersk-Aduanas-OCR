import os
from pdf2image import convert_from_path
from PIL import Image
import tensorflow as tf
import numpy as np
import cv2
import pytesseract
import json
import nltk
from nltk.tokenize import word_tokenize
import re
from flask import Flask, render_template, request

app = Flask(__name__)

nltk.download('punkt')
# Directorio de salida para las imágenes
output_folder = 'C:\\Users\\USER\\Desktop\\Datos_Modelo_1\\OCR\\BL3'

# Cargar el modelo entrenado
model = tf.keras.models.load_model('documento_classifier_model.h5')

# Funciones clasificador
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

# Funciones OCR

# Función para redimensionar y guardar la imagen
def save_image(image, output_folder, filename):

    # Guardar la imagen en el directorio de salida con el nombre especificado
    image.save(os.path.join(output_folder, filename))

# Función para guardar texto en un archivo de texto
def guardar_texto_en_archivo(texto, archivo):
    with open(archivo, 'w') as file:
        json.dump(texto, file)

# Definir el elemento estructurante
elemento_estructurante = np.ones((25, 25), np.uint8)  # Altura de 20 píxeles
# Número de iteraciones (ajusta según sea necesario)
num_iteraciones = 3
# Aplicar una pequeña dilatación
kernel_dilatacion = np.ones((5, 5), np.uint8) 
# Palabras claves y etiquetas:
palabras_clave = {
    'Bill of Lading': 'B/L',
    'B/L No': 'B/L',
    'B/LNo.': 'B/L',
    'BILL OF LADING NO.': 'B/L',
    'Port of Loading': 'POL',
    'Port of Discharge': 'POD',
    'PORT OF LOADING': 'POL',
    'PORT OF DISCHARGE': 'POD',
    'Voyage No.' : 'Voyage',
    'VOYAGE NO.' : 'Voyage',
    'Vessel': 'Vessel',
    'VESSEL' : 'Vessel',
    'Weight': 'Peso',
    'GROSS WEIGHT (KGS)': 'Peso'
}


etiquetas_y_textos = {}
resultados = {}
etiquetas_str = {}
texto = {}
resultados_lista = []

palabras_claves_a_eliminar = ['Bill of Lading', 'B/L No', 'B/LNo.', 'BILL OF LADING NO.', 'Port of Loading',
    'Port of Discharge', 'PORT OF LOADING', 'PORT OF DISCHARGE', 'Voyage No.', 'VOYAGE NO.', 'Vessel', 'VESSEL', 'Weight',
    'GROSS WEIGHT (KGS)', '(see clauses 1 + 19)', '(see clause 1 + 19)']

def es_bl(palabra):
    # Usar una expresión regular para buscar una secuencia de 7 a 13 caracteres alfanuméricos
    patron = r'\b[a-zA-Z0-9]{7,13}\b'
    return bool(re.search(patron, palabra))



# Función para seleccionar un archivo PDF y realizar la clasificación
def select_pdf_and_classify(file):

    pdf_name = file.filename
    save_path = 'C:/Users/USER/Desktop/Demo/{}'.format(pdf_name)
    file.save(save_path)

    # Convertir el PDF en imágenes
    pages = convert_from_path(save_path, 500)
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

    # OCR
    pdf_file_name = pdf_file_name = os.path.basename(save_path)

    # Inicializar una variable para el número de caja
    numero_de_caja = 1
    # Inicializar un diccionario para almacenar los textos de cada caja
    textos_de_cajas = {}
    # Convertir el PDF en imágenes
    pages2 = convert_from_path(save_path, 250)
    # Almacenar la primera imagen como img1
    img1_pillow = pages2[0]
    # Convertir la imagen de Pillow a una matriz numpy
    img1 = cv2.cvtColor(np.array(img1_pillow), cv2.COLOR_RGB2BGR)
    # Redimensionar la imagen al tamaño deseado (2066 x 2924)
    img1 = cv2.resize(img1, (2066, 2924))
    # Recortar la imagen a un largo de 2000 píxeles
    img1 = img1[:2000, :]
    # Convertir a escala de grises:
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # Aplicar Threshold:
    _, img1_thresh = cv2.threshold(img1_gray, 128, 255, cv2.THRESH_BINARY)
    # Proceso de dilatación y erosión:
    # Invertir imagen:
    img1_clean = 255 - img1_thresh
    # Definir el límite de tamaño máximo
    tamano_maximo = 450
    # Encontrar componentes conectados en la imagen binaria
    _, etiquetas, stats, _ = cv2.connectedComponentsWithStats(img1_clean, connectivity=4)
    # Crear una máscara para conservar componentes pequeños
    mascara = np.zeros_like(img1_clean)
    # Identificar los índices de los componentes que cumplen con el tamaño máximo
    selected_indices = np.where((stats[:, 2] <= tamano_maximo) & (stats[:, 3] <= tamano_maximo))
    # Marcar los componentes seleccionados en la máscara
    mascara[np.isin(etiquetas, selected_indices)] = 255

    # Aplicar el cierre iterativo
    for j in range(num_iteraciones):
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, elemento_estructurante)

    mascara_dilatada = cv2.dilate(mascara, kernel_dilatacion, iterations=1)

    # Encontrar contornos en la imagen 'mascara'
    contornos, _ = cv2.findContours(mascara_dilatada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Hacer una copia de la imagen 'img1' para dibujar las cajas
    img1_box = img1.copy()

    # Dibujar cajas verdes alrededor de los contornos
    for contorno in contornos:
        # Obtener las coordenadas de la caja que encierra el contorno
        x, y, w, h = cv2.boundingRect(contorno)
            
        # Filtrar contornos por criterios de tamaño
        if 16 < w < 1000 and 16 < h < 1000:
            # Dibujar una caja verde en 'img1_box' para los contornos que cumplen con los criterios
            cv2.rectangle(img1_box, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Recortar la región de interés (ROI) de la imagen 'img1' que corresponde a la caja
            roi = img1[y:y + h, x:x + w]

            # Ejecutar OCR en el ROI para obtener el texto
            texto = pytesseract.image_to_string(roi, config='--psm 6')  # Configura el modo de segmentación de página

            # Almacenar el texto en una variable 'filename_boxk', donde 'k' es el número de la caja
            variable_nombre = f'{pdf_file_name}_{numero_de_caja}'
            textos_de_cajas[variable_nombre] = texto
                    
            # Incrementar el número de caja
            numero_de_caja += 1

                    
    for key, texto in textos_de_cajas.items():
        if len(texto) <= 50:
            etiquetas_encontradas = []  # Lista de etiquetas para este resultado

            for palabra_clave, etiqueta in palabras_clave.items():
                if palabra_clave in texto:
                    etiquetas_encontradas.append(etiqueta)

            if etiquetas_encontradas:
                # Si se encontraron etiquetas, agrega el texto al diccionario
                etiquetas_y_textos[key] = (etiquetas_encontradas, texto)

    # Procesa textos eliminando palabras clave y muestra etiquetas y textos
    for key, (etiquetas, texto) in etiquetas_y_textos.items():
        for palabra_clave in palabras_claves_a_eliminar:
            texto = texto.replace(palabra_clave, '').strip()

        # Imprime las etiquetas y el texto
        etiquetas_str = ', '.join(etiquetas)

        resultados = (etiquetas_str, texto)
        resultados_lista.append(resultados)

        # Imprime la tupla de etiquetas y texto
        print(resultados)
        
        
    return class_labels[class_label], resultados_lista
    
@app.route('/recognize-pdf', methods = ['POST'])
def recognize_pdf():
    f = request.files['file']
    result_label, resultados_lista = select_pdf_and_classify(f)
    return {'type': result_label, 'resultados': resultados_lista}

if __name__ == "__main__":
    app.run(debug=True)