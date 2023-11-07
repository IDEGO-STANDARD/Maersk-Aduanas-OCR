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
import psycopg2

app = Flask(__name__)

nltk.download('punkt')

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

# Aplicar una pequeña dilatación
kernel_dilatacion = np.ones((5, 5), np.uint8) 

def procesar_pdf(save_path, pdf_file_name):
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
    #img1 = img1[:2000, :]
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
    # Definir el elemento estructurante
    elemento_estructurante = np.ones((25, 25), np.uint8)
    # Aplicar el cierre iterativo
    num_iteraciones = 3  # Define el número de iteraciones
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
            #cv2.rectangle(img1_box, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Recortar la región de interés (ROI) de la imagen 'img1' que corresponde a la caja
            roi = img1[y:y + h, x:x + w]

            # Ejecutar OCR en el ROI para obtener el texto
            texto = pytesseract.image_to_string(roi, config='--psm 6')  # Configura el modo de segmentación de página

            # Almacenar el texto en una variable 'filename_boxk', donde 'k' es el número de la caja
            variable_nombre = f'{pdf_file_name}_{numero_de_caja}'
            textos_de_cajas[variable_nombre] = texto
                    
            # Almacenar coordenadas y texto en las listas correspondientes
            coordenadas_lista.append({variable_nombre: (x, y, w, h)})
            texto_lista.append({variable_nombre: texto})
            
            # Incrementar el número de caja
            numero_de_caja += 1
    return img1_box, textos_de_cajas, texto

# Palabras claves y etiquetas:
palabras_clave = {
    'Bill of Lading': 'codigoBL',
    'B/L No': 'codigoBL',
    'B/LNo.': 'codigoBL',
    'BILL OF LADING NO.': 'codigoBL',
    'Weight': 'pesoBruto',
    'GROSS WEIGHT (KGS)': 'pesoBruto',
    '000000001x' : 'paisEmbarque',
    'PORT OF DISCHARGE': 'puertoDescarga',
    'Port of Discharge': 'puertoDescarga',
    '000000002x' : 'idOrdenT',
    '000000003x' : 'codigoAduana',
    '000000004x' : 'fechaBl',
    '000000005x' : 'idEstado',
    'Port of Loading': 'puertoEmbarque',
    'PORT OF LOADING': 'puertoEmbarque',
    '000000006x' : 'nroBultos',
    '000000007x' : 'flete',
    '000000008x' : 'docFeeOrig',
    '000000009x' : 'handlingOrig',
    '000000010x' : 'surcharge',
    '000000011x' : 'fuelFee',
    '000000012x' : 'docFeeDestin',
    '000000013x' : 'handlingDestin',
    '000000014x' : 'portDues',
    '000000015x' : 'codigoBLMaster',
    '000000016x' : 'ETA'
}

etiquetas_y_textos = {}
resultados = {}
etiquetas_str = {}
texto = {}
resultados_lista = []
# Crear dos listas para almacenar coordenadas y texto
coordenadas_lista = []
texto_lista = []
texto_lista_box = {}

palabras_claves_a_eliminar = ['Bill of Lading', 'B/L No', 'B/LNo.', 'BILL OF LADING NO.', 'Port of Loading',
    'Port of Discharge', 'PORT OF LOADING', 'PORT OF DISCHARGE', 'Voyage No.', 'VOYAGE NO.', 'Vessel', 'VESSEL', 'Weight',
    'GROSS WEIGHT (KGS)', '(see clauses 1 + 19)', '(see clause 1 + 19)']

def es_bl(palabra):
    # Usar una expresión regular para buscar una secuencia de 7 a 13 caracteres alfanuméricos
    patron = r'\b[a-zA-Z0-9]{7,13}\b'
    return bool(re.search(patron, palabra))

# Función para seleccionar un archivo PDF y realizar la clasificación
def select_pdf_and_classify(file):

    # Parámetros de conexión a nuestra base de datos maersk-aduanas
    DATABASE = 'postgres'
    USER = 'AdministradorMaerskAduanasOCR'
    PASSWORD = 'n\XM~M$,I4|a;[{9(|9+HT3Kf'
    HOST = 'db-maersk-aduanas-ocr.postgres.database.azure.com' 
    PORT = '5432' 

    try:
        # Establecer la conexión
        conn = psycopg2.connect(
            dbname=DATABASE,
            user=USER,
            password=PASSWORD,
            host=HOST,
            port=PORT
        )
        print("Conexión establecida con éxito.")

        # Crear un cursor
        cur = conn.cursor()


    except psycopg2.OperationalError as e:
        print("Error al conectar a la base de datos:", e)
    except Exception as e:
        print("Ocurrió un error:", e)


    pdf_name = file.filename
    current_directory = os.getcwd()
    save_path = current_directory+"\\"+pdf_name
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
        0: "Bill Of Lading",
        1: "Certificado Origen",
        2: "Factura",
        3: "Packing List",
        4: "Seguro",
        5: "Otros"
    }

    # Mostrar el resultado de la clasificación
    result_label = f"El PDF es de tipo: {class_labels[class_label]}"
    #  print(result_label)

    # OCR
    pdf_file_name = pdf_file_name = os.path.basename(save_path)

    img1_box, textos_de_cajas, texto = procesar_pdf(save_path, pdf_file_name)

    if class_labels[class_label] == 'Bill Of Lading':
        for key, texto in textos_de_cajas.items():
            if len(texto) <= 50:
                etiquetas_encontradas = []  # Lista de etiquetas para este resultado

                for palabra_clave, etiqueta in palabras_clave.items():
                    if palabra_clave in texto:
                        etiquetas_encontradas.append(etiqueta)

                if etiquetas_encontradas:
                    # Si se encontraron etiquetas, agrega el texto al diccionario
                    etiquetas_y_textos[key] = (etiquetas_encontradas, texto)
                    # Busca el diccionario correspondiente en coordenadas_lista
                    for coord_dict in coordenadas_lista:
                        if key in coord_dict:
                            coordenadas_box = coord_dict[key]
                            etiquetas_encontradas = etiquetas_y_textos[key][0]  # Obtiene las etiquetas correspondientes
                            texto_lista_box[key] = (etiquetas_encontradas, coordenadas_box)
                    
        print(etiquetas_y_textos)
        print(texto_lista_box)
        # Procesa textos eliminando palabras clave y muestra etiquetas y textos
        for key, (etiquetas, texto) in etiquetas_y_textos.items():
            for palabra_clave in palabras_claves_a_eliminar:
                texto = texto.replace(palabra_clave, '').strip()

            # Comprueba si 'texto' está vacío o es None y asigna "No encontrado" en su lugar
            if texto is None or texto == "":
                texto = None

            # Imprime las etiquetas y el texto
            etiquetas_str = ', '.join(etiquetas)

            resultados = (etiquetas_str, texto)
            resultados_lista.append(resultados)

        """
        # Itera a través de las entradas en texto_lista_box
        for key, (etiquetas, coords) in texto_lista_box.items():
            x, y, w, h = coords
            label = ', '.join(etiquetas)  # Convierte la lista de etiquetas en una cadena

            # Dibuja un rectángulo en img1_box
            cv2.rectangle(img1_box, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Dibuja las etiquetas en el rectángulo
            cv2.putText(img1_box, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        """
        # Define un diccionario que mapea etiquetas a colores
        color_map = {
            'B/L': (0, 0, 255),  # Rojo
            'POL': (0, 255, 0),  # Verde
            'POD': (255, 0, 0),  # Azul
            'Vessel': (128, 0, 128),  # Morado
            'Peso': (139, 69, 19),  # Marrón
            'Voyage': (0, 165, 255)  # Naranja
        }

        # Itera a través de las entradas en texto_lista_box
        for key, (etiquetas, coords) in texto_lista_box.items():
            x, y, w, h = coords
            label = ', '.join(etiquetas)  # Convierte la lista de etiquetas en una cadena

            # Obtiene el color correspondiente a las etiquetas
            color = color_map.get(etiquetas[0], (0, 0, 0))  # Usamos el color negro (0, 0, 0) como valor predeterminado

            # Dibuja un rectángulo coloreado en img1_box
            cv2.rectangle(img1_box, (x, y), (x + w, y + h), color, 2)

            # Dibuja las etiquetas en el rectángulo con el mismo color
            cv2.putText(img1_box, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Guardar la máscara en el directorio de salida
        #    mask_filename = f'{pdf_file_name}_colorbox.png'
         #   save_image(Image.fromarray(img1_box), output_folder, mask_filename)
        #'B/L', 'POL', 'POD', 'Vessel', 'Peso', 'Voyage'
        
        # Crear un conjunto (set) de etiquetas existentes en resultados_lista
        etiquetas_existentes = {etiqueta for etiqueta, valor in resultados_lista}

        # Recorrer las palabras clave y agregar las que no existen en resultados_lista
        for palabra, etiqueta in palabras_clave.items():
            if etiqueta not in etiquetas_existentes:
                resultados_lista.append((etiqueta, None))

        # Imprime la tupla de etiquetas y texto
        print(resultados_lista)

        # Mapeo de etiquetas a variables
        column_mapping = {
            'codigoBL': '',
            'puertoEmbarque': '',
            'puertoDescarga': '',
            'Vessel': '',
            'pesoBruto': '',
            'Voyage': ''
        }

        # Itera a través de resultados_lista y llena el mapeo
        for etiqueta, valor in resultados_lista:
            if etiqueta in column_mapping:
                column_mapping[etiqueta] = valor

        # Construye la consulta
        query = f"INSERT INTO bl (nro_bl, puerto_de_carga, puerto_de_descarga, vessel, peso, no_voyage) " \
                f"VALUES ('{column_mapping['codigoBL']}', '{column_mapping['puertoEmbarque']}', '{column_mapping['puertoDescarga']}', " \
                f"'{column_mapping['Vessel']}', '{column_mapping['pesoBruto']}', '{column_mapping['Voyage']}');"

        # Ejecuta la consulta
        cur.execute(query)
        conn.commit()

        cur.execute("SELECT * FROM bl;")

        # Obtener y mostrar los resultados
        rows = cur.fetchall()
        for row in rows:
            print(row)
    
    elif class_labels[class_label] == 'Certificado Origen':
        print('Certificado Origen')

    elif class_labels[class_label] == 'Factura':
        print('Factura')

    elif class_labels[class_label] == 'Packing List':
        print('Packing List')

    elif class_labels[class_label] == 'Seguro':
        print('Seguro')

    else:
        print('Información no disponible')

    # Cerrar el cursor y la conexión
    cur.close()
    conn.close()

    resultados_dic = dict(resultados_lista)
    # Ejemplo:
    print(resultados_dic['codigoBL'])
    print(resultados_dic['pesoBruto']) 

    return class_labels[class_label], resultados_dic
    
@app.route('/recognize-pdf', methods = ['POST'])
def recognize_pdf():
    f = request.files['file']
    result_label, resultados_dic= select_pdf_and_classify(f)
    return {'type': result_label, 'resultados': resultados_dic}

if __name__ == "__main__":
    app.run(debug=True)
    