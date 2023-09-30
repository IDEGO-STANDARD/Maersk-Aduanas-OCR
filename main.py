import psycopg2

def connect_and_query():
    # Parámetros de conexión a nuestra base de datos maersk-aduanas
    DATABASE = 'db_maersk_aduanas'
    USER = 'AdministradorMaerskAduanas'
    PASSWORD = 'Idego2023.'
    HOST = 'sv-maersk-aduanas.postgres.database.azure.com' 
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

        # Ejecutar una consulta
        cur.execute("SELECT * FROM maersk_aduanas.bl;")

        cur.execute("INSERT INTO maersk_aduanas.bl (nro_bl, pais_embarque, peso, puerto_descarga) VALUES ('numero_prueba_bl3','Peru','20kg','el callao');")
        conn.commit()

        cur.execute("SELECT * FROM maersk_aduanas.bl;")

        # Obtener y mostrar los resultados
        rows = cur.fetchall()
        for row in rows:
            print(row)

        # Cerrar el cursor y la conexión
        cur.close()
        conn.close()

    except psycopg2.OperationalError as e:
        print("Error al conectar a la base de datos:", e)
    except Exception as e:
        print("Ocurrió un error:", e)


if __name__ == '__main__':
    connect_and_query()
