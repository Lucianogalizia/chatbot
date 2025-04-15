import os
import sys
import pandas as pd
# Importamos las dependencias necesarias de LangChain
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ================================
# Funciones de utilidad
# ================================

def cargar_dataframe(excel_path):
    """
    Carga el DataFrame a partir de un archivo Excel.
    """
    if not os.path.exists(excel_path):
        print(f"El archivo {excel_path} no existe. Verifica la ruta.")
        sys.exit(1)
    df = pd.read_excel(excel_path)
    if 'well_legal_name' not in df.columns:
        raise ValueError("La columna 'well_legal_name' no existe en el archivo Excel.")
    return df

def obtener_lista_pozos(df):
    """
    Retorna una lista ordenada de pozos disponibles a partir de la columna 'well_legal_name'.
    """
    return sorted(df['well_legal_name'].dropna().unique())

def seleccionar_pozo(lista_pozos):
    """
    Muestra la lista de pozos y solicita al usuario que seleccione uno.
    """
    print("Lista de pozos disponibles:")
    for idx, pozo in enumerate(lista_pozos):
        print(f"{idx+1}. {pozo}")
    seleccion = input("Ingresa el número correspondiente al pozo que deseas seleccionar: ")
    try:
        idx = int(seleccion) - 1
        if idx < 0 or idx >= len(lista_pozos):
            print("Selección inválida.")
            sys.exit(1)
        return lista_pozos[idx]
    except ValueError:
        print("Entrada inválida. Debes ingresar un número.")
        sys.exit(1)

def convertir_a_documentos(df_subset):
    """
    Convierte las filas del DataFrame en una lista de documentos.
    Cada documento es una concatenación de los valores no nulos de cada columna.
    """
    documentos = []
    for _, fila in df_subset.iterrows():
        contenido = "\n".join([f"{col}: {fila[col]}" for col in df_subset.columns if pd.notnull(fila[col])])
        documentos.append(Document(page_content=contenido))
    return documentos

def indexar_documentos(documentos, db_directory, embeddings):
    """
    Separa los documentos en fragmentos y crea la base vectorial usando Chroma.
    """
    os.makedirs(db_directory, exist_ok=True)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = splitter.split_documents(documentos)
    vector_store_local = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_directory
    )
    return vector_store_local

def indexar_pozo(df, pozo, db_directory="vectordb_excel"):
    """
    Filtra el DataFrame según el pozo seleccionado, convierte las filas en documentos
    e indexa dichos documentos.
    """
    df_filtrado = df[df['well_legal_name'] == pozo]
    if df_filtrado.empty:
        raise ValueError(f"No hay datos para el pozo {pozo}.")
    documentos = convertir_a_documentos(df_filtrado)
    print(f"Documentos generados: {len(documentos)}")
    # Configuramos los embeddings (asegúrate de tener el modelo adecuado descargado)
    embeddings = OllamaEmbeddings(model="deepseek-r1:14b")
    vector_store = indexar_documentos(documentos, db_directory, embeddings)
    print("Indexación completada. Puedes proceder a la consulta en la siguiente parte.")
    return vector_store

# ================================
# Función principal
# ================================

def main():
    # Ajusta la ruta del archivo Excel según corresponda. 
    # Para RunPod es recomendable usar rutas relativas y ubicar el archivo dentro del repositorio (p.ej., en la raíz o en una carpeta específica).
    excel_path = "datos_agrupados.xlsx"
    
    # Cargar el DataFrame
    df = cargar_dataframe(excel_path)
    
    # Mostrar la lista de pozos y solicitar la selección al usuario
    lista_pozos = obtener_lista_pozos(df)
    pozo_seleccionado = seleccionar_pozo(lista_pozos)
    print(f"Pozo seleccionado: {pozo_seleccionado}")
    
    # Realizar la indexación y obtener el vector_store
    vector_store = indexar_pozo(df, pozo_seleccionado)
    
    # Aquí podríamos guardar el vector_store en disco o pasarlo a la siguiente parte (por ejemplo, serializándolo)
    # Por ahora, dejamos el proceso preparado para la consulta posterior.
    
if __name__ == "__main__":
    main()
