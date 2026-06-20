"""
utils.py - Utilidades para el Clasificador de Poder de Naruto

Este módulo contiene funciones auxiliares para:
- Carga y preprocesamiento de datos
- Generación de embeddings BERT
- Evaluación y visualización de modelos
- Gestión de archivos y configuraciones

Autor: César Delgado
Fecha: Diciembre 2024
"""

import json
import os
import re
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import logging

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTES Y CONFIGURACIÓN
# =============================================================================

CLASES_PODER = ['Taijutsu', 'Ninjutsu', 'Genjutsu', 'Kekkai Genkai']

CLASE_DESCRIPCIONES = {
    'Taijutsu': 'Combate cuerpo a cuerpo, artes marciales, fuerza física',
    'Ninjutsu': 'Técnicas ninja, chakra elemental, jutsus de fuego/agua/tierra/viento/rayo',
    'Genjutsu': 'Ilusiones, técnicas mentales, control de la mente',
    'Kekkai Genkai': 'Habilidades hereditarias, límites de línea de sangre, ojos especiales'
}

MODELO_BERT_DEFAULT = 'paraphrase-multilingual-MiniLM-L12-v2'

# Paths del proyecto
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'data'
MODELS_DIR = PROJECT_ROOT / 'models'


# =============================================================================
# FUNCIONES DE CARGA DE DATOS
# =============================================================================

def cargar_dataset(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Carga el dataset de personajes de Naruto.
    
    Args:
        filepath: Ruta al archivo JSON. Si es None, usa el archivo por defecto.
    
    Returns:
        DataFrame con columnas ['nombre', 'descripcion', 'clase']
    """
    if filepath is None:
        filepath = DATA_DIR / 'naruto_characters.json'
    
    filepath = Path(filepath)
    
    if not filepath.exists():
        logger.warning(f"Archivo no encontrado: {filepath}. Generando dataset sintético.")
        return generar_dataset_sintetico()
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data['personajes'])
    logger.info(f"Dataset cargado: {len(df)} personajes")
    return df


def generar_dataset_sintetico(n_samples_per_class: int = 25) -> pd.DataFrame:
    """
    Genera un dataset sintético de personajes de Naruto para entrenamiento.
    
    Args:
        n_samples_per_class: Número de ejemplos por clase
    
    Returns:
        DataFrame con datos sintéticos
    """
    logger.info("Generando dataset sintético...")
    
    # Templates de descripciones por clase
    templates = {
        'Taijutsu': [
            "Experto en combate cuerpo a cuerpo con técnicas de {tecnica}",
            "Maestro de artes marciales que utiliza {tecnica} en batalla",
            "Especialista en {tecnica} con fuerza física sobrehumana",
            "Ninja que domina el combate físico mediante {tecnica}",
            "Guerrero con habilidades excepcionales en {tecnica} y velocidad",
            "Luchador que perfeccionó las técnicas de {tecnica}",
            "Combatiente de élite conocido por su dominio del {tecnica}",
        ],
        'Ninjutsu': [
            "Usuario de técnicas de elemento {elemento} con gran poder",
            "Domina el {elemento} y puede crear técnicas devastadoras",
            "Especialista en jutsus de {elemento} aprendidos del clan",
            "Ninja con afinidad al {elemento} que manipula el chakra",
            "Maestro del {elemento} capaz de grandes ataques a distancia",
            "Experto en combinar chakra de {elemento} en sus técnicas",
            "Usuario avanzado de ninjutsu de tipo {elemento}",
        ],
        'Genjutsu': [
            "Maestro de ilusiones que atrapa a sus enemigos en {tipo}",
            "Especialista en {tipo} que manipula la percepción",
            "Experto en técnicas de {tipo} para control mental",
            "Usuario de genjutsu avanzado basado en {tipo}",
            "Ninja que domina el arte de las {tipo} ilusorias",
            "Combatiente que utiliza {tipo} para derrotar sin contacto",
            "Maestro ilusionista conocido por sus técnicas de {tipo}",
        ],
        'Kekkai Genkai': [
            "Poseedor del {kekkei} heredado de su clan",
            "Usuario del {kekkei} con habilidades únicas de línea de sangre",
            "Heredero del {kekkei} que le otorga poderes especiales",
            "Ninja con el {kekkei} que le permite técnicas exclusivas",
            "Portador del {kekkei} con capacidades sobrehumanas",
            "Miembro del clan con acceso al {kekkei} ancestral",
            "Guerrero bendecido con el {kekkei} de sus ancestros",
        ]
    }
    
    variables = {
        'Taijutsu': {
            'tecnica': ['Goken (Puño Fuerte)', 'Juken (Puño Suave)', 'Konoha Senpuu', 
                       'patadas dinámicas', 'la Hoja de Konoha', 'puños de hierro',
                       'las Ocho Puertas', 'combate de alta velocidad', 'taijutsu avanzado']
        },
        'Ninjutsu': {
            'elemento': ['fuego', 'agua', 'tierra', 'viento', 'rayo', 'madera',
                        'hielo', 'lava', 'vapor', 'explosión']
        },
        'Genjutsu': {
            'tipo': ['Sharingan', 'ilusiones visuales', 'control mental', 
                    'mundos ilusorios', 'pesadillas', 'parálisis mental',
                    'Tsukuyomi', 'técnicas oculares', 'hipnosis']
        },
        'Kekkai Genkai': {
            'kekkei': ['Sharingan', 'Byakugan', 'Rinnegan', 'elemento madera',
                      'elemento hielo', 'elemento lava', 'Shikotsumyaku',
                      'elemento magnético', 'elemento explosión', 'cadenas de chakra']
        }
    }
    
    nombres_base = [
        "Shinobi", "Ninja", "Guerrero", "Maestro", "Sensei", "Genin", "Chunin",
        "Jonin", "ANBU", "Kage", "Sannin", "Akatsuki", "Cazador", "Guardián"
    ]
    
    data = []
    np.random.seed(42)
    
    for clase, template_list in templates.items():
        for i in range(n_samples_per_class):
            template = np.random.choice(template_list)
            var_key = list(variables[clase].keys())[0]
            var_value = np.random.choice(variables[clase][var_key])
            
            descripcion = template.format(**{var_key: var_value})
            nombre = f"{np.random.choice(nombres_base)}_{clase[:3]}_{i+1}"
            
            data.append({
                'nombre': nombre,
                'descripcion': descripcion,
                'clase': clase
            })
    
    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    logger.info(f"Dataset sintético generado: {len(df)} personajes")
    return df


# =============================================================================
# FUNCIONES DE EMBEDDINGS
# =============================================================================

def obtener_modelo_embeddings(model_name: str = MODELO_BERT_DEFAULT):
    """
    Carga el modelo de sentence-transformers para generar embeddings.
    
    Args:
        model_name: Nombre del modelo de HuggingFace
    
    Returns:
        Modelo de SentenceTransformer
    """
    try:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Cargando modelo de embeddings: {model_name}")
        modelo = SentenceTransformer(model_name)
        return modelo
    except ImportError:
        logger.error("sentence-transformers no instalado. Ejecuta: pip install sentence-transformers")
        raise


def generar_embeddings(
    textos: List[str], 
    modelo=None, 
    batch_size: int = 32,
    show_progress: bool = True
) -> np.ndarray:
    """
    Genera embeddings BERT para una lista de textos.
    
    Args:
        textos: Lista de textos a procesar
        modelo: Modelo de SentenceTransformer (si None, se carga el default)
        batch_size: Tamaño del batch para procesamiento
        show_progress: Mostrar barra de progreso
    
    Returns:
        Array de embeddings (n_textos, embedding_dim)
    """
    if modelo is None:
        modelo = obtener_modelo_embeddings()
    
    logger.info(f"Generando embeddings para {len(textos)} textos...")
    embeddings = modelo.encode(
        textos, 
        batch_size=batch_size, 
        show_progress_bar=show_progress,
        convert_to_numpy=True
    )
    
    logger.info(f"Embeddings generados: shape {embeddings.shape}")
    return embeddings


# =============================================================================
# FUNCIONES DE PREPROCESAMIENTO
# =============================================================================

def limpiar_texto(texto: str) -> str:
    """
    Limpia y normaliza un texto para procesamiento.
    
    Args:
        texto: Texto a limpiar
    
    Returns:
        Texto limpio
    """
    # Convertir a minúsculas
    texto = texto.lower()
    
    # Remover caracteres especiales (mantener acentos)
    texto = re.sub(r'[^\w\sáéíóúüñ]', ' ', texto)
    
    # Remover espacios múltiples
    texto = re.sub(r'\s+', ' ', texto).strip()
    
    return texto


def preparar_datos(
    df: pd.DataFrame,
    columna_texto: str = 'descripcion',
    columna_clase: str = 'clase',
    limpiar: bool = True
) -> Tuple[List[str], np.ndarray, Dict[str, int]]:
    """
    Prepara los datos para entrenamiento.
    
    Args:
        df: DataFrame con los datos
        columna_texto: Nombre de la columna con el texto
        columna_clase: Nombre de la columna con la clase
        limpiar: Si se debe limpiar el texto
    
    Returns:
        Tuple de (textos, labels_encoded, label_mapping)
    """
    textos = df[columna_texto].tolist()
    
    if limpiar:
        textos = [limpiar_texto(t) for t in textos]
    
    # Codificar clases
    clases_unicas = sorted(df[columna_clase].unique())
    label_mapping = {clase: i for i, clase in enumerate(clases_unicas)}
    labels = np.array([label_mapping[c] for c in df[columna_clase]])
    
    logger.info(f"Datos preparados: {len(textos)} textos, {len(clases_unicas)} clases")
    return textos, labels, label_mapping


# =============================================================================
# FUNCIONES DE EVALUACIÓN Y VISUALIZACIÓN
# =============================================================================

def evaluar_modelo(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    clases: List[str]
) -> Dict:
    """
    Evalúa el modelo y retorna métricas.
    
    Args:
        y_true: Etiquetas verdaderas
        y_pred: Predicciones del modelo
        clases: Lista de nombres de clases
    
    Returns:
        Diccionario con métricas
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, 
        f1_score, classification_report, confusion_matrix
    )
    
    metricas = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro'),
        'recall_macro': recall_score(y_true, y_pred, average='macro'),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'classification_report': classification_report(
            y_true, y_pred, target_names=clases, output_dict=True
        )
    }
    
    return metricas


def plot_confusion_matrix(
    confusion_mat: np.ndarray, 
    clases: List[str],
    figsize: Tuple[int, int] = (10, 8),
    title: str = 'Matriz de Confusión'
):
    """
    Visualiza la matriz de confusión.
    
    Args:
        confusion_mat: Matriz de confusión
        clases: Nombres de las clases
        figsize: Tamaño de la figura
        title: Título del gráfico
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        confusion_mat, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=clases,
        yticklabels=clases
    )
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Predicción', fontsize=12)
    plt.ylabel('Real', fontsize=12)
    plt.tight_layout()
    return plt.gcf()


def plot_metricas_por_clase(metricas: Dict, clases: List[str]):
    """
    Visualiza las métricas por clase.
    
    Args:
        metricas: Diccionario con el reporte de clasificación
        clases: Lista de clases
    """
    import matplotlib.pyplot as plt
    
    report = metricas['classification_report']
    
    precision = [report[c]['precision'] for c in clases]
    recall = [report[c]['recall'] for c in clases]
    f1 = [report[c]['f1-score'] for c in clases]
    
    x = np.arange(len(clases))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#2ecc71')
    bars2 = ax.bar(x, recall, width, label='Recall', color='#3498db')
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#9b59b6')
    
    ax.set_xlabel('Clase de Poder', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Métricas por Clase', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(clases, rotation=15)
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    # Añadir valores sobre las barras
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    return fig


def plot_distribucion_clases(df: pd.DataFrame, columna_clase: str = 'clase'):
    """
    Visualiza la distribución de clases en el dataset.
    
    Args:
        df: DataFrame con los datos
        columna_clase: Nombre de la columna con la clase
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gráfico de barras
    conteo = df[columna_clase].value_counts()
    colors = ['#e74c3c', '#3498db', '#9b59b6', '#2ecc71']
    
    sns.barplot(x=conteo.index, y=conteo.values, ax=axes[0], palette=colors)
    axes[0].set_title('Distribución de Clases', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Clase de Poder', fontsize=12)
    axes[0].set_ylabel('Cantidad', fontsize=12)
    axes[0].tick_params(axis='x', rotation=15)
    
    for i, v in enumerate(conteo.values):
        axes[0].text(i, v + 0.5, str(v), ha='center', fontweight='bold')
    
    # Gráfico de pastel
    axes[1].pie(conteo.values, labels=conteo.index, autopct='%1.1f%%', 
                colors=colors, startangle=90, explode=[0.02]*len(conteo))
    axes[1].set_title('Proporción de Clases', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


# =============================================================================
# FUNCIONES DE GESTIÓN DE MODELOS
# =============================================================================

def guardar_modelo(
    modelo, 
    label_mapping: Dict,
    filepath: Optional[str] = None,
    nombre_modelo: str = 'clasificador_poder'
):
    """
    Guarda el modelo y el mapping de etiquetas.
    
    Args:
        modelo: Modelo entrenado
        label_mapping: Diccionario de mapeo clase -> índice
        filepath: Ruta donde guardar (si None, usa MODELS_DIR)
        nombre_modelo: Nombre base del archivo
    """
    if filepath is None:
        MODELS_DIR.mkdir(exist_ok=True)
        filepath = MODELS_DIR / f'{nombre_modelo}.joblib'
    
    data_to_save = {
        'modelo': modelo,
        'label_mapping': label_mapping,
        'inverse_mapping': {v: k for k, v in label_mapping.items()}
    }
    
    joblib.dump(data_to_save, filepath)
    logger.info(f"Modelo guardado en: {filepath}")


def cargar_modelo(filepath: Optional[str] = None) -> Tuple:
    """
    Carga un modelo guardado.
    
    Args:
        filepath: Ruta del archivo (si None, usa el default)
    
    Returns:
        Tuple de (modelo, label_mapping, inverse_mapping)
    """
    if filepath is None:
        filepath = MODELS_DIR / 'clasificador_poder.joblib'
    
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {filepath}")
    
    data = joblib.load(filepath)
    logger.info(f"Modelo cargado desde: {filepath}")
    
    return data['modelo'], data['label_mapping'], data['inverse_mapping']


# =============================================================================
# FUNCIONES DE PREDICCIÓN
# =============================================================================

def predecir_clase(
    texto: str,
    modelo=None,
    embedding_model=None,
    label_mapping: Optional[Dict] = None,
    mostrar_probabilidades: bool = True
) -> Dict:
    """
    Predice la clase de poder de un texto.
    
    Args:
        texto: Descripción del personaje
        modelo: Modelo clasificador (si None, carga el guardado)
        embedding_model: Modelo de embeddings (si None, carga el default)
        label_mapping: Mapeo de etiquetas
        mostrar_probabilidades: Si se muestran las probabilidades por clase
    
    Returns:
        Diccionario con predicción y probabilidades
    """
    # Cargar modelo si no se proporciona
    if modelo is None:
        modelo, label_mapping, inverse_mapping = cargar_modelo()
    else:
        inverse_mapping = {v: k for k, v in label_mapping.items()}
    
    # Cargar modelo de embeddings si no se proporciona
    if embedding_model is None:
        embedding_model = obtener_modelo_embeddings()
    
    # Limpiar y generar embedding
    texto_limpio = limpiar_texto(texto)
    embedding = embedding_model.encode([texto_limpio], convert_to_numpy=True)
    
    # Predecir
    prediccion_idx = modelo.predict(embedding)[0]
    clase_predicha = inverse_mapping[prediccion_idx]
    
    resultado = {
        'texto_original': texto,
        'texto_procesado': texto_limpio,
        'clase_predicha': clase_predicha,
        'descripcion_clase': CLASE_DESCRIPCIONES.get(clase_predicha, '')
    }
    
    # Obtener probabilidades si el modelo lo soporta
    if mostrar_probabilidades and hasattr(modelo, 'predict_proba'):
        probabilidades = modelo.predict_proba(embedding)[0]
        resultado['probabilidades'] = {
            inverse_mapping[i]: float(prob) 
            for i, prob in enumerate(probabilidades)
        }
    
    return resultado


def mostrar_prediccion(resultado: Dict):
    """
    Muestra la predicción de forma formateada.
    
    Args:
        resultado: Diccionario con los resultados de predicción
    """
    print("\n" + "="*60)
    print("🥷 CLASIFICADOR DE PODER - RESULTADO")
    print("="*60)
    print(f"\n📝 Texto: {resultado['texto_original']}")
    print(f"\n🎯 Clase Predicha: {resultado['clase_predicha']}")
    print(f"📖 Descripción: {resultado['descripcion_clase']}")
    
    if 'probabilidades' in resultado:
        print("\n📊 Probabilidades por clase:")
        for clase, prob in sorted(resultado['probabilidades'].items(), 
                                  key=lambda x: x[1], reverse=True):
            bar = "█" * int(prob * 20)
            print(f"   {clase:15} {bar:20} {prob:.1%}")
    
    print("\n" + "="*60)


# =============================================================================
# FUNCIÓN PRINCIPAL DE EJEMPLO
# =============================================================================

if __name__ == "__main__":
    # Ejemplo de uso de las utilidades
    print("🥷 Utilidades del Clasificador de Poder")
    print("-" * 40)
    
    # Generar dataset sintético
    df = generar_dataset_sintetico(n_samples_per_class=10)
    print(f"\nDataset generado: {len(df)} filas")
    print(df.head())
    
    # Preparar datos
    textos, labels, mapping = preparar_datos(df)
    print(f"\nMapping de clases: {mapping}")
    
    # Ejemplo de limpieza de texto
    texto_ejemplo = "¡Un NINJA con poder de FUEGO increíble!"
    texto_limpio = limpiar_texto(texto_ejemplo)
    print(f"\nTexto original: {texto_ejemplo}")
    print(f"Texto limpio: {texto_limpio}")
