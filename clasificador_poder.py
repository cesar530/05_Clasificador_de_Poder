#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
clasificador_poder.py - Clasificador de Poder de Personajes de Naruto

Script ejecutable para entrenar, evaluar y usar el clasificador de poder.
Utiliza BERT embeddings y clasificadores de Scikit-Learn para predecir
la clase de poder de personajes del universo de Naruto.

Uso:
    python clasificador_poder.py --train
    python clasificador_poder.py --predict "descripción del personaje"
    python clasificador_poder.py --evaluate
    python clasificador_poder.py --demo

Autor: César Delgado
Fecha: Diciembre 2024
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# Importar utilidades locales
from utils import (
    cargar_dataset,
    generar_dataset_sintetico,
    preparar_datos,
    obtener_modelo_embeddings,
    generar_embeddings,
    evaluar_modelo,
    plot_confusion_matrix,
    plot_metricas_por_clase,
    plot_distribucion_clases,
    guardar_modelo,
    cargar_modelo,
    predecir_clase,
    mostrar_prediccion,
    CLASES_PODER,
    MODELS_DIR
)

warnings.filterwarnings('ignore')


class ClasificadorPoder:
    """
    Clasificador de Poder para personajes de Naruto.
    
    Utiliza embeddings de BERT y clasificadores de Scikit-Learn
    para predecir la clase de poder de un personaje.
    """
    
    def __init__(
        self, 
        modelo_tipo: str = 'svm',
        modelo_embeddings: str = 'paraphrase-multilingual-MiniLM-L12-v2'
    ):
        """
        Inicializa el clasificador.
        
        Args:
            modelo_tipo: Tipo de clasificador ('rf', 'svm', 'mlp')
            modelo_embeddings: Nombre del modelo de embeddings
        """
        self.modelo_tipo = modelo_tipo
        self.modelo_embeddings_name = modelo_embeddings
        self.modelo_embeddings = None
        self.clasificador = None
        self.label_mapping = None
        self.inverse_mapping = None
        self._inicializar_clasificador()
    
    def _inicializar_clasificador(self):
        """Inicializa el clasificador según el tipo especificado."""
        clasificadores = {
            'rf': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'svm': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            ),
            'mlp': MLPClassifier(
                hidden_layer_sizes=(256, 128),
                activation='relu',
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )
        }
        
        if self.modelo_tipo not in clasificadores:
            raise ValueError(f"Tipo de modelo no válido: {self.modelo_tipo}. "
                           f"Opciones: {list(clasificadores.keys())}")
        
        self.clasificador = clasificadores[self.modelo_tipo]
        print(f"✓ Clasificador inicializado: {self.modelo_tipo.upper()}")
    
    def _cargar_modelo_embeddings(self):
        """Carga el modelo de embeddings si no está cargado."""
        if self.modelo_embeddings is None:
            print("⏳ Cargando modelo de embeddings BERT...")
            self.modelo_embeddings = obtener_modelo_embeddings(
                self.modelo_embeddings_name
            )
            print("✓ Modelo de embeddings cargado")
    
    def entrenar(
        self, 
        df: pd.DataFrame = None,
        test_size: float = 0.2,
        validar: bool = True
    ) -> dict:
        """
        Entrena el clasificador con los datos proporcionados.
        
        Args:
            df: DataFrame con columnas ['nombre', 'descripcion', 'clase']
            test_size: Proporción de datos para test
            validar: Si se realiza validación cruzada
        
        Returns:
            Diccionario con métricas de entrenamiento
        """
        print("\n" + "="*60)
        print("🏋️ ENTRENAMIENTO DEL MODELO")
        print("="*60)
        
        # Cargar datos si no se proporcionan
        if df is None:
            print("\n📂 Cargando dataset...")
            df = cargar_dataset()
        
        print(f"   Total de muestras: {len(df)}")
        print(f"   Clases: {df['clase'].unique().tolist()}")
        
        # Preparar datos
        print("\n🔧 Preparando datos...")
        textos, labels, self.label_mapping = preparar_datos(df)
        self.inverse_mapping = {v: k for k, v in self.label_mapping.items()}
        
        # Generar embeddings
        self._cargar_modelo_embeddings()
        print("\n🧠 Generando embeddings BERT...")
        X = generar_embeddings(textos, self.modelo_embeddings)
        
        # Dividir datos
        print(f"\n📊 Dividiendo datos (test_size={test_size})...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=test_size, random_state=42, stratify=labels
        )
        print(f"   Train: {len(X_train)} muestras")
        print(f"   Test: {len(X_test)} muestras")
        
        # Validación cruzada
        if validar:
            print("\n🔄 Realizando validación cruzada (5-fold)...")
            cv_scores = cross_val_score(
                self.clasificador, X_train, y_train, cv=5, scoring='f1_macro'
            )
            print(f"   F1-Score CV: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        # Entrenar modelo final
        print("\n🎯 Entrenando modelo final...")
        self.clasificador.fit(X_train, y_train)
        
        # Evaluar en test
        print("\n📈 Evaluando en conjunto de test...")
        y_pred = self.clasificador.predict(X_test)
        clases = [self.inverse_mapping[i] for i in range(len(self.label_mapping))]
        
        metricas = evaluar_modelo(y_test, y_pred, clases)
        
        print(f"\n   Accuracy: {metricas['accuracy']:.4f}")
        print(f"   F1-Score (Macro): {metricas['f1_macro']:.4f}")
        print(f"   Precision (Macro): {metricas['precision_macro']:.4f}")
        print(f"   Recall (Macro): {metricas['recall_macro']:.4f}")
        
        print("\n📋 Reporte de Clasificación:")
        print(classification_report(y_test, y_pred, target_names=clases))
        
        # Guardar modelo
        print("\n💾 Guardando modelo...")
        guardar_modelo(self.clasificador, self.label_mapping)
        
        print("\n" + "="*60)
        print("✅ ENTRENAMIENTO COMPLETADO")
        print("="*60)
        
        return {
            'metricas': metricas,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'clases': clases
        }
    
    def predecir(self, texto: str, verbose: bool = True) -> dict:
        """
        Predice la clase de poder de un texto.
        
        Args:
            texto: Descripción del personaje
            verbose: Si se muestra el resultado formateado
        
        Returns:
            Diccionario con la predicción
        """
        # Cargar modelo si no está entrenado
        if self.label_mapping is None:
            try:
                self.clasificador, self.label_mapping, self.inverse_mapping = cargar_modelo()
            except FileNotFoundError:
                print("❌ Error: No hay modelo entrenado. Ejecuta --train primero.")
                return None
        
        self._cargar_modelo_embeddings()
        
        resultado = predecir_clase(
            texto,
            modelo=self.clasificador,
            embedding_model=self.modelo_embeddings,
            label_mapping=self.label_mapping,
            mostrar_probabilidades=True
        )
        
        if verbose:
            mostrar_prediccion(resultado)
        
        return resultado
    
    def evaluar_con_ejemplos(self) -> None:
        """Evalúa el modelo con ejemplos predefinidos."""
        ejemplos = [
            "Un ninja experto en artes marciales que puede romper rocas con sus puños",
            "Maestro de ilusiones que atrapa a sus enemigos en mundos de pesadillas",
            "Usuario del elemento fuego capaz de crear dragones de llamas",
            "Poseedor del Sharingan heredado de su clan con poderes oculares únicos",
            "Combatiente veloz especializado en patadas giratorias",
            "Ninja que domina técnicas de agua y puede crear tsunamis",
            "Experto en hipnosis y control mental mediante genjutsu",
            "Heredero del elemento madera con habilidades de línea de sangre"
        ]
        
        print("\n" + "="*60)
        print("🧪 EVALUACIÓN CON EJEMPLOS")
        print("="*60)
        
        for i, ejemplo in enumerate(ejemplos, 1):
            print(f"\n--- Ejemplo {i} ---")
            self.predecir(ejemplo, verbose=True)


def demo_interactivo():
    """Modo demo interactivo."""
    print("\n" + "="*60)
    print("🎮 MODO DEMO INTERACTIVO")
    print("="*60)
    print("\nEscribe la descripción de un personaje para clasificar.")
    print("Escribe 'salir' para terminar.\n")
    
    clasificador = ClasificadorPoder()
    
    while True:
        texto = input("\n🥷 Descripción del personaje: ").strip()
        
        if texto.lower() in ['salir', 'exit', 'quit', 'q']:
            print("\n¡Hasta luego! 👋")
            break
        
        if not texto:
            print("Por favor, ingresa una descripción válida.")
            continue
        
        clasificador.predecir(texto)


def main():
    """Función principal del script."""
    parser = argparse.ArgumentParser(
        description='🥷 Clasificador de Poder - Naruto Character Class Predictor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python clasificador_poder.py --train
  python clasificador_poder.py --predict "Un ninja experto en fuego"
  python clasificador_poder.py --evaluate
  python clasificador_poder.py --demo
        """
    )
    
    parser.add_argument(
        '--train', 
        action='store_true',
        help='Entrena el modelo con el dataset'
    )
    
    parser.add_argument(
        '--predict', 
        type=str,
        metavar='TEXTO',
        help='Predice la clase de poder del texto proporcionado'
    )
    
    parser.add_argument(
        '--evaluate', 
        action='store_true',
        help='Evalúa el modelo con ejemplos predefinidos'
    )
    
    parser.add_argument(
        '--demo', 
        action='store_true',
        help='Inicia el modo demo interactivo'
    )
    
    parser.add_argument(
        '--modelo', 
        type=str,
        choices=['rf', 'svm', 'mlp'],
        default='svm',
        help='Tipo de modelo: rf (RandomForest), svm (SVM), mlp (MLP)'
    )
    
    args = parser.parse_args()
    
    # Si no se proporciona ningún argumento, mostrar ayuda
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    # Crear instancia del clasificador
    clasificador = ClasificadorPoder(modelo_tipo=args.modelo)
    
    # Ejecutar acción solicitada
    if args.train:
        clasificador.entrenar()
    
    elif args.predict:
        clasificador.predecir(args.predict)
    
    elif args.evaluate:
        clasificador.evaluar_con_ejemplos()
    
    elif args.demo:
        demo_interactivo()


if __name__ == "__main__":
    main()
