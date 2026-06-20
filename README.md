# 🥷 Clasificador de Poder - Naruto Character Class Predictor

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-orange.svg)
![Transformers](https://img.shields.io/badge/Transformers-BERT-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📖 Descripción

Modelo de Machine Learning y NLP que predice la clase de poder de un personaje del universo de Naruto basándose en su descripción textual. El clasificador determina si un personaje pertenece a una de las siguientes categorías:

| Clase | Descripción |
|-------|-------------|
| **Taijutsu** | Especialistas en combate cuerpo a cuerpo y artes marciales |
| **Ninjutsu** | Usuarios de técnicas ninja basadas en chakra elemental |
| **Genjutsu** | Maestros de ilusiones y técnicas mentales |
| **Kekkai Genkai** | Poseedores de habilidades hereditarias únicas |

## 🎯 Objetivos del Proyecto

- Demostrar dominio de **procesamiento de lenguaje natural (NLP)**
- Implementar **clasificación multicategoría** con Scikit-Learn
- Utilizar **BERT embeddings** para representación semántica de texto
- Crear un pipeline reproducible y bien documentado

## 🏗️ Arquitectura del Proyecto

```
05_Clasificador_de_Poder/
│
├── clasificador_poder.ipynb      # Notebook principal con análisis completo
├── clasificador_poder.py         # Script ejecutable del modelo
├── utils.py                      # Funciones auxiliares y utilidades
├── data/
│   └── naruto_characters.json    # Dataset de personajes
├── models/                       # Modelos entrenados (generado)
├── requirements.txt              # Dependencias del proyecto
├── .gitignore                    # Archivos ignorados por Git
└── README.md                     # Este archivo
```

## 🔧 Instalación

1. **Clonar el repositorio:**
```bash
git clone https://github.com/tu-usuario/clasificador-poder-naruto.git
cd clasificador-poder-naruto
```

2. **Crear entorno virtual:**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. **Instalar dependencias:**
```bash
pip install -r requirements.txt
```

## 🚀 Uso

### Notebook Interactivo
```bash
jupyter notebook clasificador_poder.ipynb
```

### Script de Línea de Comandos
```bash
# Entrenar el modelo
python clasificador_poder.py --train

# Predecir clase de un personaje
python clasificador_poder.py --predict "Un ninja que domina el fuego y puede lanzar bolas de fuego gigantes"

# Evaluar el modelo
python clasificador_poder.py --evaluate
```

## 📊 Pipeline del Modelo

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Texto de      │───▶│  BERT Encoder   │───▶│  Embeddings     │───▶│  Clasificador   │
│   Entrada       │    │  (sentence-     │    │  (768 dims)     │    │  (RandomForest/ │
│                 │    │   transformers) │    │                 │    │   SVM/MLP)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
                                                                              │
                                                                              ▼
                                                                    ┌─────────────────┐
                                                                    │  Predicción:    │
                                                                    │  - Taijutsu     │
                                                                    │  - Ninjutsu     │
                                                                    │  - Genjutsu     │
                                                                    │  - Kekkai Genkai│
                                                                    └─────────────────┘
```

## 📈 Resultados

| Modelo | Accuracy | F1-Score (Macro) | Tiempo Entrenamiento |
|--------|----------|------------------|----------------------|
| Random Forest | ~85% | ~0.84 | ~2s |
| SVM (RBF) | ~87% | ~0.86 | ~1s |
| MLP | ~88% | ~0.87 | ~5s |

## 🛠️ Tecnologías Utilizadas

- **Python 3.9+**: Lenguaje de programación principal
- **Scikit-Learn**: Algoritmos de Machine Learning
- **Sentence-Transformers**: Embeddings BERT preentrenados
- **Pandas/NumPy**: Manipulación de datos
- **Matplotlib/Seaborn**: Visualización de resultados
- **Joblib**: Serialización de modelos

## 📚 Conceptos Demostrados

- [x] Procesamiento de Lenguaje Natural (NLP)
- [x] Transfer Learning con BERT
- [x] Clasificación Multicategoría
- [x] Validación Cruzada
- [x] Métricas de Evaluación (Precision, Recall, F1)
- [x] Matriz de Confusión
- [x] Pipeline de ML reproducible

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue primero para discutir los cambios propuestos.

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

## 👤 Autor

- 👤 Autor : **César Adrián Delgado Díaz**
- 💼 LinkedIn: [linkedin.com/in/cesar-delgado-diaz](linkedin.com/in/cesar-delgado-diaz)
- 🐙 GitHub: [github.com/cesar530](https://github.com/cesar530)

---
*Proyecto desarrollado como parte de un portafolio profesional de Data Science y Machine Learning.*
