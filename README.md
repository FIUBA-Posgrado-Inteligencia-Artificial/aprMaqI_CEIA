[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

# Aprendizaje de Máquina

Este repositorio contiene el material de clases (presentaciones, ejercicios y notebooks) para la materia **Aprendizaje de Máquina** de la Especialización en Inteligencia Artificial (CEIA - FIUBA).

Para revisar los criterios de aprobación, consulta el documento de [Criterios de Aprobación](CriteriosAprobacion.md).

---

## 📂 Organización del Repositorio

La estructura general para cada clase es la siguiente:

```text
clase#/
├── teoria/              # Diapositivas y material teórico
├── ejercicios/          # Enunciados y guías prácticas
├── jupyter_notebooks/   # Notebooks de Jupyter para las clases
└── README.md            # Detalle específico de la clase
```

---

## 🛠️ Requerimientos y Entorno de Desarrollo

### Requisitos del Sistema
- **Lenguaje**: Python `>= 3.12`
- **Gestor de paquetes recomendado**: [uv](https://github.com/astral-sh/uv) (también se admite `pip` o `conda`)

### Tecnologías y Librerías Utilizadas
El entorno de desarrollo incluye las siguientes librerías principales (definidas en el archivo `pyproject.toml`):

| Librería / Herramienta | Propósito |
| :--- | :--- |
| **NumPy** | Cálculo numérico y álgebra lineal |
| **Pandas** | Manipulación y análisis de datos estructurados |
| **SciPy** | Herramientas científicas y estadísticas |
| **Scikit-Learn** | Modelado e implementación de algoritmos de Machine Learning |
| **XGBoost** | Algoritmo de gradient boosting optimizado |
| **Optuna** | Optimización de hiperparámetros |
| **Matplotlib** | Creación de gráficos y visualizaciones estáticas |
| **Seaborn** | Visualización de datos estadísticos basada en Matplotlib |
| **Pillow** | Procesamiento y manipulación de imágenes |
| **Jupyter** | Entorno interactivo y ejecución de notebooks |
| **IPython** / **ipykernel** | Consola interactiva y kernel para Jupyter |

### 🚀 Configuración del Entorno con `uv`

Este proyecto utiliza `pyproject.toml` para la gestión de dependencias. Se recomienda utilizar **`uv`**, un instalador y gestor de paquetes de Python extremadamente rápido escrito en Rust.

1. **Instalar `uv`** (si aún no lo tienes):
   Consulta la [documentación oficial de uv](https://docs.astral.sh/uv/getting-started/installation/).

2. **Crear y sincronizar el entorno virtual**:
   Navega al directorio raíz del proyecto y ejecuta:
   ```bash
   uv sync
   ```
   Esto creará automáticamente un entorno virtual (`.venv`) e instalará todas las dependencias requeridas.

3. **Ejecutar Jupyter**:
   Para levantar el servidor de Jupyter utilizando el entorno virtual creado:
   ```bash
   uv run jupyter notebook
   ```

---

## 📚 Contenido del Curso

A continuación se detallan los temas cubiertos en cada clase:

* ### 📝 [Clase 1](clase1/README.md)
  * Introducción a la materia.
  * Conceptos básicos de Machine Learning.

* ### 📝 [Clase 2](clase2/README.md)
  * Clasificador K-Nearest Neighbors (KNN).
  * Métricas de distancia.
  * Métodos de ajuste y selección de hiperparámetros.

* ### 📝 [Clase 3](clase3/README.md)
  * Máquinas de Vector de Soporte (Support Vector Machines - SVM).
  * SVM como clasificador.
  * SVM como modelo de regresión.

* ### 📝 [Clase 4](clase4/README.md)
  * Árboles de decisión.
  * Árboles de regresión.
  * Árboles de clasificación.

* ### 📝 [Clase 5](clase5/README.md)
  * Métodos de ensamble.
  * Boosting, Bagging y Random Forests.
  * Importancia de características (Feature Importance).

* ### 📝 [Clase 6](clase6/README.md)
  * Calibración de modelos.
  * Métricas de discriminación vs. calibración.
  * Métodos comunes de calibración.

* ### 📝 [Clase 7](clase7/README.md)
  * Aprendizaje no supervisado.
  * Métodos de clustering.
  * Algoritmo K-Means.
  * Modelos de Mixturas Gaussianas (GMM).
  * Clustering Jerárquico.

---

## 📖 Bibliografía

Se recomiendan principalmente los dos primeros libros de esta lista:

1. 📘 **Practical Statistics for Data Scientists**: *50+ Essential Concepts Using R and Python* - Peter Bruce (Ed. O’Reilly)
2. 📘 [**An Introduction to Statistical Learning**](https://www.statlearning.com/) - Gareth James et al. (Ed. Springer)
3. 📙 [**Python Data Science Handbook**](https://jakevdp.github.io/PythonDataScienceHandbook/) - Jake VanderPlas
4. 📗 **The Elements of Statistical Learning** - Trevor Hastie et al. (Ed. Springer)
5. 📕 **Data Science from Scratch**: *First Principles with Python* - Joel Grus (Ed. O’Reilly)
6. 📘 **The Hundred-Page Machine Learning Book** - Andriy Burkov (Ed. Burkov)
7. 📙 **Artificial Intelligence for Humans** (Series) - Jeff Heaton (Ed. Heaton Research):
   - *Volume 1: Fundamental Algorithms*
   - *Volume 2: Nature-Inspired Algorithms*
   - *Volume 3: Deep Learning and Neural Networks*
8. 📗 **Pattern Recognition and Machine Learning** - Christopher Bishop (Ed. Springer)
9. 📕 [**Deep Learning**](https://www.deeplearningbook.org/) - Ian Goodfellow et al.

---

Este proyecto está bajo una licencia [Creative Commons Atribución-NoComercial-CompartirIgual 4.0 Internacional][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: https://creativecommons.org/licenses/by-nc-sa/4.0/deed.es
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
