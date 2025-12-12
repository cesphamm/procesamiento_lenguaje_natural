# Procesamiento de Lenguaje Natural

Repositorio con los desafíos de la materia **Procesamiento de Lenguaje Natural** de la Especialización en Inteligencia Artificial (FIUBA).

---

## Desafío 1: Vectorización de Texto y Clasificación Naïve Bayes

📄 **Notebook:** `Desafio_1.ipynb`

### Descripción
Este desafío aborda la vectorización de documentos de texto y su clasificación utilizando el dataset clásico **20 Newsgroups**, que contiene aproximadamente 20,000 documentos distribuidos en 20 categorías temáticas diferentes.

### Objetivos
1. **Vectorizar documentos** y medir la similaridad entre ellos usando TF-IDF y similitud coseno.
2. **Análisis de similaridad**: Seleccionar 5 documentos al azar y encontrar los 5 más similares, analizando si la similaridad tiene sentido según el contenido y las etiquetas.
3. **Clasificación Zero-Shot**: Implementar un modelo de clasificación por prototipos asignando la clase del documento de entrenamiento más similar.
4. **Entrenar modelos Naïve Bayes** (MultinomialNB y ComplementNB) optimizando hiperparámetros para maximizar el F1-score macro.
5. **Vectorización de palabras**: Transponer la matriz documento-término para estudiar similaridad entre palabras.

### Solución
- **Vectorización**: Se utilizó `TfidfVectorizer` de scikit-learn para convertir los documentos en vectores TF-IDF con un vocabulario de ~100,000 términos.
- **Similaridad**: Se aplicó similitud coseno para encontrar documentos relacionados, observando que documentos de la misma categoría (ej: religión, deportes) tienden a agruparse.
- **Zero-Shot**: Obtuvo un F1-score macro de ~50%, con mejor rendimiento en clases con vocabulario distintivo como `rec.sport.hockey` y `comp.windows.x`.
- **Naïve Bayes**: Se realizó búsqueda de hiperparámetros con GridSearchCV:
  - **MultinomialNB**: F1-score macro = 67.3%
  - **ComplementNB**: F1-score macro = 69.9% (mejor rendimiento)
- **Análisis de palabras**: La transposición de la matriz permitió encontrar relaciones semánticas (ej: "moon" → lunar, phases, satellite; "space" → NASA, shuttle, SETI).

### Resultados Clave
| Modelo | F1-Score Macro | Accuracy |
|--------|:--------------:|:--------:|
| Zero-Shot | 50.5% | 50.9% |
| MultinomialNB | 67.3% | 69.7% |
| ComplementNB | 69.9% | 71.8% |

---

## Desafío 2: Custom Embeddings con Gensim

📄 **Notebook:** `Desafio_2.ipynb`

### Descripción
Este desafío consiste en crear embeddings de palabras personalizados utilizando la biblioteca **Gensim** y el texto completo de **"El Ingenioso Hidalgo Don Quijote de la Mancha"** obtenido de Project Gutenberg.

### Objetivos
1. **Crear vectores Word2Vec** propios basados en el corpus de Don Quijote.
2. **Explorar términos de interés**: Buscar palabras más similares y menos similares a términos clave de la obra.
3. **Reducción de dimensionalidad**: Aplicar t-SNE para proyectar los embeddings a 2 dimensiones.
4. **Análisis de clusters**: Identificar e interpretar grupos de palabras que se formen en la visualización.

### Solución
- **Modelo Word2Vec**: Se entrenó un modelo Skipgram con los siguientes parámetros:
  - `vector_size=300`: Dimensionalidad de los embeddings
  - `window=2`: Contexto de 2 palabras antes y después
  - `min_count=5`: Frecuencia mínima para incluir palabras
  - `negative=20`: Negative sampling
  - 20 épocas de entrenamiento

- **Corpus**: 36,470 documentos (líneas) con 5,509 palabras únicas en el vocabulario.

- **Análisis de similaridad**:
  - *windmills* → amadises, inhabitants, tombs, alcaldes
  - *dulcinea* → del, dulcinea's, campo, toboso
  - *armour* → shield, wrist, robe, doublet, fist
  - *king* → lion, exploit, manuscript, marsilio

- **Visualización**: Se aplicó t-SNE para reducir a 2D y se identificaron clusters temáticos:
  - Grupos de personajes y lugares
  - Verbos modales y conectores
  - Términos de caballería y aventura
  - Números y expresiones temporales

### Conclusiones
El modelo captura patrones lingüísticos y temáticos consistentes con la narrativa de Don Quijote, agrupando términos que comparten contextos similares y mostrando cohesión semántica entre personajes, lugares y conceptos de la obra.

---

## Desafío 3: Modelo de Lenguaje con Tokenización por Caracteres

📄 **Notebook:** `Desafio_3.ipynb`

### Descripción
Implementación de un **modelo de lenguaje a nivel de caracteres** utilizando redes neuronales recurrentes (RNN). Se entrena con el texto de **"La Odisea"** de Homero para generar texto nuevo.

### Objetivos
1. **Seleccionar un corpus** de texto para entrenar el modelo.
2. **Preprocesamiento**: Tokenizar por caracteres, estructurar el dataset y separar train/validación.
3. **Proponer arquitecturas RNN**: Implementar y comparar SimpleRNN, LSTM y GRU.
4. **Generación de texto**: Implementar estrategias de decodificación (greedy search, beam search determinístico y estocástico) y analizar el efecto de la temperatura.

### Solución
- **Corpus**: La Odisea de Homero (~681,000 caracteres, 58 caracteres únicos en el vocabulario).
- **Preprocesamiento**: Secuencias de 100 caracteres, 90% entrenamiento / 10% validación.
- **Arquitectura**: 2 capas RNN con 256 unidades, dropout 0.5, one-hot encoding.

**Modelos entrenados:**

| Arquitectura | Parámetros | Perplejidad (Val) |
|--------------|:----------:|:-----------------:|
| SimpleRNN | 227,386 | 9.08 |
| LSTM | 863,290 | 4.33 |
| GRU | 652,858 | **4.11** |

**Estrategias de generación:**
- **Greedy Search**: Determinístico, rápido, pero tiende a loops.
- **Beam Search Determinístico**: Mejor coherencia, explora múltiples hipótesis.
- **Beam Search Estocástico**: Añade diversidad controlada con temperatura.

**Efecto de la temperatura:**
- T=0.1: Texto coherente, menor variabilidad
- T=0.2: Introduce errores graduales
- T≥0.5: Texto incoherente

### Conclusiones
- **GRU** obtuvo el mejor rendimiento (PPL=4.11) con menos parámetros que LSTM.
- La temperatura óptima para este modelo está entre 0.1 y 0.2.
- Beam Search supera consistentemente a Greedy, evitando loops y produciendo texto más natural.

---

## Desafío 4: Traductor LSTM Seq2Seq

📄 **Notebook:** `Desafio_4.ipynb`

### Descripción
Construcción de un **traductor automático inglés-español** utilizando una arquitectura **Sequence-to-Sequence (Seq2Seq)** con redes LSTM encoder-decoder, basado en el dataset del Tatoeba Project.

### Objetivos
1. **Extender el entrenamiento** a más datos y tamaños de secuencias mayores.
2. **Explorar el impacto de la cantidad de neuronas** en las capas recurrentes (64, 128, 256 unidades).
3. **Mostrar 5 ejemplos** de traducciones generadas.
4. **Extras**:
   - Utilizar embeddings pre-entrenados (GloVe) para el idioma de entrada.
   - Cambiar la estrategia de generación implementando muestreo aleatorio y beam search estocástico.

### Solución
- **Dataset**: 10,000 pares de oraciones inglés-español del Tatoeba Project (de ~120,000 disponibles).
- **Vocabulario**: 
  - Inglés: 4,948 palabras (entrada máx: 16 tokens)
  - Español: 7,731 palabras (salida máx: 18 tokens)
- **Embeddings**: GloVe pre-entrenados (50 dimensiones) para el encoder, embeddings entrenables para el decoder.
- **Tokens especiales**: `<sos>` (start of sequence), `<eos>` (end of sequence).

**Arquitectura:**
- **Encoder**: Embedding (GloVe) + LSTM que produce estados (h, c)
- **Decoder**: Embedding entrenable + LSTM + Dense con softmax

**Impacto de la cantidad de neuronas:**

| Unidades LSTM | Val Accuracy | Val Loss | Observaciones |
|:-------------:|:------------:|:--------:|---------------|
| 64 | 72.6% | 1.69 | Entrenamiento rápido, menor capacidad |
| **128** | **73.3%** | **1.65** | Mejor balance rendimiento/complejidad |
| 256 | 74.2% | 1.60 | Mayor capacidad, más lento |

**Estrategias de inferencia implementadas:**
- **Greedy**: Selección del token más probable en cada paso.
- **Muestreo aleatorio (sampling)**: Selección estocástica con temperatura ajustable.
- **Beam Search estocástico**: Exploración de múltiples hipótesis con muestreo probabilístico.

### Ejemplos de traducción

| # | Entrada (Inglés) | Salida (Español) |
|:-:|------------------|------------------|
| 1 | "My mother say hi" | "mi madre dice hola" |
| 2 | "Every end is a new beginning" | "cada fin es un nuevo comienzo" |
| 3 | "The best of both worlds" | "lo mejor de ambos mundos" |
| 4 | "I know what you mean" | "sé lo que quieres decir" |
| 5 | "Give me a break" | "dame un descanso" |

### Conclusiones
- El modelo con **128 unidades LSTM** ofrece el mejor balance entre rendimiento y complejidad.
- **Impacto de las neuronas**: Aumentar de 64 a 256 unidades mejora el accuracy (~2%), pero incrementa significativamente el tiempo de entrenamiento.
- El accuracy alcanzado (~73%) es prometedor, aunque las traducciones a veces pierden precisión semántica en oraciones complejas.
- **Embeddings pre-entrenados**: GloVe mejoró la representación del encoder al capturar relaciones semánticas del inglés.
- **Estrategias de generación**: Beam search estocástico con temperatura baja (0.1-0.2) produjo traducciones más variadas y naturales que greedy.
- **Limitaciones**: Por restricciones de RAM, solo se usaron 10,000 de los ~120,000 pares disponibles. Se recomienda usar DataGenerator para escalar.

---

## Estructura del Repositorio

```
├── Desafio_1.ipynb          # Vectorización y Naïve Bayes
├── Desafio_2.ipynb          # Custom Embeddings con Gensim
├── Desafio_3.ipynb          # Modelo de Lenguaje RNN
├── Desafio_4.ipynb          # Traductor Seq2Seq
├── Desafio_4_pytorch.ipynb  # Versión alternativa en PyTorch
├── Desafio_4_dg.ipynb       # Versión con DataGenerator
└── README.md
```

---

## Tecnologías Utilizadas

- **Python 3.x**
- **TensorFlow / Keras** - Modelos de deep learning
- **scikit-learn** - Vectorización y clasificación
- **Gensim** - Word2Vec embeddings
- **NumPy / Pandas** - Manipulación de datos
- **Matplotlib / Seaborn** - Visualización

---

## Autor

**Carla Espínola Hamm**  
Especialización en Inteligencia Artificial - FIUBA
