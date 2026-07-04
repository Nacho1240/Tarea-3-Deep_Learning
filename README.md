

## 1. Estructura del repositorio

- `task3/requirements.txt` - dependencias Python necesarias
- `task3/config.yaml` - configuración de tema, modelos y números de ejemplos
- `task3/generate_dataset.py` - genera el dataset sintético para un modelo local
- `task3/evaluate_dataset.py` - calcula métricas de comparación entre archivos generados
- `task3/setup_models.py` - descarga modelos de prueba y los guarda localmente

## 2. Fase 1: Preparación del prompt

Se diseñó un prompt unificado que pide:
- texto de reseña en español
- comentarios sobre sonido, comodidad, batería y diseño
- etiqueta de sentimiento
- puntaje del 1 al 5

Este prompt se usa igual para los tres modelos, lo que permite comparar resultados de forma justa.

## 3. Fase 2: Generación del dataset

El script `generate_dataset.py` realiza:
1. carga del tokenizer y modelo local
2. generación de N ejemplos con el mismo prompt base
3. almacenamiento de cada ejemplo en formato JSONL

Ejemplo de ejecución:

```bash
cd "c:\Users\chanc\Downloads\T3 Deep\task3"
python -m pip install -r requirements.txt
python setup_models.py
python generate_dataset.py --model-id ./models/llama --output-file ./data/generated_llama.jsonl --num-examples 20
python generate_dataset.py --model-id ./models/phi --output-file ./data/generated_phi.jsonl --num-examples 20
python generate_dataset.py --model-id ./models/mistral --output-file ./data/generated_mistral.jsonl --num-examples 20
```

> Ajusta `--model-id` para apuntar a las carpetas locales de LLaMA, Phi y Mistral.
> 
> `python setup_models.py` crea modelos de prueba en `./models/llama`, `./models/phi` y `./models/mistral`.
> Reemplaza esos modelos de prueba con los pesos reales para la tarea definitiva.

## 4. Fase 3: Evaluación cuantitativa

El script `evaluate_dataset.py` calcula:
- variedad de ejemplos (`distinct_ratio`)
- cumplimiento del formato solicitado
- detección de palabras clave relevantes
- longitud promedio de los textos
- distribución de sentimientos detectados

Ejemplo de uso:

```bash
python evaluate_dataset.py --files ./data/generated_llama.jsonl ./data/generated_phi.jsonl ./data/generated_mistral.jsonl
```

## 5. Fase 4: Comparación y análisis

### Criterios evaluados
- **Coherencia:** el texto generado debe ser legible
- **Relevancia:** debe ajustarse al tema de auriculares y altavoces
- **Diversidad:** no debe repetir las mismas expresiones
- **Formato:** debe incluir `Reseña`, `Sentimiento` y `Puntaje`

### Resultados esperados

Al ejecutar los tres modelos, se deben comparar por:
- tasa de cumplimiento de formato
- cantidad de palabras clave por generación
- diversidad entre ejemplos
- distribución de sentimientos

## 6. Cómo presentar la tarea

La entrega final debe incluir:
- código Python en `task3/`
- dataset generado en `task3/data/`
- análisis en el README
- material de apoyo adicional como video y presentación (no incluidos aquí)

## 7. Recomendaciones finales

1. Coloca los modelos locales en las rutas indicadas en `config.yaml`.
2. Si no hay GPU, los scripts detectan CPU automáticamente.
3. Genera el dataset con la misma cantidad de ejemplos para cada modelo.
4. Revisa manualmente algunos ejemplos para el análisis cualitativo.
5. Usa `evaluate_dataset.py` para obtener métricas comparativas.
