"""
Generador de dataset sintético para Tarea 3 - CIT3360
Versión compatible con CPU y AMD GPU (sin CUDA requerido).
Usa llama-cpp-python con modelos GGUF pre-cuantizados.

Instalación:
    pip install llama-cpp-python

Modelos GGUF sugeridos (descargar de HuggingFace):
    LLaMA  → bartowski/Meta-Llama-3.1-8B-Instruct-GGUF  (archivo: *Q4_K_M.gguf)
    Mistral → bartowski/Mistral-7B-Instruct-v0.2-GGUF    (archivo: *Q4_K_M.gguf)
    Phi-3  → bartowski/Phi-3-mini-4k-instruct-GGUF       (archivo: *Q4_K_M.gguf)

Uso de RAM aproximado con Q4_K_M:
    LLaMA 3.1 8B  → ~5.5 GB  ✅
    Mistral 7B    → ~4.8 GB  ✅
    Phi-3 mini    → ~2.5 GB  ✅
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict

from llama_cpp import Llama

DEFAULT_SYSTEM = (
    "Eres un asistente que genera reseñas de productos tecnológicos en español. "
    "Responde siempre con exactamente el siguiente formato:\n"
    "Reseña: <texto de 3-4 oraciones sobre sonido, comodidad, batería y diseño>\n"
    "Sentimiento: <Positivo|Negativo|Neutro>\n"
    "Puntaje: <número del 1 al 5>"
)

CONTEXTS = [
    "auriculares inalámbricos gaming con bajo latencia",
    "auriculares over-ear con cancelación de ruido activa",
    "audífonos True Wireless con estuche de carga",
    "altavoz portátil resistente al agua para exteriores",
    "auriculares deportivos con clip para la oreja",
    "barra de sonido compacta para escritorio",
    "auriculares on-ear con diadema plegable para viajes",
    "altavoz Bluetooth para ducha con micrófono",
    "auriculares inalámbricos de gama media con micrófono",
    "auriculares de estudio cerrados con cable desmontable",
]


def load_model(model_path: str, n_ctx: int = 512, n_threads: int = 8) -> Llama:
    """
    Carga un modelo GGUF con llama-cpp-python.
    n_gpu_layers=0 fuerza ejecución en CPU (compatible con AMD y cualquier hardware).
    Aumenta n_gpu_layers si tienes ROCm configurado correctamente.
    """
    print(f"Cargando modelo: {model_path}")
    model = Llama(
        model_path=model_path,
        n_ctx=n_ctx,           # Contexto de tokens (512 es suficiente para este task)
        n_threads=n_threads,   # Usa múltiples hilos del Xeon
        n_gpu_layers=0,        # 0 = CPU puro. Cambiar a -1 si tienes ROCm+llama-cpp compilado con ROCm
        verbose=False,
    )
    print("Modelo cargado.")
    return model


def build_messages(context: str) -> List[Dict]:
    """Construye el historial de mensajes en formato chat para el modelo."""
    user_msg = (
        f"Genera una reseña de producto tecnológico en español sobre: {context}. "
        "Usa exactamente el formato indicado: Reseña, Sentimiento y Puntaje."
    )
    return [
        {"role": "system", "content": DEFAULT_SYSTEM},
        {"role": "user", "content": user_msg},
    ]


def generate_examples(
    model: Llama,
    num_examples: int,
    max_tokens: int,
    temperature: float,
) -> List[Dict]:
    """Genera num_examples reseñas usando contextos rotatorios."""
    results = []
    for i in range(num_examples):
        context = CONTEXTS[i % len(CONTEXTS)]
        messages = build_messages(context)

        print(f"  Generando ejemplo {i + 1}/{num_examples} — contexto: '{context}'")
        response = model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            repeat_penalty=1.1,    # Reduce repeticiones
            stop=["<|endoftext|>", "<|im_end|>", "[/INST]"],
        )

        output_text = response["choices"][0]["message"]["content"].strip()
        results.append(
            {
                "example_id": i + 1,
                "context": context,
                "prompt_system": DEFAULT_SYSTEM,
                "prompt_user": messages[-1]["content"],
                "output": output_text,
            }
        )
    return results


def save_dataset(dataset: List[Dict], model_path: str, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in dataset:
            record = {"model": Path(model_path).stem, **item}
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")
    print(f"Dataset guardado en {output_path} ({len(dataset)} ejemplos)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generador de dataset sintético con llama-cpp-python (CPU/AMD compatible)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Ruta al archivo .gguf del modelo (ej: ./models/llama-3.1-8b-q4_k_m.gguf)",
    )
    parser.add_argument("--output-file", type=str, required=True, help="Ruta de salida JSONL")
    parser.add_argument("--num-examples", type=int, default=20, help="Número de ejemplos")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument(
        "--n-threads",
        type=int,
        default=8,
        help="Hilos de CPU a usar (el Xeon E5-2670 v3 tiene 24 hilos, puedes subir hasta 20)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model = load_model(
        model_path=args.model_path,
        n_threads=args.n_threads,
    )

    print(f"\nGenerando {args.num_examples} ejemplos...")
    dataset = generate_examples(
        model=model,
        num_examples=args.num_examples,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    save_dataset(dataset, args.model_path, Path(args.output_file))


if __name__ == "__main__":
    main()