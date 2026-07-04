import argparse
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

PLACEHOLDER_MODEL = "distilgpt2"
MODEL_DIRS = {
    "llama": "./models/llama",
    "phi": "./models/phi",
    "mistral": "./models/mistral",
}


def create_local_model(model_name: str, target_dir: Path, dtype: str):
    print(f"Descargando modelo de prueba '{model_name}' en {target_dir}...")
    target_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=getattr(torch, dtype))
    tokenizer.save_pretrained(target_dir)
    model.save_pretrained(target_dir)
    print(f"Modelo guardado en {target_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Configura modelos locales de prueba para la tarea")
    parser.add_argument(
        "--placeholder-model",
        type=str,
        default=PLACEHOLDER_MODEL,
        help="Identificador de modelo de prueba a descargar desde Hugging Face",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float16", "float32"],
        help="Tipo de dato para la carga del modelo",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("Este script crea modelos locales de prueba en los directorios de `config.yaml`.")
    print("Reemplaza estos modelos de prueba con los pesos reales de LLaMA, Phi y Mistral cuando estén disponibles.")

    for name, path_str in MODEL_DIRS.items():
        target = Path(path_str)
        create_local_model(args.placeholder_model, target, args.dtype)

    print("\nListo. Los modelos locales de prueba están preparados en:")
    for path_str in MODEL_DIRS.values():
        print(f" - {path_str}")
    print("\nEjemplo de ejecución de generación:")
    print("python generate_dataset.py --model-id ./models/llama --output-file ./data/generated_llama.jsonl --num-examples 20")


if __name__ == '__main__':
    main()
