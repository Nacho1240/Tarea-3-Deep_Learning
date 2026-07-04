import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List

KEYWORDS = ['auriculares', 'inalámbricos', 'altavoz', 'altavoces', 'sonido', 'comodidad', 'batería', 'diseño']


def load_jsonl(path: Path) -> List[Dict]:
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data.append(json.loads(line))
    return data


def distinct_ratio(texts: List[str]) -> float:
    unique = len(set(texts))
    return unique / len(texts) if texts else 0.0


def format_compliance(text: str) -> bool:
    return all(tag in text for tag in ['Reseña:', 'Sentimiento:', 'Puntaje:'])


def keyword_relevance(text: str) -> int:
    lower = text.lower()
    return sum(1 for keyword in KEYWORDS if keyword in lower)


def parse_sentiment(text: str) -> str:
    match = re.search(r'Sentimiento:\s*(Positivo|Negativo|Neutro)', text, re.IGNORECASE)
    return match.group(1).capitalize() if match else 'No detectado'


def average_length(texts: List[str]) -> float:
    lengths = [len(t.split()) for t in texts]
    return sum(lengths) / len(lengths) if lengths else 0.0


def summarize(dataset: List[Dict]) -> Dict:
    outputs = [item['output'] for item in dataset]
    compliance = [format_compliance(output) for output in outputs]
    relevance_scores = [keyword_relevance(output) for output in outputs]
    sentiments = [parse_sentiment(output) for output in outputs]
    sentiment_counts = Counter(sentiments)

    return {
        'total_examples': len(outputs),
        'distinct_ratio': distinct_ratio(outputs),
        'format_compliance_rate': sum(compliance) / len(compliance) if compliance else 0.0,
        'average_relevance_keywords': sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0,
        'average_length_words': average_length(outputs),
        'sentiment_counts': dict(sentiment_counts),
    }


def print_summary(name: str, summary: Dict):
    print(f"\n=== Resumen para {name} ===")
    print(f"Total de ejemplos: {summary['total_examples']}")
    print(f"Diversidad de ejemplos (distinct ratio): {summary['distinct_ratio']:.2f}")
    print(f"Cumplimiento de formato: {summary['format_compliance_rate']:.2f}")
    print(f"Palabras clave detectadas por ejemplo: {summary['average_relevance_keywords']:.2f}")
    print(f"Longitud promedio (palabras): {summary['average_length_words']:.1f}")
    print(f"Conteo de sentimientos: {summary['sentiment_counts']}\n")


def parse_args():
    parser = argparse.ArgumentParser(description='Evalúa datasets generados por modelos locales')
    parser.add_argument('--files', type=str, nargs='+', required=True, help='Rutas de archivos JSONL generados por modelo')
    return parser.parse_args()


def main():
    args = parse_args()
    for path_str in args.files:
        path = Path(path_str)
        dataset = load_jsonl(path)
        summary = summarize(dataset)
        print_summary(path.name, summary)


if __name__ == '__main__':
    main()
