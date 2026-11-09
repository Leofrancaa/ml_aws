import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "modelo_sentimento.joblib"

FRASES_EXEMPLO = [
    "Adorei o produto, chegou rápido e funciona muito bem!",
    "Péssimo atendimento, nunca mais compro aqui.",
    "O produto é ok, nada de especial.",
    "Qualidade excelente, recomendo a todos.",
    "Veio quebrado e o suporte não resolveu.",
]


def predict(model, frases: list[str]) -> None:
    preds = model.predict(frases)
    probas = model.predict_proba(frases) if hasattr(model, "predict_proba") else None

    for i, (frase, pred) in enumerate(zip(frases, preds)):
        conf = f" ({probas[i].max():.1%})" if probas is not None else ""
        print(f"[{pred}]{conf}  {frase}")


def main():
    model = joblib.load(MODEL_PATH)

    # Frases passadas na linha de comando: python src/inference.py "frase 1" "frase 2"
    if len(sys.argv) > 1:
        predict(model, sys.argv[1:])
        return

    predict(model, FRASES_EXEMPLO)

    # Modo interativo
    print("\nDigite uma frase (ou Enter vazio para sair):")
    while frase := input("> ").strip():
        predict(model, [frase])


if __name__ == "__main__":
    main()
