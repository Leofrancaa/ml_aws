import joblib
from pathlib import Path 
import pandas as pd 
import logging

from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "comentarios.csv"
OUTPUT_MODEL = ROOT / "models"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def load_data(data_path: Path) -> pd.DataFrame:
        """Load data from a CSV file."""
        df = pd.read_csv(data_path)
        return df["texto"].tolist(), df['sentimento'].tolist()

def main():
    x, y = load_data(DATA)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('tfidf',TfidfVectorizer(strip_accents='unicode', ngram_range=(1,2))),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ])

    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")

    OUTPUT_MODEL.mkdir(exist_ok=True)
    joblib.dump(pipeline, OUTPUT_MODEL / "modelo_sentimento.joblib")

    logger.info(f"Model saved to {OUTPUT_MODEL / 'modelo_sentimento.joblib'}")

if __name__ == "__main__":
    main()