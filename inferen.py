import torch
import json
import pickle
import numpy as np
from huggingface_hub import hf_hub_download
from types import SimpleNamespace

# --- Настройки ---
REPO_ID = "Neweret/SimpleClassifier-85k"

# --- Скачиваем необходимые файлы с HF ---
config_path = hf_hub_download(REPO_ID, "config.json")
weights_path = hf_hub_download(REPO_ID, "pytorch_model.bin")
vectorizer_path = hf_hub_download(REPO_ID, "vectorizer.pkl")
svd_path = hf_hub_download(REPO_ID, "svd.pkl")  # если есть
le_path = hf_hub_download(REPO_ID, "label_encoder.pkl")

# --- Загружаем конфиг ---
with open(config_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)
cfg = SimpleNamespace(**cfg)

# --- Динамическая модель ---
class SimpleClassifier(torch.nn.Module):
    def __init__(self, input_dim, num_classes, p_dropout=0.3):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_dim, 256)
        self.ln1 = torch.nn.LayerNorm(256)
        self.dropout = torch.nn.Dropout(p_dropout)
        self.linear2 = torch.nn.Linear(256, 128)
        self.ln2 = torch.nn.LayerNorm(128)
        self.linear_out = torch.nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.nn.functional.gelu(self.ln1(self.linear1(x)))
        x = self.dropout(x)
        x = torch.nn.functional.gelu(self.ln2(self.linear2(x)))
        x = self.dropout(x)
        return self.linear_out(x)

# --- Создаём модель и загружаем веса ---
model = SimpleClassifier(cfg.input_dim, cfg.num_classes, cfg.p_dropout)
state_dict = torch.load(weights_path, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# --- Загружаем препроцессоры ---
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

vectorizer = load_pickle(vectorizer_path)
svd = load_pickle(svd_path) if svd_path else None
le = load_pickle(le_path)

# --- Предобработка текста ---
def preprocess_text(text: str) -> np.ndarray:
    X_vec = vectorizer.transform([text])
    if svd:
        X_vec = svd.transform(X_vec)
    return X_vec.astype(np.float32)

# --- Предсказание ---
def predict(text: str) -> str:
    X = preprocess_text(text)
    xb = torch.from_numpy(X).float()
    with torch.inference_mode():
        logits = model(xb)
        pred = torch.argmax(logits, dim=1).cpu().numpy()[0]
    return le.inverse_transform([pred])[0]

# --- Пример ---
if __name__ == "__main__":
    sample_text = "Как меня зовут?"
    print("Prediction:", predict(sample_text))
