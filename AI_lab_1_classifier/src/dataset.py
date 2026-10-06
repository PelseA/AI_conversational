import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from src.config import Config

# Отключаем SSL
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["PYTHONHTTPSVERIFY"] = "0"

class GazetaDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def get_data_loaders(sample_size=8000):
    print("Загружаем отфильтрованную базу данных...")
    df = pd.read_csv(Config.DATA_PATH)

    # Переводим текстовые темы в числа (0-6)
    unique_topics = sorted(df['topic'].unique())
    topic2idx = {topic: idx for idx, topic in enumerate(unique_topics)}
    df['label'] = df['topic'].map(topic2idx)
    
    print(f"Словарь тем зафиксирован: {topic2idx}")

    # БЕЗОПАСНОЕ СЭМПЛИРОВАНИЕ ДЛЯ СТАНДАРТНЫХ ТИПОВ PYTHON
    if sample_size and len(df) > sample_size:
        size_per_topic = sample_size // len(unique_topics)
        
        # Сэмплируем данные
        df = df.groupby('topic', group_keys=False).apply(
            lambda x: x.sample(min(len(x), size_per_topic), random_state=42)
        )
        print(f"✅ Датасет успешно сэмплирован! Выборка составляет {len(df)} строк.")

    # КРИТИЧЕСКИЙ ШАГ: Принудительно конвертируем в чистые списки строк и чисел Python,
    # чтобы полностью стереть любые следы скрытых типов данных PyArrow
    X = [str(text) for text in df['text'].tolist()]
    y = [int(label) for label in df['label'].tolist()]

    # Разделяем на обучающую (80%) и валидационную (20%) выборки
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Инициализируем токенизатор Сбера...")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Создаем объекты датасетов PyTorch
    train_dataset = GazetaDataset(X_train, y_train, tokenizer, Config.MAX_LEN)
    val_dataset = GazetaDataset(X_val, y_val, tokenizer, Config.MAX_LEN)

    # Оборачиваем в DataLoader для нарезки на батчи
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    print(f"Данные готовы. Батчей в Train: {len(train_loader)}, в Val: {len(val_loader)}")
    return train_loader, val_loader
