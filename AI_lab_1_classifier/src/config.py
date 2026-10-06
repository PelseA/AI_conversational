import torch

class Config:
    # Имя модели на Hugging Face
    MODEL_NAME = "ai-forever/ruBert-base"
    
    # Путь к отфильтрованной базе данных
    DATA_PATH = "data/processed/gazeta_5_min_topics.csv"
    
    # Гиперпараметры текста и обучения
    MAX_LEN = 128           # Максимальная длина текста в токенах (оптимально для ruBert)
    BATCH_SIZE = 16         # Размер батча (если будет падать по памяти OOM, уменьшить до 8)
    EPOCHS = 3              # Количество эпох обучения
    LEARNING_RATE = 2e-5    # Стандартный шаг обучения для BERT моделей
    
    # Выбор устройства (видеокарта CUDA, Mac MPS или обычный процессор CPU)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else 
                          "mps" if torch.backends.mps.is_available() else "cpu")

print(f"Конфигурация загружена. Выбранное устройство: {Config.DEVICE}")
