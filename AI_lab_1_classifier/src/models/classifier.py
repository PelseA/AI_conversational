import torch
import torch.nn as nn
from transformers import AutoModel
from src.config import Config

class BertClassifier(nn.Module):
    """
    Универсальный классификатор на базе ruBert-base.
    Принимает готовую стратегию пулинга (Mean или CLS) при инициализации.
    """
    def __init__(self, pooling_layer, num_classes=7):
        super().__init__()
        
        # 1. Загружаем модель векторизации от Сбера из Hugging Face
        # (Используем AutoModel, чтобы загрузить чистый BERT без встроенных голов классификации)
        self.bert = AutoModel.from_pretrained(Config.MODEL_NAME)
        
        # 2. Подключаем переданный слой пулинга (MeanPooling или ClsPooling)
        self.pooling = pooling_layer
        
        # 3. Финальный линейный классификатор 
        # ruBert-base выдает векторы размерностью 768. Мы переводим их в 7 классов тем новостей.
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        # Прогоняем токены через ruBert-base
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # outputs.last_hidden_state содержит матрицу векторов всех токенов
        # Размерность: [batch_size, seq_len, 768]
        hidden_states = outputs.last_hidden_state
        
        # Сжимаем матрицу токенов в один вектор текста с помощью выбранного пулинга
        # Размерность после пулинга: [batch_size, 768]
        pooled_output = self.pooling(hidden_states, attention_mask)
        
        # Отправляем вектор текста в линейный слой для предсказания темы
        # Размерность на выходе: [batch_size, 7]
        logits = self.fc(pooled_output)
        
        return logits
