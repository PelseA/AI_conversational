import torch
import torch.nn as nn

class MeanPooling(nn.Module):
    """
    Стратегия Mean Pooling:
    Усредняет векторы всех токенов текста, игнорируя пустые токены [PAD]
    с помощью маски внимания (attention_mask).
    """
    def __init__(self):
        super().__init__()

    def forward(self, model_output, attention_mask):
        # model_output имеет размерность: [batch_size, seq_len, 768]
        token_embeddings = model_output
        
        # НАДЕЖНОЕ РАСШИРЕНИЕ МАСКИ:
        # Добавляем измерение в конец: [batch_size, seq_len, 1]
        input_mask_expanded = attention_mask.unsqueeze(-1)
        
        # Вместо .expand() используем умножение с автоматическим приведением типов (broadcasting)
        # Умножаем эмбеддинги [batch_size, seq_len, 768] на маску [batch_size, seq_len, 1]
        # Все PAD-токены гарантированно зануляются
        masked_embeddings = token_embeddings * input_mask_expanded.float()
        
        # Складываем векторы токенов вдоль оси текста (измерение 1) -> получаем [batch_size, 768]
        sum_embeddings = torch.sum(masked_embeddings, dim=1)
        
        # Считаем количество реальных (не PAD) токенов в каждой строке батча
        # Ограничиваем снизу микро-числом 1e-9, чтобы застраховаться от деления на ноль
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        
        # Возвращаем средний вектор текста: делим сумму векторов на количество реальных слов
        return sum_embeddings / sum_mask


class ClsPooling(nn.Module):
    """
    Стратегия CLS Pooling:
    Берет скрытое состояние только самого первого токена [CLS],
    который в BERT аккумулирует смысл всего текста.
    """
    def __init__(self):
        super().__init__()

    def forward(self, model_output, attention_mask=None):
        # model_output имеет размерность: [batch_size, seq_len, 768]
        # Используем .select(измерение, индекс), чтобы строго вытащить 
        # первый токен (индекс 0) вдоль оси seq_len (измерение 1)
        # Это гарантирует, что на выходе будет идеальный тензор [batch_size, 768]
        cls_embeddings = model_output.select(1, 0)
        
        return cls_embeddings
