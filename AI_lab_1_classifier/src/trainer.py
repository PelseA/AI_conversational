import torch
import numpy as np
from tqdm import tqdm
from src.metrics import calculate_metrics
from src.config import Config

def train_epoch(model, data_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    
    for batch in tqdm(data_loader, desc="Training"):
        optimizer.zero_grad()
        
        # Переносим батч на GPU/CPU
        input_ids = batch['input_ids'].to(Config.DEVICE)
        attention_mask = batch['attention_mask'].to(Config.DEVICE)
        labels = batch['label'].to(Config.DEVICE)
        
        # Прямой проход (Forward)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)
        
        # Обратный проход (Backward)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(data_loader)

def evaluate(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(Config.DEVICE)
            attention_mask = batch['attention_mask'].to(Config.DEVICE)
            labels = batch['label'].to(Config.DEVICE)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # Получаем индексы самых вероятных классов
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
    # Считаем итоговые метрики по всему валидационному датасету
    metrics = calculate_metrics(np.array(all_labels), np.array(all_preds))
    metrics["Loss"] = round(total_loss / len(data_loader), 4)
    
    return metrics
