from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def calculate_metrics(y_true, y_pred):
    """
    Вычисляет Accuracy, Precision, Recall и F1-score для мультиклассовой классификации.
    """
    # Считаем Precision, Recall и F1 одновременно с взвешиванием по объему классов
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    
    return {
        "Accuracy": round(acc, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1-score": round(f1, 4)
    }
