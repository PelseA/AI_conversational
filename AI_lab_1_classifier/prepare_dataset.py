import os
import requests
import pandas as pd
from datasets import load_dataset

# Отключаем SSL для безопасного чтения кэша
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["PYTHONHTTPSVERIFY"] = "0"

def prepare_data():
    # 7 топ тем
    #TARGET_TOPICS = {'sport', 'social', 'politics', 'business', 'culture', 'army', 'auto'}
    # Наш целевой список 5 тем с наименьшими записями
    TARGET_TOPICS = {'tech', 'financial', 'science', 'auto', 'army'}
    
    print("Загружаем исходные данные...")
    # Датасет уже должен быть загружен, подтянется из кэша мгновенно
    dataset = load_dataset('IlyaGusev/gazeta')
    
    # Объединяем все части (train, validation, test) в один датафрейм, 
    # чтобы собрать полноценную кастомную базу данных
    df_all = pd.concat([
        pd.DataFrame(dataset['train']),
        pd.DataFrame(dataset['validation']),
        pd.DataFrame(dataset['test'])
    ], ignore_index=True)
    
    print("Извлекаем темы из URL...")
    # Функция безопасного извлечения темы
    def extract_topic(url):
        parts = url.split('/')
        return parts[3] if len(parts) > 3 else 'unknown'
    
    df_all['topic'] = df_all['url'].apply(extract_topic)
    
    print("Фильтруем датасет по 5 минимальным темам...")
    # Оставляем только строки, где тема входит в наш топ-7
    df_filtered = df_all[df_all['topic'].isin(TARGET_TOPICS)].copy()
    
    # Оставляем только нужные колонки: text, title, url, topic
    df_filtered = df_filtered[['text', 'title', 'url', 'topic']]
    
    # Создаем папку data/processed, если её ещё нет
    os.makedirs('data/processed', exist_ok=True)
    
    # Сохраняем итоговый датасет в формате CSV
    output_path = 'data/processed/gazeta_5_min_topics.csv'
    df_filtered.to_csv(output_path, index=False, encoding='utf-8')
    
    print("--- Подготовка завершена! ---")
    print(f"Новая база данных сохранена в: {output_path}")
    print(f"Итоговый размер датасета: {len(df_filtered)} строк.")
    print("\nРаспределение по темам в собранной базе:")
    print(df_filtered['topic'].value_counts())

if __name__ == "__main__":
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    prepare_data()
