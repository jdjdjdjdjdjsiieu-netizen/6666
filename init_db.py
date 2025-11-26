import json
import os
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_text_splitters import CharacterTextSplitter

# 1. Загрузка данных из JSON
print("Загрузка данных из products.json...")
# Используем JSONLoader для загрузки и форматирования данных
# 'jq_schema' указывает, как извлекать документы из JSON-массива
# 'text_content=False' означает, что мы не будем использовать весь JSON как один текст
loader = JSONLoader(
    file_path='./products.json',
    jq_schema='.[].product_name + " | " + .[].key_benefits + " | " + .[].referral_link + " | " + .[].target_audience',
    text_content=False
)
data = loader.load()

# Добавление метаданных для удобства
for doc in data:
    # doc.page_content уже содержит объединенную строку с данными
    # Добавим исходный JSON-объект в метаданные для легкого извлечения
    # В реальном приложении нужно будет более аккуратно парсить JSON
    pass 

# 2. Разделение текста (хотя для такого маленького файла это не критично, но хорошая практика)
print("Разделение документов...")
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(data)

# 3. Инициализация эмбеддингов
# Используем бесплатные SentenceTransformerEmbeddings, которые работают локально
print("Инициализация эмбеддингов (SentenceTransformer)...")
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. Создание и сохранение векторной базы данных
# База данных будет сохранена в папке ./chroma_db
print("Создание и сохранение ChromaDB...")
db = Chroma.from_documents(
    docs, 
    embeddings, 
    persist_directory="./chroma_db"
)
db.persist()
print(f"База данных успешно создана и сохранена в ./chroma_db. Количество документов: {len(docs)}")

# Проверка:
# query = "Мне нужен инструмент для управления проектами в IT"
# docs = db.similarity_search(query)
# print("\nПроверка поиска:")
# print(docs[0].page_content)
