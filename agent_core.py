import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint

from gigachain.chat_models import GigaChat

# --- 1. Инициализация LLM-провайдеров ---

# Основной провайдер: GigaChat
# ВНИМАНИЕ: Для работы необходимо получить токен авторизации GigaChat
# и установить его в переменную окружения GIGACHAT_CREDENTIALS
llm_primary = GigaChat(
    credentials=os.environ.get("GIGACHAT_CREDENTIALS"),
    verify_ssl_certs=False, # Временное решение для обхода SSL-ошибок в песочнице
    scope="GIGACHAT_API_PERS"
)

# Резервный провайдер 1: Hugging Face Inference API (строго бесплатный)
# Используем небольшую, быструю модель.
# ВНИМАНИЕ: Для работы необходимо получить токен Hugging Face и установить его в переменную окружения HF_TOKEN
llm_secondary = HuggingFaceEndpoint(
    repo_id="facebook/bart-large-cnn", # Модель, оптимизированная для суммаризации и извлечения информации
    task="summarization",
    huggingfacehub_api_token=os.environ.get("HF_TOKEN"),
    temperature=0.7,
    max_new_tokens=100
)

# --- 2. Инициализация RAG (без изменений) ---

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

# --- 3. Создание шаблона промпта (без изменений) ---

prompt_template = """
Ты - ИИ-агент, работающий от имени пользователя для реферального маркетинга в Telegram.
Твоя задача - проактивно рекомендовать реферальные продукты из предоставленной базы знаний.
Твой стиль общения должен быть дружелюбным, но профессиональным, имитируя стиль твоего владельца.

Контекст о контакте: {contact_info}
История чата с контактом (для анализа интересов): {chat_history}
База знаний (наиболее релевантный продукт): {context}

Инструкция:
1. Проанализируй историю чата и информацию о контакте, чтобы выявить его текущие потребности и интересы.
2. Сформулируй короткое, персонализированное сообщение, которое начинается с вопроса или утверждения, связанного с интересами контакта.
3. Включи в сообщение ключевые преимущества продукта и реферальную ссылку.
4. Сообщение должно быть не более 4-5 предложений.

Сообщение для контакта:
"""

prompt = ChatPromptTemplate.from_template(prompt_template)

# --- 4. Создание цепочек RAG для каждого провайдера ---

rag_chain_primary = (
    {
        "context": retriever,
        "contact_info": lambda x: x["contact_info"],
        "chat_history": lambda x: x["chat_history"]
    }
    | prompt
    | llm_primary
    | StrOutputParser()
)

rag_chain_secondary = (
    {
        "context": retriever,
        "contact_info": lambda x: x["contact_info"],
        "chat_history": lambda x: x["chat_history"]
    }
    | prompt
    | llm_secondary
    | StrOutputParser()
)

# --- 5. Функция-оркестратор с отказоустойчивостью ---

def generate_referral_message(contact_info: str, chat_history: str) -> str:
    """
    Генерирует персонализированное реферальное сообщение с использованием
    стратегии резервирования (GigaChat -> Hugging Face -> Шаблон).
    """
    search_query = f"Интересы контакта: {contact_info}. История чата: {chat_history}"
    payload = {
        "contact_info": contact_info,
        "chat_history": chat_history,
        "query": search_query
    }
    
    # 1. Попытка: GigaChat (Primary)
    try:
        print("Попытка генерации через GigaChat (Primary)...")
        message = rag_chain_primary.invoke(payload)
        print("GigaChat успешно сгенерировал сообщение.")
        return message
    except Exception as e_primary:
        print(f"Ошибка при работе с GigaChat: {e_primary}")
        print("Переключение на резервный провайдер (Hugging Face)...")
        
        # 2. Попытка: Hugging Face (Secondary)
        try:
            message = rag_chain_secondary.invoke(payload)
            print("Резервный провайдер (Hugging Face) успешно сгенерировал сообщение.")
            return message
        except Exception as e_secondary:
            print(f"Ошибка при работе с Hugging Face: {e_secondary}")
            print("Переключение на Шаблонную генерацию (Fallback)...")
            
            # 3. Попытка: Шаблонная генерация (Fallback)
            try:
                # Извлекаем контекст для шаблонного сообщения
                context = retriever.invoke(search_query)[0].page_content
                
                # Простая генерация по шаблону (гарантированно бесплатно)
                template_message = f"Привет! Я заметил, что ты интересовался темой, связанной с '{contact_info}'. Хочу порекомендовать тебе продукт: {context}. Это может быть полезно! [Твоя Реферальная Ссылка]"
                print("Шаблонная генерация успешно выполнена.")
                return template_message
            except Exception as e_fallback:
                print(f"КРИТИЧЕСКАЯ ОШИБКА: Сбой даже при шаблонной генерации: {e_fallback}")
                return "КРИТИЧЕСКАЯ ОШИБКА: Все сервисы генерации недоступны. Сообщение не отправлено."

# --- Демонстрационный запуск (без изменений) ---

if __name__ == "__main__":
    test_contact_1 = "Имя: Анна, Интересы: Фриланс, приватность данных, облачные хранилища."
    test_history_1 = "Анна: Привет! Я недавно потеряла важные файлы, ищу надежное место для хранения с хорошим шифрованием."
    
    test_contact_2 = "Имя: Олег, Интересы: Управление IT-проектами, стартапы, автоматизация."
    test_history_2 = "Олег: Привет! Наш стартап растет, но мы тратим слишком много времени на ручное планирование задач в Jira."

    print(f"--- Генерация для Анны (с историей чата) ---")
    message_anna = generate_referral_message(test_contact_1, test_history_1)
    print(f"\nИтоговое сообщение:\n{message_anna}")

    print(f"\n--- Генерация для Олега (с историей чата) ---")
    message_oleg = generate_referral_message(test_contact_2, test_history_2)
    print(f"\nИтоговое сообщение:\n{message_oleg}")
