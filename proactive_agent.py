import os
import asyncio
from pyrogram import Client
from pyrogram.enums import ChatType
from agent_core import generate_referral_message

# --- 1. Конфигурация ---

# Переменные окружения для авторизации Pyrogram
API_ID = os.environ.get("TG_API_ID")
API_HASH = os.environ.get("TG_API_HASH")
SESSION_NAME = "my_account" # Имя файла сессии

# --- 2. Функции Pyrogram ---

async def get_chat_history(client: Client, chat_id: int, limit: int = 10) -> str:
    """
    Извлекает последние сообщения из чата с контактом.
    """
    try:
        messages = []
        async for message in client.get_chat_history(chat_id, limit=limit):
            # Форматируем сообщение для передачи в LLM
            sender = "Я" if message.from_user.is_self else message.from_user.first_name
            messages.append(f"{sender}: {message.text}")
        
        # Возвращаем историю в обратном порядке (от старых к новым)
        return "\n".join(messages[::-1])
    except Exception as e:
        # print(f"Ошибка при получении истории чата с {chat_id}: {e}")
        return "История чата недоступна."

async def get_target_users(client: Client) -> list:
    """
    Динамически извлекает список целевых пользователей из контактов,
    личных чатов и групп/каналов.
    """
    target_users = []
    
    # 1. Извлечение контактов и личных чатов
    print("Извлечение контактов и личных чатов...")
    async for dialog in client.get_dialogs():
        # Фильтруем: только пользователи (не боты) и не сам пользователь
        if dialog.chat.type == ChatType.PRIVATE and not dialog.chat.is_bot and dialog.chat.id != client.me.id:
            # Проверяем, что у нас есть имя пользователя или ID для обращения
            if dialog.chat.username or dialog.chat.first_name:
                # Имитация получения информации о контакте и истории чата
                # В реальном приложении здесь будет логика фильтрации по дате активности
                
                # Реальный вызов Pyrogram для получения истории чата
                chat_history = await get_chat_history(client, dialog.chat.id)
                
                target_users.append({
                    "user_id": dialog.chat.id,
                    "username": dialog.chat.username,
                    "first_name": dialog.chat.first_name,
                    "contact_info": "Интересуется инвестициями в криптовалюту.",
                    "chat_history": chat_history
                })
        
        # 2. Извлечение пользователей из групп/каналов (только если вы администратор)
        # ВНИМАНИЕ: Массовая рассылка в группы может привести к бану.
        # Эта логика оставлена для демонстрации возможности, но требует осторожности.
        if dialog.chat.type in [ChatType.GROUP, ChatType.SUPERGROUP]:
            try:
                # Получаем участников группы (только для демонстрации)
                async for member in client.get_chat_members(dialog.chat.id):
                    if not member.user.is_bot and member.user.id != client.me.id:
                        # Проверяем, что пользователя еще нет в списке
                        if not any(u["user_id"] == member.user.id for u in target_users):
                            # Реальный вызов Pyrogram для получения истории чата
                            group_history = await get_chat_history(client, member.user.id)
                            
                            target_users.append({
                                "user_id": member.user.id,
                                "username": member.user.username,
                                "first_name": member.user.first_name,
                                "contact_info": f"Участник группы '{dialog.chat.title}'",
                                "chat_history": group_history
                            })
            except Exception as e:
                # Игнорируем ошибки, если нет прав на просмотр участников
                # print(f"Не удалось получить участников чата {dialog.chat.title}: {e}")
                pass
                
    # Удаляем дубликаты по user_id
    unique_users = {user["user_id"]: user for user in target_users}.values()
    return list(unique_users)

async def main():
    """
    Основная функция для запуска агента.
    """
    if not API_ID or not API_HASH:
        print("КРИТИЧЕСКАЯ ОШИБКА: Не установлены переменные окружения TG_API_ID и TG_API_HASH.")
        return

    # Инициализация клиента Pyrogram
    app = Client(SESSION_NAME, api_id=API_ID, api_hash=API_HASH)

    async with app:
        print(f"Авторизация успешна. Пользователь: @{app.me.username} ({app.me.first_name})")
        
        # Динамическое получение списка целевых пользователей
        target_users = await get_target_users(app)
        
        if not target_users:
            print("Не найдено целевых пользователей для проактивной рассылки.")
            return

        print(f"Найдено {len(target_users)} уникальных целевых пользователей.")
        
        for user in target_users:
            user_id = user["user_id"]
            contact_info = user["contact_info"]
            chat_history = user["chat_history"]
            
            print(f"\n--- Обработка пользователя {user.get('username', user_id)} ---")
            
            # 1. Генерация персонализированного сообщения
            message_text = generate_referral_message(contact_info, chat_history)
            
            if message_text.startswith("КРИТИЧЕСКАЯ ОШИБКА"):
                print(f"Пропуск отправки: {message_text}")
                continue
            
            print(f"Сгенерированное сообщение:\n{message_text}")
            
            # 2. Отправка сообщения (закомментировано для безопасности)
            # ВНИМАНИЕ: Раскомментируйте строку ниже ТОЛЬКО после тщательного тестирования!
            # try:
            #     await app.send_message(user_id, message_text)
            #     print(f"Сообщение успешно отправлено пользователю {user.get('username', user_id)}.")
            # except Exception as e:
            #     print(f"Ошибка отправки сообщения пользователю {user.get('username', user_id)}: {e}")
            
            # Имитация задержки для предотвращения бана
            await asyncio.sleep(5) 

if __name__ == "__main__":
    # Запуск асинхронной функции
    asyncio.run(main())
