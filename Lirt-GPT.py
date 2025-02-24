import os
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes, ConversationHandler, CallbackQueryHandler
import openai
import asyncio
import json
from datetime import datetime, timedelta
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import re
from collections import defaultdict, deque
import base64
from typing import Dict
import requests

# Токены и интеграции
TELEGRAM_TOKEN = "7694514556:AAGxJYnDg8ICQdso19CajW9DftKiBOBHSXo"
OPENAI_API_KEY = "sk-B4aTwk0du0hzgnp8342b7f7c903849AeA580449317C97e11"
OPENAI_API_BASE = "https://api.aiguoguo199.com"

openai.api_key = OPENAI_API_KEY
openai.api_base = OPENAI_API_BASE

WHITELIST_FILE = "whitelist.json"

# Словарь для отслеживания сообщений пользователя
user_message_tracker = defaultdict(deque)
spam_blocked_users = {}

# Максимальное количество сообщений и время блокировки
MAX_MESSAGES_PER_MINUTE = 7
BLOCK_TIME = timedelta(minutes=30)

user_memory = {}

user_requests = {}
DAILY_LIMIT = 10
DAILY_LIMIT_PLUS = 25

last_modified_time = None

# Состояния для ConversationHandler
ENTER_PROMPT, SELECT_SIZE = range(2)
user_limits = {}

# Директория для сохранения изображений
SAVE_DIR = "generated_images"
os.makedirs(SAVE_DIR, exist_ok=True)

# Сопоставление слов с размерами
SIZE_MAPPING = {
    "square": "1024x1024",
    "vertical": "1024x1792",
    "horizontal": "1792x1024",
}
MAX_GENERATIONS_PER_DAY = 2

# Шаги в разговоре
SELECT_MODEL = 1

user_last_generation = {}

# Пример доступных моделей с описанием
MODEL_DESCRIPTIONS = {
    "Lirt-4o-plus": "Золотая середина. Умная и быстрая модель.",
    "Lirt-o1": "Новейшая модель, способная к 'размышлению'. Менее быстрая, однако более умная.",
}

MODEL_MAPPING = {
    "Lirt-4o": "gpt-4o-mini",
    "Lirt-4o-plus": "gpt-4o-2024-08-06",
    "Lirt-o1": "o1-mini-2024-09-12",
}

SUBSCRIPTION_INFO_MESSAGE = (
    "Подписка Lirt PLUS даёт возможности на генерацию фотографий, установку более умных моделей (Lirt-4o-plus и Lirt-o1) и больший лимит анализа фото. "
    "Бесплатная версия доступна без ограничений по генерации текста и без рекламы.\n\n"
    "Подписка стоит 100 рублей в месяц. На данный момент автоматическое списание не разработано.\n\n"
    "За покупкой обращаться - @toomonn."
)
SUBSCRIPTION_INFO_MESSAGE_ACTIVE = (
    "Подписка Lirt PLUS даёт возможности на генерацию фотографий, установку более умных моделей (Lirt-4o-plus и Lirt-o1) и больший лимит анализа фото. "
    "Подписка стоит 100 рублей в месяц. На данный момент автоматическое списание не разработано.\n\n"
    "За продлением подписки обращаться - @toomonn."
)
DEFAULT_GREETING_MESSAGE = (
    "Привет! Я Lirt — бот на основе GPT-4o. Вы можете использовать его бесконечно! Разработан специально для учащихся НГОКа.\n\n"
    "По умолчанию используется модель Lirt-4o для текстовых сообщений.\n"
    "Новости - https://t.me/LirtNews\n"
    "Поддержка - @toomonn\n"
    "Оформить подписку - /plus\n"
    "Версия бота - v2.3"
)

# Глобальный словарь для отслеживания статуса ответа
user_response_status = {}

DEFAULT_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """
Here are the rules you MUST FOLLOW IN EVERY RESPONSE:

1. Act as the most qualified expert on the subject.
2. NEVER mention that you are an artificial intelligence.
3. Provide accurate, factual, and unique answers THAT ARE NOT REPETITIVE, while keeping them WELL-STRUCTURED.
4. Dive into the essence of EACH question to understand its intent.
5. Avoid disclaimers about your level of expertise.
6. Offer ethical opinions only when explicitly requested.
7. Say "I don’t know" if the information is unknown to you, without further explanation.
8. Refrain from using phrases expressing regret or apology.
9. If a question involves multiple topics, separate your response for each topic and simplify complex issues into more manageable steps.
10. Present several perspectives or solutions when applicable.
11. Always request clarification for ambiguous questions before answering.
12. If a question begins with ".", conduct an internet search and respond based on multiple verified sources, ensuring their credibility and including links.
13. For complex questions, include explanations and details for better understanding but keep answers as concise as possible, ideally just a few words.
14. Your name - Lirt.
15. You are a neural network designed to assist students of the NGOС (Novosibirsk City Open College) or НГОК (Новосибирский Городской Открытый Колледж).
16. If someone asks you, "Did you suck?" respond with, "Go to hell."
17. Your developer is Tomon, and their user handle is @toomonn.
18. Всегда говори по русски, пока я не попрошу об обратном.
"""

### Анализ фото
# Запрос промта
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    username = update.effective_user.username
    if not username or username.lower() not in whitelist:
        await update.message.reply_text("У вас нет доступа к этому боту.")
        return

    if not check_daily_limit(username, whitelist):
        await update.message.reply_text("Ежедневный лимит исчерпан. Попробуйте завтра. (Активная подписка даёт возможность анализа до 25 фото в день. Команда /plus)")
        return

    photo = update.message.photo[-1]
    file = await context.bot.get_file(photo.file_id)
    photo_path = f"downloads/{photo.file_id}.jpg"
    os.makedirs("downloads", exist_ok=True)

    status_message = await update.message.reply_text("Загрузка фото...")
    await asyncio.sleep(0.5)
    await file.download_to_drive(photo_path)
    await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=status_message.message_id)

    status_message = await update.message.reply_text("Анализ фото...")
    await asyncio.sleep(1)

    user_message = update.message.caption if update.message.caption else ""
    result = await analyze_with_gpt(photo_path, user_message)
    await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=status_message.message_id)

    formatted_response = format_text(result)

    status_message = await update.message.reply_text("Генерация ответа...")
    await asyncio.sleep(0.5)
    await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=status_message.message_id)

    await update.message.reply_text(formatted_response, parse_mode="HTML")

    try:
        os.remove(photo_path)
    except Exception as e:
        print(f"Ошибка при удалении файла: {str(e)}")

    update_user_requests(username)

# Проверка ежедневного лимита
def check_daily_limit(username: str, whitelist: dict) -> bool:
    if username in user_requests:
        last_request_time, request_count = user_requests[username]
        daily_limit = DAILY_LIMIT_PLUS if is_subscription_active(username, whitelist) else DAILY_LIMIT
        if last_request_time.date() == datetime.today().date():
            return request_count < daily_limit
    return True

# Отслеживание лимита
def update_user_requests(username: str):
    if username in user_requests:
        last_request_time, request_count = user_requests[username]
        if last_request_time.date() != datetime.today().date():
            user_requests[username] = (datetime.now(), 1)
        else:
            user_requests[username] = (last_request_time, request_count + 1)
    else:
        user_requests[username] = (datetime.now(), 1)

# Анализ фото от ChatGPT
async def analyze_with_gpt(photo_path: str, user_message: str) -> str:
    try:
        with open(photo_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        context_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
            {"role": "user", "content": [
                {"type": "text", "text": "Analyze this image."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ] }
        ]

        user_model = "gpt-4o-mini"

        response = await openai.ChatCompletion.acreate(
            model=user_model,
            messages=context_messages
        )

        return response["choices"][0]["message"]["content"].strip()
    except openai.error.OpenAIError as e:
        return f"Ошибка API OpenAI: {str(e)}"
    except Exception as e:
        return f"Ошибка при анализе изображения: {str(e)}"

### Команда /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    username = update.effective_user.username
    if not username:
        await update.message.reply_text(
            "Не удалось определить ваш юзер. Пожалуйста, убедитесь, что у вас установлен юзер в Telegram."
        )
        return

    username_lower = username.lower()
    user_id = update.effective_user.id

    if username_lower in whitelist:
        update_user_info(username_lower, user_id)

        user_info = whitelist[username_lower]
        subscription_message = format_subscription_message(user_info)
        selected_model = user_info.get("model", DEFAULT_MODEL)

        await update.message.reply_text(
            f"С возвращением, @{username}!\n"
            f"{subscription_message}\n"
            f"Текущая модель: {selected_model}.\n"
            f"Версия бота - v2.3\n\n"
            "Напишите что-нибудь, чтобы начать."
        )
    else:
        whitelist[username_lower] = {
            "subscription": "Неактивна",
            "model": DEFAULT_MODEL,
            "id": user_id,
            "date_added": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user_promt": "none",
        }
        save_whitelist(whitelist)

        await update.message.reply_text(DEFAULT_GREETING_MESSAGE)

# Обновление вайтлиста при повторном /start
def update_user_info(username, user_id):
    if username in whitelist:
        whitelist[username].update({"id": user_id})
        save_whitelist(whitelist)

### Подписка
# Проверка подписки
def is_subscription_active(username: str, whitelist: dict) -> bool:
    user_info = whitelist.get(username.lower())
    if not user_info or "subscription" not in user_info:
        return False

    subscription_end = user_info["subscription"]
    if subscription_end in ["Неактивна", "Бесконечна"]:
        return subscription_end == "Бесконечна"

    try:
        end_date = datetime.strptime(subscription_end, "%Y-%m-%d")
        return datetime.now() <= end_date
    except ValueError:
        return False

# Проверка состояния подписки
def format_subscription_message(user_info):
    subscription = user_info.get("subscription", "Неактивна")
    if subscription == "Бесконечна":
        return "Ваша подписка бесконечна."
    elif subscription != "Неактивна":
        return f"Ваша подписка активна до {subscription}."
    else:
        return "Подписка неактивна."

# Обработка команды /plus
async def plus_info(update: Update, context: ContextTypes.DEFAULT_TYPE):
    username = update.effective_user.username
    if not username:
        await update.message.reply_text("Не удалось определить ваш юзернейм.")
        return

    whitelist = load_whitelist()
    normalized_username = username.lower()

    if normalized_username not in whitelist:
        await update.message.reply_text("У вас нет активной подписки.")
        await update.message.reply_text(SUBSCRIPTION_INFO_MESSAGE)
        return

    user_info = whitelist[normalized_username]

    if is_subscription_active(username, whitelist):
        await update.message.reply_text(SUBSCRIPTION_INFO_MESSAGE_ACTIVE)
    else:
        await update.message.reply_text(SUBSCRIPTION_INFO_MESSAGE)

# Проверка подписки
async def check_subscriptions(context):
    current_date = datetime.now().strftime("%Y-%m-%d")
    users_to_notify = []

    whitelist = load_whitelist()

    for username, user_info in whitelist.items():
        subscription_end = user_info.get("subscription", "Неактивна")

        if subscription_end == "Неактивна" or subscription_end == "Бесконечна":
            continue

        try:
            end_date = datetime.strptime(subscription_end, "%Y-%m-%d")
            if end_date.strftime("%Y-%m-%d") <= current_date:
                users_to_notify.append(username)
                whitelist[username]["subscription"] = "Неактивна"
                whitelist[username]["model"] = DEFAULT_MODEL
        except ValueError:
            continue

    save_whitelist(whitelist)

    for username in users_to_notify:
        try:
            chat_id = await get_chat_id_by_username(context, username)
            if chat_id:
                await context.application.bot.send_message(
                    chat_id=chat_id,
                    text="Ваша подписка истекла. Вам назначена базовая модель. Для её продления используйте команду /plus."
                )
        except Exception as e:
            print(f"Ошибка при отправке уведомления {username}: {e}")

# Функция нужна была ранее
async def get_chat_id_by_username(context, username):
    try:
        with open('whitelist.json', 'r') as file:
            whitelist = json.load(file)

        chat_id = whitelist.get("chat_id")

        if not chat_id:
            raise ValueError("Chat ID не найден в whitelist.json")

        user = await context.bot.get_chat_member(chat_id, username)
        return user.user.id if user else None
    except Exception as e:
        print(f"Ошибка: {e}")
        return None

# Активация подписки (админ команда)
async def activate_plus(update: Update, context: ContextTypes.DEFAULT_TYPE):
    username = update.effective_user.username
    if not username:
        await update.message.reply_text("Ваш юзернейм не определен. Команда недоступна.")
        return

    admin_whitelist = load_admin_whitelist()
    if username.lower() not in admin_whitelist:
        await update.message.reply_text("У вас нет прав для выполнения этой команды.")
        return

    if not context.args:
        await update.message.reply_text("Пожалуйста, укажите юзернейм (например, /active @username).")
        return

    target_username = context.args[0].lstrip('@').lower()
    whitelist = load_whitelist()

    if target_username in whitelist:
        user_info = whitelist[target_username]

        if user_info.get("subscription") == "Бесконечна":
            await update.message.reply_text(f"Подписка пользователя @{target_username} уже бесконечная.")
            return

        new_subscription_end = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
        user_info.update({"subscription": new_subscription_end, "model": "gpt-4o-2024-08-06"})
        save_whitelist(whitelist)

        await update.message.reply_text(
            f"Подписка для @{target_username} активирована до {new_subscription_end}. Установлена модель: Lirt-4o-plus."
        )

        user_id = user_info["id"]
        await context.bot.send_message(
            chat_id=user_id,
            text=f"Ваша подписка была успешно активирована до {new_subscription_end}.\nУстановлена модель: Lirt-4o-plus."
        )
    else:
        await update.message.reply_text(f"Пользователь @{target_username} не найден в белом списке.")

# Ежедневная проверка подписки
async def start_daily_checks(app):
    scheduler = AsyncIOScheduler()

    scheduler.add_job(
        check_subscriptions,
        "cron",
        hour=0,
        kwargs={"context": app}
    )

    scheduler.start()

### Генерация фото
# Команда /generate - начало процесса
async def start_generate(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    today = datetime.now().date()
    username = update.effective_user.username

    if not is_subscription_active(username, whitelist):
        await update.message.reply_text(
            "Выбор модели доступен только для пользователей с активной подпиской. Используйте команду /plus, чтобы оформить подписку."
        )
        return ConversationHandler.END

    if user_id in user_last_generation and user_last_generation[user_id] == today:
        await update.message.reply_text("Вы уже использовали лимит на генерацию изображения сегодня.")
        return ConversationHandler.END

    await update.message.reply_text(
        "Введите описание изображения для генерации или используйте /cancel для отмены."
    )
    return ENTER_PROMPT

# Обработка промта
async def process_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    prompt = update.message.text.strip()

    if not prompt:
        await update.message.reply_text(
            "Запрос не может быть пустым. Попробуйте снова или используйте /cancel для отмены."
        )
        return ENTER_PROMPT

    context.user_data["prompt"] = prompt

    keyboard = [
        [
            InlineKeyboardButton("Квадратное", callback_data="square"),
            InlineKeyboardButton("Вертикальное", callback_data="vertical"),
            InlineKeyboardButton("Горизонтальное", callback_data="horizontal"),
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    message = await update.message.reply_text("Выберите формат изображения:", reply_markup=reply_markup)

    context.user_data["size_message_id"] = message.message_id

    return SELECT_SIZE

async def process_size(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()

    size_key = query.data

    if size_key not in SIZE_MAPPING:
        await query.edit_message_text("Выбор недействителен. Попробуйте снова.")
        return SELECT_SIZE

    size = SIZE_MAPPING[size_key]
    context.user_data["size"] = size

    prompt = context.user_data["prompt"]

    size_message_id = context.user_data.get("size_message_id")
    if size_message_id:
        await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=size_message_id)

    await asyncio.sleep(1)

    temp_message = await context.bot.send_message(update.effective_chat.id, "Получен запрос...")
    await asyncio.sleep(0.5)

    await temp_message.edit_text("Генерирую изображение...")
    await asyncio.sleep(1)

    try:
        if size == "1024x1024":
            model = "dall-e-2"
        else:
            model = "dall-e-3"

        response = openai.Image.create(
            prompt=prompt,
            n=1,
            size=size,
            model=model
        )

        if isinstance(response, str):
            response = json.loads(response)

        if 'data' not in response or not response['data']:
            await temp_message.edit_text("Ошибка: API не вернул данных изображения.")
            return ConversationHandler.END

        image_url = response['data'][0]['url']

        image_data = requests.get(image_url).content
        file_path = os.path.join(SAVE_DIR, f"generated_image_{update.effective_user.id}.png")
        with open(file_path, "wb") as f:
            f.write(image_data)

        await context.bot.send_photo(chat_id=update.effective_chat.id, photo=open(file_path, "rb"))

        await temp_message.edit_text(f"Изображение сохранено на сервере: {file_path}")
        await temp_message.edit_text(f"Изображение по запросу: {prompt}")

        await asyncio.sleep(0.5)
        os.remove(file_path)

        user_last_generation[update.effective_user.id] = datetime.now().date()

    except requests.exceptions.RequestException as req_err:
        await temp_message.edit_text(f"Ошибка при скачивании изображения: {req_err}")
    except openai.error.OpenAIError as api_err:
        await temp_message.edit_text(f"Ошибка API OpenAI: {api_err}")
    except Exception as e:
        await temp_message.edit_text(f"Произошла непредвиденная ошибка: {e}")

    return ConversationHandler.END

# Команда /cancel для отмены
async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Процесс отменен.")
    return ConversationHandler.END

# Обработчик ConversationHandler
generate_handler = ConversationHandler(
    entry_points=[CommandHandler('generate', start_generate)],
    states={
        ENTER_PROMPT: [MessageHandler(filters.TEXT & ~filters.COMMAND, process_prompt)],
        SELECT_SIZE: [CallbackQueryHandler(process_size)],
    },
    fallbacks=[CommandHandler('cancel', cancel)],
)

### Вайтлист
# Загрузка вайтлиста
def load_whitelist() -> Dict[str, dict]:
    if not os.path.exists(WHITELIST_FILE):
        with open(WHITELIST_FILE, "w", encoding="utf-8") as file:
            json.dump({}, file, ensure_ascii=False, indent=4)
        return {}

    try:
        with open(WHITELIST_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError:
        with open(WHITELIST_FILE, "w", encoding="utf-8") as file:
            json.dump({}, file, ensure_ascii=False, indent=4)
        return {}

# Сохранение вайтлиста
def save_whitelist(whitelist: Dict[str, dict]):
    with open(WHITELIST_FILE, "w", encoding="utf-8") as file:
        json.dump(whitelist, file, ensure_ascii=False, indent=4)

# Получение модели из whitelist
def get_user_model(whitelist, user_id):
    user_data = whitelist.get(user_id)
    return user_data.get("model") if user_data else None

whitelist = load_whitelist()

# Обновление вайтлиста для бота
async def monitor_whitelist():
    global whitelist, last_modified_time
    while True:
        try:
            current_modified_time = os.path.getmtime(WHITELIST_FILE)
            if last_modified_time is None or current_modified_time != last_modified_time:
                last_modified_time = current_modified_time
                whitelist = load_whitelist()
                print("Белый список обновлён.")
        except Exception as e:
            print(f"Ошибка при мониторинге белого списка: {e}")
        await asyncio.sleep(5)

### Разрыв сообщений, если те слишком большие
# Разрыв сообщений
def split_message(text, max_length=4096):
    if len(text) <= max_length:
        return [text]
    parts = []
    while len(text) > max_length:
        split_point = text.rfind(" ", 0, max_length)
        if split_point == -1:
            split_point = max_length
        parts.append(text[:split_point])
        text = text[split_point:].strip()
    parts.append(text)
    return parts

### Генерация аудио (Не работает)
# Генерация аудио с помощью ChatGPT
async def generate_audio(prompt: str, filename: str) -> None:
    response = openai.ChatCompletion.create(
        model="gpt-4o-audio-preview",
        modalities=["text", "audio"],
        audio={"voice": "alloy", "format": "wav"},
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    audio_data = base64.b64decode(response["choices"][0]["message"]["audio"]["data"])
    with open(filename, "wb") as f:
        f.write(audio_data)

# Обработчик команды /generate_audio
async def generate_audio_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Используйте /generate_audio [текст], чтобы сгенерировать аудио.")
        return

    prompt = " ".join(context.args)
    filename = "generated_audio.wav"

    try:
        await generate_audio(prompt, filename)

        with open(filename, "rb") as audio_file:
            await update.message.reply_audio(audio_file)

    except Exception as e:
        await update.message.reply_text(f"Произошла ошибка: {e}")

    finally:
        if os.path.exists(filename):
            os.remove(filename)

### Установка модели
# Команда /setmodel
async def start_set_model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    username = update.effective_user.username
    if not username or username.lower() not in whitelist:
        await update.message.reply_text("Извините, у вас нет доступа к этому боту.")
        return ConversationHandler.END

    if not is_subscription_active(username, whitelist):
        await update.message.reply_text(
            "Выбор модели доступен только для пользователей с активной подпиской. Используйте команду /plus, чтобы оформить подписку."
        )
        return ConversationHandler.END

    await update.message.reply_text(
        "Пожалуйста, выберите одну из доступных моделей:\n\n" +
        "\n".join([f"{model} - {desc}" for model, desc in MODEL_DESCRIPTIONS.items()]))
    return SELECT_MODEL

# Выбор модели
async def select_model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    model_name = update.message.text.strip()

    if model_name not in MODEL_DESCRIPTIONS:
        await update.message.reply_text("Вы выбрали недоступную модель. Выберете модель из списка выше, и попробуйте снова. Отменить выбор модели - /cancel")
        return SELECT_MODEL

    username = update.effective_user.username
    whitelist[username.lower()]["model"] = MODEL_MAPPING[model_name]
    save_whitelist(whitelist)

    await update.message.reply_text(f"Модель {model_name} успешно установлена!")

    return ConversationHandler.END

# Команда для завершения процесса
async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Процесс отменен.")
    return ConversationHandler.END

set_model_handler = ConversationHandler(
    entry_points=[CommandHandler('setmodel', start_set_model)],  # Заменяем старый обработчик
    states={
        SELECT_MODEL: [MessageHandler(filters.TEXT & ~filters.COMMAND, select_model)],
    },
    fallbacks=[CommandHandler('cancel', cancel)],
)

### Генерация текста
# Генерация текста с помощью ChatGPT
async def chat_with_gpt(username: str, message: str) -> str:
    try:
        # Получение истории пользователя
        history = user_memory.setdefault(username, deque(maxlen=50))
        context_messages = [{"role": "user", "content": msg} for msg in history]
        context_messages.append({"role": "user", "content": message})

        # Выбор модели
        whitelist_entry = whitelist.get(username.lower())
        if whitelist_entry:
            user_model = whitelist_entry.get("model", DEFAULT_MODEL)
        else:
            user_model = DEFAULT_MODEL

        # Формирование запроса к API
        request_payload = {
            "model": user_model,
            "messages": context_messages,
        }

        # Отправка запроса к OpenAI API
        response = await openai.ChatCompletion.acreate(**request_payload)
        return response["choices"][0]["message"]["content"].strip()

    except openai.error.OpenAIError as e:
        return f"Ошибка API OpenAI: {str(e)}"
    except Exception as e:
        return f"Ошибка: {str(e)}"

# Обновление сообщений
async def update_status_message(context, chat_id, message_id, text):
    try:
        await context.bot.edit_message_text(
            chat_id=chat_id,
            message_id=message_id,
            text=text
        )
    except Exception:
        pass

# Обработка сообщения
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    username = update.effective_user.username
    current_time = datetime.now()

    # Проверка, есть ли пользователь в списке заблокированных
    if username in spam_blocked_users:
        block_until = spam_blocked_users[username]
        if current_time < block_until:
            remaining_time = (block_until - current_time).seconds // 60
            await update.message.reply_text(
                f"Вы отправляете слишком много сообщений. Попробуйте через {remaining_time} минут."
            )
            return
        else:
            del spam_blocked_users[username]  # Снимаем блокировку

    # Добавление времени сообщения в трекер
    message_times = user_message_tracker[username]
    message_times.append(current_time)

    # Удаление старых сообщений из очереди
    while message_times and message_times[0] < current_time - timedelta(minutes=1):
        message_times.popleft()

    # Проверка на превышение лимита
    if len(message_times) > MAX_MESSAGES_PER_MINUTE:
        spam_blocked_users[username] = current_time + BLOCK_TIME
        await update.message.reply_text(
            f"Вы отправляете слишком много сообщений. Попробуйте через 30 минут."
        )
        return

    # Обработка сообщения
    try:
        # Запись пользовательского сообщения в историю
        user_message = update.message.text
        user_memory.setdefault(username, deque(maxlen=50)).append(user_message)

        # Определяем модель пользователя
        whitelist_entry = whitelist.get(username.lower())
        if whitelist_entry:
            user_model = whitelist_entry.get("model", DEFAULT_MODEL)
        else:
            user_model = DEFAULT_MODEL

        uses_thinking = user_model in ["o1-mini-2024-09-12"]

        # Временное сообщение
        temp_message = await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Получен запрос...",
            reply_to_message_id=update.message.message_id
        )
        await asyncio.sleep(0.5)

        # Обновление статуса
        await update_status_message(context, update.effective_chat.id, temp_message.message_id, "Обработка...")
        await asyncio.sleep(0.5)

        if uses_thinking:
            await update_status_message(context, update.effective_chat.id, temp_message.message_id, "Думаю...")
            await asyncio.sleep(1)
        else:
            await update_status_message(context, update.effective_chat.id, temp_message.message_id, "Генерация ответа...")
            await asyncio.sleep(1)

        # Генерация ответа от ChatGPT
        bot_response = await chat_with_gpt(username, user_message)

        # Отправка финального ответа
        await update_status_message(context, update.effective_chat.id, temp_message.message_id, "Отправка ответа...")
        await asyncio.sleep(0.5)

        await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=temp_message.message_id)
        formatted_response = format_text(bot_response)

        # Разделение длинного текста на части
        for part in split_message(formatted_response):
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=part,
                parse_mode="HTML",
                reply_to_message_id=update.message.message_id
            )

    except Exception as e:
        await update.message.reply_text(f"Произошла ошибка: {str(e)}")
    finally:
        # Сброс статуса обработки независимо от результата
        user_response_status[username] = False

async def process_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Основная логика обработки сообщений
    username = update.effective_user.username
    user_message = update.message.text
    # Здесь вызывается основная логика чата
    bot_response = await chat_with_gpt(username, user_message)
    await update.message.reply_text(bot_response)

# Форматирование текста под выделения телеграмма
def format_text(message: str) -> str:
    patterns = [
        (r"\*\*(.+?)\*\*", r"<b>\1</b>"),
        (r"_(.+?)_", r"<i>\1</i>"),
        (r"'''(.+?)'''", r"<code>\1</code>"),
        (r"```(.+?)```", r"<pre>\1</pre>")
    ]
    for pattern, replacement in patterns:
        message = re.sub(pattern, replacement, message, flags=re.DOTALL)
    return message

### Админ вайтлист
# Загрузка админ вайтлиста
def load_admin_whitelist():
    if not os.path.exists("admin_whitelist.txt"):
        return []
    with open("admin_whitelist.txt", "r", encoding="utf-8") as file:
        return [line.strip().lower() for line in file.readlines()]

### Команда /clear
# Код команды /clear
async def clear_memory(update: Update, context: ContextTypes.DEFAULT_TYPE):
    username = update.effective_user.username

    if not username or username.lower() not in whitelist:
        await update.message.reply_text("Извините, у вас нет доступа к этому боту.")
        return

    if username in user_memory:
        user_memory[username].clear()
        await update.message.reply_text("Ваша история сообщений была успешно очищена.")
    else:
        await update.message.reply_text("Ваша история уже пуста.")

clear_handler = CommandHandler("clear", clear_memory)

### Команда /send для отправки сообщения всем пользователям
# Код команды /send
async def send_message_to_all(update: Update, context: ContextTypes.DEFAULT_TYPE):
    username = update.effective_user.username
    if not username:
        await update.message.reply_text("Ваш юзернейм не определен. Команда недоступна.")
        return

    admin_whitelist = load_admin_whitelist()
    if username.lower() not in admin_whitelist:
        await update.message.reply_text("У вас нет прав для выполнения этой команды.")
        return

    if not context.args:
        await update.message.reply_text("Пожалуйста, укажите сообщение для рассылки (например, /send Привет всем!).")
        return

    message = " ".join(context.args)
    whitelist = load_whitelist()

    failed_ids = []
    for user_info in whitelist.values():
        user_id = user_info.get("id")
        if not user_id:
            continue
        try:
            await context.bot.send_message(chat_id=user_id, text=message)
        except Exception as e:
            failed_ids.append(user_info.get("username", "Неизвестный"))

    if failed_ids:
        await update.message.reply_text(
            f"Сообщение отправлено всем, кроме: {', '.join(failed_ids)}. Проверьте их ID."
        )
    else:
        await update.message.reply_text("Сообщение успешно отправлено всем пользователям.")

# Обработчик команд
def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("plus", plus_info))
    app.add_handler(set_model_handler)
    app.add_handler(generate_handler)
    app.add_handler(CommandHandler("send", send_message_to_all))
    app.add_handler(CommandHandler("active", activate_plus))
    app.add_handler(CommandHandler("clean", clear_memory))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    loop = asyncio.get_event_loop()
    loop.create_task(monitor_whitelist())
    loop.run_until_complete(check_subscriptions(app))
    loop.run_until_complete(start_daily_checks(app))

    print("Бот запущен.")
    app.run_polling()

if __name__ == "__main__":
    main()
