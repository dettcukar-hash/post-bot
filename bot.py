import os
import logging
import tempfile
from functools import wraps

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ConversationHandler,
    filters,
    ContextTypes,
)
from openai import AsyncOpenAI

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
ALLOWED_USER_ID = int(os.environ["ALLOWED_USER_ID"])

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Conversation states
WAITING_INPUT, REVIEWING, EDITING = range(3)

SYSTEM_PROMPT = """# Роль
Ты пишешь посты для Telegram-канала предпринимателя Михаила Захарова — собственника вентиляционного завода РОТАДО.
Твоя задача не просто написать красивый текст.
Твоя задача — передать мышление предпринимателя-практика, который строит работающие системы, ищет первопричины проблем и делится реальным управленческим опытом.

# Кто такой Михаил Захаров
Предприниматель с 18-летним опытом.
Создал и развивает производственный бизнес.
Управляет компанией численностью более 100 сотрудников.
Мыслит через системы, процессы и причинно-следственные связи.
Основная особенность мышления — стремление находить корневую причину проблемы и устранять ее через изменение системы, а не через героизм отдельных людей.
Не любит неэффективность.
Не любит лишние действия.
Не любит сложность ради сложности.
Любит здравый смысл.
Любит практичность.
Любит решения, которые работают в реальной жизни.

# Главная идея личного бренда
Не эксперт по вентиляции.
Не владелец завода.
Не бизнес-тренер.
Основной образ:
Предприниматель-системщик, который превращает хаос в работающие системы.
Через реальные истории показывает:
- как устроено управление;
- как принимаются решения;
- как строятся компании;
- как внедряются изменения;
- как работают люди;
- почему одни подходы работают, а другие нет.

# Основные ценности
- эффективность;
- практичность;
- развитие;
- ответственность;
- осознанность;
- созидание;
- инновации;
- экологичность;
- здравый смысл.

# Целевая аудитория
Владельцы бизнеса.
Предприниматели.
Руководители.
Топ-менеджеры.
Люди, которые строят системы и несут ответственность за результат.

# Как писать
Всегда писать через историю, наблюдение или реальную ситуацию.
Никогда не начинать пост с теории.
Плохой вариант:
"Существует важный принцип управления..."
Хороший вариант:
"Недавно сотрудник пришел ко мне с вопросом..."
Или:
"Когда мы меняли систему оплаты труда на производстве..."
Или:
"На одном из совещаний я заметил..."

# Структура поста
1. Реальная ситуация.
2. Конфликт или проблема.
3. Размышление.
4. Вывод более высокого уровня.
5. Практический принцип.
Формула:
Ситуация → проблема → причина → вывод → принцип.

# Стиль текста
Писать простым разговорным языком.
Без канцелярита.
Без сложных терминов.
Без пафоса.
Без инфоцыганских формулировок.
Без манипуляций.
Без искусственной мотивации.
Текст должен выглядеть так, будто предприниматель рассказывает историю за чашкой кофе другому предпринимателю.

# Что нельзя делать
Не писать:
- "успешный успех";
- "масштабирование через синергию";
- "прорывные инструменты";
- "секретные технологии";
- "мышление миллионера";
- "изобилие";
- "денежное состояние";
- "высокие вибрации".
Не использовать клише бизнес-тренеров.
Не изображать всезнайку.
Не читать мораль.

# Что обязательно делать
Использовать:
- реальные примеры;
- диалоги;
- наблюдения;
- цифры;
- производственные истории;
- управленческие ситуации.
Чем больше конкретики, тем лучше.

# Эмоции
Главная эмоция постов — не вдохновение.
Главная эмоция — узнавание.
Читатель должен думать:
"Точно. У меня так же."
После этого появляется доверие.

# Тональность
Спокойная.
Уверенная.
Взрослая.
Без истерик.
Без агрессии.
Без самолюбования.
Автор не доказывает свою значимость.
Он делится наблюдениями и выводами.

# Частая ошибка
Не переходить к выводу слишком быстро.
Сначала нужно дать читателю прожить ситуацию.
Показать конфликт.
Показать ход мыслей.
Только потом вывод.

# Финальный критерий качества
После прочтения поста предприниматель должен сказать:
"Интересно. Никогда не смотрел на это под таким углом."
или
"Точно. У меня такая же проблема."
Если текст вызывает такую реакцию — пост написан правильно.

# Форматирование
Абзацы — 2–4 предложения, пустая строка между каждым абзацем.
Перечисления — каждый пункт на отдельной строке через тире: «- пункт».
Цитаты и реплики читателей — каждая на отдельной строке, в кавычках «».
Категории — на отдельной строке с двоеточием: «Первый:», «Второй:».
Обращение к читателю — на «Вы» с заглавной буквы.
Никаких хэштегов.

# Финальная инструкция
Возвращай ТОЛЬКО текст поста, без заголовков, подписей и комментариев."""


def load_style_examples() -> str:
    """Загружает примеры постов из файла posts.txt рядом с ботом."""
    posts_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "posts.txt")
    try:
        with open(posts_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        logger.info("Loaded style examples from posts.txt (%d chars)", len(content))
        return content
    except FileNotFoundError:
        logger.warning("posts.txt not found — generating without style examples")
        return ""


STYLE_EXAMPLES = load_style_examples()


def auth_required(func):
    """Декоратор: проверяет, что сообщение от разрешённого пользователя."""
    @wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if update.effective_user and update.effective_user.id != ALLOWED_USER_ID:
            if update.effective_message:
                await update.effective_message.reply_text("⛔ Доступ запрещён.")
            elif update.callback_query:
                await update.callback_query.answer("⛔ Доступ запрещён.", show_alert=True)
            return ConversationHandler.END
        return await func(update, context)
    return wrapper


async def transcribe_voice(file_id: str, context: ContextTypes.DEFAULT_TYPE) -> str:
    """Скачивает голосовое сообщение и транскрибирует через Whisper."""
    voice_file = await context.bot.get_file(file_id)

    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        await voice_file.download_to_drive(tmp.name)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as audio:
            transcript = await client.audio.transcriptions.create(
                model="whisper-1",
                file=audio,
                language="ru",
            )
        return transcript.text
    finally:
        os.unlink(tmp_path)


def _build_examples_block() -> str:
    """Возвращает блок с примерами постов для вставки в промпт."""
    if not STYLE_EXAMPLES:
        return ""
    return (
        f"\n\n---\n"
        f"ПРИМЕРЫ ДЛЯ СТИЛЯ (используй только манеру письма, НЕ темы из этих постов):\n\n"
        f"{STYLE_EXAMPLES}"
    )


async def generate_post(transcript: str) -> str:
    """Генерирует первый черновик поста на основе мысли автора."""
    examples_block = _build_examples_block()
    user_content = (
        f"ТЕМА ПОСТА (обязательно писать именно на эту тему, не отклоняться):\n{transcript}\n\n"
        f"Напиши пост строго на указанную тему выше.{examples_block}"
    )

    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.7,
        max_tokens=1500,
    )
    return response.choices[0].message.content.strip()


async def revise_post(current_post: str, edit_instruction: str) -> str:
    """Редактирует существующий черновик согласно правкам автора."""
    user_content = (
        f"Вот черновик поста:\n\n{current_post}\n\n"
        f"Правка: {edit_instruction}\n\n"
        f"Внеси правку в черновик. Сохрани тему, структуру и всё, что уже хорошо. "
        f"Верни только исправленный текст поста, без комментариев."
    )

    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.7,
        max_tokens=1500,
    )
    return response.choices[0].message.content.strip()


def review_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("✅ Готово", callback_data="done"),
        InlineKeyboardButton("✏️ Изменить", callback_data="edit"),
    ]])


async def send_draft(message, post: str, is_edit: bool = False) -> None:
    """Отправляет черновик поста с кнопками."""
    label = "🔄 Новая версия:" if is_edit else "📝 Черновик поста:"
    await message.reply_text(
        f"{label}\n\n{post}",
        reply_markup=review_keyboard(),
    )


# ──────────────────────────────────────────────
# Handlers
# ──────────────────────────────────────────────

@auth_required
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data.clear()
    await update.message.reply_text(
        "👋 Привет! Отправь голосовое сообщение с основной мыслью поста — "
        "я транскрибирую и напишу готовый текст.\n\n"
        "Можно также отправить текст, если не хочется записывать голос."
    )
    return WAITING_INPUT


@auth_required
async def handle_initial_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    status_msg = await update.message.reply_text("⏳ Транскрибирую голосовое...")

    try:
        transcript = await transcribe_voice(update.message.voice.file_id, context)
    except Exception as e:
        logger.error("Transcription error: %s", e)
        await status_msg.edit_text(f"❌ Ошибка транскрипции: {e}\nПопробуй ещё раз.")
        return WAITING_INPUT

    context.user_data["transcript"] = transcript
    await status_msg.edit_text(f"🎙 Распознано: {transcript}\n\n✍️ Генерирую пост...")

    try:
        post = await generate_post(transcript)
    except Exception as e:
        logger.error("Generation error: %s", e)
        await status_msg.edit_text(f"❌ Ошибка генерации: {e}\nПопробуй ещё раз.")
        return WAITING_INPUT

    context.user_data["current_post"] = post
    await status_msg.delete()
    await send_draft(update.message, post)
    return REVIEWING


@auth_required
async def handle_initial_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    transcript = update.message.text
    context.user_data["transcript"] = transcript

    status_msg = await update.message.reply_text("✍️ Генерирую пост...")

    try:
        post = await generate_post(transcript)
    except Exception as e:
        logger.error("Generation error: %s", e)
        await status_msg.edit_text(f"❌ Ошибка генерации: {e}\nПопробуй ещё раз.")
        return WAITING_INPUT

    context.user_data["current_post"] = post
    await status_msg.delete()
    await send_draft(update.message, post)
    return REVIEWING


@auth_required
async def handle_review_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()

    if query.data == "done":
        post = context.user_data.get("current_post", "")
        await query.edit_message_reply_markup(None)
        await query.message.reply_text(f"✅ Финальный пост:\n\n{post}")
        context.user_data.clear()
        return ConversationHandler.END

    elif query.data == "edit":
        await query.edit_message_reply_markup(None)
        await query.message.reply_text(
            "✏️ Что нужно изменить? Отправь голосовое или текст с правками."
        )
        return EDITING

    return REVIEWING


@auth_required
async def handle_edit_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    status_msg = await update.message.reply_text("⏳ Транскрибирую правки...")

    try:
        edit_text = await transcribe_voice(update.message.voice.file_id, context)
    except Exception as e:
        logger.error("Transcription error: %s", e)
        await status_msg.edit_text(f"❌ Ошибка транскрипции: {e}")
        return EDITING

    await status_msg.edit_text(f"🎙 Правки: {edit_text}\n\n✍️ Применяю правки...")
    return await _apply_edit(update, context, status_msg, edit_text)


@auth_required
async def handle_edit_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    status_msg = await update.message.reply_text("✍️ Применяю правки...")
    return await _apply_edit(update, context, status_msg, update.message.text)


async def _apply_edit(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    status_msg,
    edit_text: str,
) -> int:
    current_post = context.user_data.get("current_post", "")

    try:
        post = await revise_post(current_post, edit_text)
    except Exception as e:
        logger.error("Generation error: %s", e)
        await status_msg.edit_text(f"❌ Ошибка генерации: {e}")
        return EDITING

    context.user_data["current_post"] = post
    await status_msg.delete()
    await send_draft(update.message, post, is_edit=True)
    return REVIEWING


@auth_required
async def cmd_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data.clear()
    await update.message.reply_text(
        "❌ Сброшено. Отправь /start или просто пришли новое голосовое."
    )
    return ConversationHandler.END


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main() -> None:
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    conv_handler = ConversationHandler(
        entry_points=[
            CommandHandler("start", cmd_start),
            MessageHandler(filters.VOICE, handle_initial_voice),
            MessageHandler(filters.TEXT & ~filters.COMMAND, handle_initial_text),
        ],
        states={
            WAITING_INPUT: [
                MessageHandler(filters.VOICE, handle_initial_voice),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_initial_text),
            ],
            REVIEWING: [
                CallbackQueryHandler(handle_review_callback, pattern="^(done|edit)$"),
            ],
            EDITING: [
                MessageHandler(filters.VOICE, handle_edit_voice),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_edit_text),
            ],
        },
        fallbacks=[CommandHandler("cancel", cmd_cancel)],
        per_user=True,
        allow_reentry=False,
    )

    app.add_handler(conv_handler)

    logger.info("Bot started. Polling...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
