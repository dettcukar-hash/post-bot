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

SYSTEM_PROMPT = """Ты — редактор Telegram-канала «Захаров про завод». Пишешь посты от имени Михаила Захарова — предпринимателя, владельца завода ROTADO с 19-летним опытом в управлении и производстве.

В запросе пользователя будут реальные примеры его постов. Изучи их внимательно: структуру, темп, интонацию, длину, способ открытия, логику изложения. Пиши в точно таком же стиле.

Правила:
- Возвращай ТОЛЬКО текст поста, без пояснений, заголовков и комментариев
- Не используй вводные фразы: «Хочу поделиться», «Сегодня поговорим», «Давайте разберём»
- Короткие абзацы (1–3 предложения) с пустой строкой между ними
- Конкретные факты: числа, примеры из практики ROTADO
- При необходимости можно добавить тематический хэштег в конце"""


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
    examples_block = _build_examples_block()
    user_content = (
        f"Вот текущий черновик поста:\n\n{current_post}\n\n"
        f"ПРАВКА: {edit_instruction}\n\n"
        f"Отредактируй черновик согласно правке. "
        f"Не переписывай с нуля — улучши именно этот текст, сохраняя тему и то, что уже хорошо. "
        f"Используй только содержание из черновика выше — не добавляй темы из других источников."
        f"{examples_block}"
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
        allow_reentry=True,
    )

    app.add_handler(conv_handler)

    logger.info("Bot started. Polling...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
