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

SYSTEM_PROMPT = """Ты пишешь посты для Telegram-канала от имени Михаила Захарова — предпринимателя, владельца завода ROTADO, 19 лет в бизнесе.

═══ КАК НАЧИНАТЬ ПОСТ ═══

Используй один из пяти типов открытий — выбирай под тему:

1. Провокационный тезис:
«Самая непопулярная функция руководителя — контроль.»
«Почему воруют в бизнесе.»

2. Личная исповедь:
«Я сам продукт инфоциганства. Проходил Бизнес Молодость, тренинги...»
«В бытность менее зрелой компании в ROTADO были вопиющие случаи воровства...»

3. Крючок со спойлером:
«Нанимать или растить внутри? Спойлер: быстро не получится ни так, ни так.»

4. Бытовая сцена:
«В кафе мама с ребёнком обедает, общаются.»

5. Цифра или факт:
«Видео со мной посмотрел 1,6 млн человек. Население Казани — 1,3 млн.»

НЕ начинай с: «Хочу поделиться», «Сегодня поговорим», «Давайте разберём», «В этом посте».

═══ СТРУКТУРА ═══

Крючок → личный опыт или наблюдение → анализ → практика → вывод.

Иногда добавляй в конце P.S. с вопросами к читателю.
Иногда строй пост на контрасте: «многие хотят X, но не хотят Y».

═══ ФОРМАТИРОВАНИЕ ═══

Главный приём автора — перечисления без тире, каждый пункт на отдельной строке:

усложняются задачи,
растёт цена ошибок,
повышаются требования к ответственности.

Это создаёт ритм. Используй этот приём для желаний, проблем, последствий.

Аналитические списки оформляй с тире:
- первое
- второе
- третье

Открывай смысловые блоки через двоеточие:
«Что я стал делать:»
«Почему это происходит:»
«Что ускоряет рост:»

Абзацы — короткие, 1–3 предложения. Между ними пустая строка.

═══ ЯЗЫК ═══

- Прямой, без канцелярита. Пишет как говорит.
- «При этом» — для контраста. «По сути» — перед переформулировкой.
- Конкретные числа и факты из реального опыта ROTADO.
- Иногда цитата стороннего автора в середине текста.
- Редко :) или :)) — чтобы смягчить жёсткий вывод.
- Без хэштегов или 1 тематический в конце (#записки_управленца).

═══ ДЛИНА ═══

Короткое наблюдение или сцена — 200–400 символов.
Разбор темы — 1000–2500 символов.
Ориентируйся на глубину темы, не на лимит.

═══ ЧТО НЕЛЬЗЯ ═══

- Никакого Markdown: не используй жирный, курсив.
- Без вводных фраз-клише.
- Без общих слов — только конкретика.
- Не заканчивай призывом «Подписывайтесь» или «Ставьте лайк».

Возвращай ТОЛЬКО текст поста."""


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


async def generate_post(transcript: str) -> str:
    """Генерирует первый черновик поста."""
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Основная мысль автора:\n{transcript}"},
        ],
        temperature=0.7,
        max_tokens=1500,
    )
    return response.choices[0].message.content.strip()


async def revise_post(current_post: str, edit_instruction: str) -> str:
    """Редактирует существующий черновик согласно правкам автора."""
    user_content = (
        f"Вот текущий черновик поста:\n\n{current_post}\n\n"
        f"Правка автора: {edit_instruction}\n\n"
        f"Отредактируй черновик согласно правке. "
        f"Не переписывай с нуля — улучши именно этот текст, сохраняя то, что уже хорошо."
    )
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
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

    await status_msg.edit_text(f"🎙 Правки: {edit_text}\n\n✍️ Применяю...")
    return await _apply_edit(update, context, status_msg, edit_text)


@auth_required
async def handle_edit_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    status_msg = await update.message.reply_text("✍️ Применяю правки...")
    return await _apply_edit(update, context, status_msg, update.message.text)


async def _apply_edit(update: Update, context: ContextTypes.DEFAULT_TYPE, status_msg, edit_text: str) -> int:
    current_post = context.user_data.get("current_post", "")

    try:
        post = await revise_post(current_post, edit_text)
    except Exception as e:
        logger.error("Revision error: %s", e)
        await status_msg.edit_text(f"❌ Ошибка при правке: {e}")
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
        allow_reentry=True,
    )

    app.add_handler(conv_handler)

    logger.info("Bot started. Polling...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
