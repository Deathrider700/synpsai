import logging
import os
import httpx
import uuid
import re
import asyncio
from collections import deque
from datetime import date

import motor.motor_asyncio
from pymongo import ReturnDocument

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, error
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler, MessageHandler,
    filters, ContextTypes, ConversationHandler
)
from telegram.constants import ChatAction, ParseMode

# --- CONFIGURATION ---
TELEGRAM_BOT_TOKEN = "8385126802:AAEqYo6r3IyteSnPgLHUTpAaxdNU1SfHlB4"
A4F_API_KEY = "ddc-a4f-4c0658a7764c432c9aa8e4a6d409afb3"
A4F_API_BASE_URL = "https://api.a4f.co/v1"

# !! IMPORTANT !!
# PASTE THE "STANDARD CONNECTION STRING" FROM MONGODB ATLAS HERE.
# It should start with "mongodb://" and NOT "mongodb+srv://"
# Make sure to replace <password> with your actual database password.
MONGO_DB_URI = "mongodb+srv://kuntaldebnath777:CRbyIO8WhWbTUTGO@cluster0.phnj4cn.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

ADMIN_CHAT_ID = 7088711806
DAILY_CREDITS = 10

# --- LOGGING SETUP ---
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# --- DATABASE SETUP ---
try:
    db_client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_DB_URI)
    db = db_client["telegram_bot_db"]
    users_collection = db["users"]
    redeem_codes_collection = db["redeem_codes"]
except Exception as e:
    logger.critical(f"Could not connect to MongoDB: {e}")
    exit(1)


# --- MODELS & CONSTANTS ---
MODELS = {
    "chat": [
        "provider-1/chatgpt-4o-latest", "provider-3/gpt-4", "provider-3/gpt-4.1-mini", "provider-6/gpt-4.1-mini",
        "provider-6/gpt-4.1-nano", "provider-3/gpt-4.1-nano", "provider-6/gpt-4o-mini-search-preview",
        "provider-3/gpt-4o-mini-search-preview", "provider-6/gpt-4o", "provider-6/o3-medium", "provider-6/o3-high",
        "provider-6/o3-low", "provider-6/gpt-4.1", "provider-6/o4-mini-medium", "provider-6/o4-mini-high",
        "provider-6/o4-mini-low", "provider-1/gemini-2.5-pro", "provider-3/deepseek-v3", "provider-1/deepseek-v3-0324",
        "provider-1/sonar", "provider-1/sonar-deep-research", "provider-2/mistral-small", "provider-6/minimax-m1-40k",
        "provider-6/kimi-k2", "provider-3/kimi-k2", "provider-6/qwen3-coder-480b-a35b", "provider-3/llama-3.1-405b",
        "provider-3/qwen-3-235b-a22b-2507", "provider-1/mistral-large", "provider-2/llama-4-scout",
        "provider-2/llama-4-maverick", "provider-6/gemini-2.5-flash-thinking", "provider-6/gemini-2.5-flash",
        "provider-1/gemma-3-12b-it", "provider-1/llama-3.3-70b-instruct-turbo", "provider-2/codestral",
        "provider-1/llama-3.1-405b-instruct-turbo", "provider-3/llama-3.1-70b", "provider-2/qwq-32b",
        "provider-3/qwen-2.5-coder-32b", "provider-6/kimi-k2-instruct", "provider-2/mistral-saba",
        "provider-6/r1-1776", "provider-6/deepseek-r1-uncensored", "provider-1/deepseek-r1-0528",
        "provider-1/sonar-reasoning-pro", "provider-1/sonar-reasoning", "provider-1/sonar-pro",
        "provider-3/mistral-small-latest", "provider-3/magistral-medium-latest"
    ],
    "image": [
        "provider-4/imagen-3", "provider-4/imagen-4", "provider-6/sana-1.5-flash", "provider-1/FLUX.1-schnell",
        "provider-2/FLUX.1-schnell", "provider-3/FLUX.1-schnell", "provider-6/sana-1.5", "provider-3/FLUX.1-dev",
        "provider-6/FLUX.1-dev", "provider-1/FLUX.1.1-pro", "provider-6/FLUX.1-pro", "provider-1/FLUX.1-kontext-pro",
        "provider-6/FLUX.1-kontext-pro", "provider-6/FLUX.1-1-pro", "provider-6/FLUX.1-kontext-dev",
        "provider-2/FLUX.1-schnell-v2", "provider-6/FLUX.1-kontext-max"
    ],
    "image_edit": [
        "provider-6/black-forest-labs-flux-1-kontext-dev", "provider-6/black-forest-labs-flux-1-kontext-pro",
        "provider-6/black-forest-labs-flux-1-kontext-max", "provider-3/flux-kontext-dev"
    ],
    "video": ["provider-6/wan-2.1"],
    "tts": ["provider-3/tts-1", "provider-6/sonic-2", "provider-6/sonic"],
    "transcription": [
        "provider-2/whisper-1", "provider-3/whisper-1", "provider-6/distil-whisper-large-v3-en",
        "provider-3/gpt-4o-mini-transcribe"
    ]
}
MODELS_PER_PAGE = 5
TTS_VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
IMAGE_SIZES = {"Square ⏹️": "1024x1024", "Wide  widescreen": "1792x1024", "Tall 📲": "1024x1792"}
VIDEO_RATIOS = {"Wide 🎬": "16:9", "Vertical 📱": "9:16", "Square 🖼️": "1:1"}
LOADING_MESSAGES = {
    "chat": "🤔 Cogitating on a thoughtful response...",
    "image": "🎨 Painting your masterpiece...",
    "image_edit": "🖌️ Applying artistic edits...",
    "video": "🎬 Directing your short film...",
    "tts": "🎙️ Warming up the vocal cords...",
    "transcription": "👂 Listening closely to your audio..."
}

(SELECTING_ACTION, SELECTING_MODEL, AWAITING_PROMPT, AWAITING_TTS_INPUT, AWAITING_AUDIO,
 AWAITING_IMAGE_FOR_EDIT, AWAITING_EDIT_PROMPT, AWAITING_IMAGE_SIZE, AWAITING_TTS_VOICE,
 AWAITING_VIDEO_RATIO) = range(10)


async def setup_database():
    await users_collection.create_index("user_id", unique=True)
    await redeem_codes_collection.create_index("code", unique=True)
    logger.info("Database setup complete and indexes ensured.")

async def get_or_create_user(user_id: int) -> tuple[dict, bool]:
    today = date.today().isoformat()
    user_doc = await users_collection.find_one({"user_id": user_id})

    if user_doc:
        if user_doc.get("last_login_date") != today and user_id != ADMIN_CHAT_ID:
            updated_doc = await users_collection.find_one_and_update(
                {"user_id": user_id},
                {"$set": {"credits": DAILY_CREDITS, "last_login_date": today}},
                return_document=ReturnDocument.AFTER
            )
            return updated_doc, False
        return user_doc, False
    else:
        new_user_data = {
            "user_id": user_id,
            "credits": DAILY_CREDITS,
            "last_login_date": today,
        }
        await users_collection.insert_one(new_user_data)
        return new_user_data, True

async def check_and_use_credit(user_id: int) -> bool:
    if user_id == ADMIN_CHAT_ID:
        return True
    result = await users_collection.find_one_and_update(
        {"user_id": user_id, "credits": {"$gt": 0}},
        {"$inc": {"credits": -1}}
    )
    return result is not None

async def refund_credit(user_id: int):
    if user_id != ADMIN_CHAT_ID:
        await users_collection.update_one({"user_id": user_id}, {"$inc": {"credits": 1}})
        logger.info(f"Refunded 1 credit to user {user_id}")

def escape_markdown_v2(text: str) -> str:
    escape_chars = r'_*[]()~`>#+-=|{}.!'
    return re.sub(f'([{re.escape(escape_chars)}])', r'\\\1', text)

async def cleanup_files(*files):
    for file_path in files:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError as e:
                logger.error(f"Error removing file {file_path}: {e}")

def create_paginated_keyboard(model_list, category, page=0):
    buttons = []
    start_index = page * MODELS_PER_PAGE
    end_index = start_index + MODELS_PER_PAGE
    for i, model in enumerate(model_list[start_index:end_index], start=start_index):
        buttons.append([InlineKeyboardButton(f"⚙️ {model.split('/')[-1]}", callback_data=f"ms_{category}_{i}")])
    nav_buttons = []
    if page > 0:
        nav_buttons.append(InlineKeyboardButton("⬅️ Back", callback_data=f"mp_{category}_{page-1}"))
    if end_index < len(model_list):
        nav_buttons.append(InlineKeyboardButton("Next ➡️", callback_data=f"mp_{category}_{page+1}"))
    if nav_buttons:
        buttons.append(nav_buttons)
    buttons.append([InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")])
    return InlineKeyboardMarkup(buttons)

async def handle_api_error(update_or_query, error_obj):
    logger.error(f"API Error: {error_obj}")
    error_message = "❌ An unexpected error occurred."
    try:
        response = getattr(error_obj, 'response', None)
        if response:
            details = response.json()
            error_message = f"❌ *API Error:*\n{escape_markdown_v2(details.get('error', {}).get('message', 'No details from API.'))}"
        else:
             error_message = f"❌ *An unexpected error occurred:*\n{escape_markdown_v2(str(error_obj))}"
    except Exception:
        error_message = f"❌ *An unexpected API error occurred:*\n{escape_markdown_v2(str(error_obj))}"

    message_to_edit = update_or_query.message if hasattr(update_or_query, 'message') else update_or_query
    try:
        await message_to_edit.edit_text(error_message, parse_mode=ParseMode.MARKDOWN_V2)
    except Exception as e:
        logger.error(f"Failed to edit message with error: {e}")
        if hasattr(message_to_edit, 'reply_text'):
            await message_to_edit.reply_text("An API error occurred and I couldn't update the status message.")

def is_admin(user_id: int) -> bool:
    return user_id == ADMIN_CHAT_ID

async def notify_admin_of_new_user(context: ContextTypes.DEFAULT_TYPE, user: Update.effective_user):
    if not ADMIN_CHAT_ID:
        return
    base_text = (
        f"🎉 <b>New User Alert</b> 🎉\n\n"
        f"A new user has started the bot!\n\n"
        f"👤 <b>Name:</b> {user.full_name}\n"
        f"🔗 <b>Mention:</b> {user.mention_html()}\n"
        f"🆔 <b>ID:</b> <code>{user.id}</code>"
    )
    username_text = f"\n✍️ <b>Username:</b> @{user.username}" if user.username else ""
    text = base_text + username_text
    try:
        await context.bot.send_message(chat_id=ADMIN_CHAT_ID, text=text, parse_mode=ParseMode.HTML)
    except Exception as e:
        logger.error(f"Failed to send new user notification to admin {ADMIN_CHAT_ID}: {e}")

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user = update.effective_user
    _, was_created = await get_or_create_user(user.id)
    if was_created:
        await notify_admin_of_new_user(context, user)

    context.user_data.pop('chat_history', None)
    context.user_data.pop('image_edit_path', None)
    context.user_data.pop('temp_file_path', None)

    keyboard = [
        [InlineKeyboardButton("💬 AI Chat", callback_data="act_chat"), InlineKeyboardButton("🎨 Image Generation", callback_data="act_image")],
        [InlineKeyboardButton("🖼️ Image Editing", callback_data="act_image_edit"), InlineKeyboardButton("🎬 Video Generation", callback_data="act_video")],
        [InlineKeyboardButton("🎙️ Text-to-Speech", callback_data="act_tts"), InlineKeyboardButton("✍️ Audio Transcription", callback_data="act_transcription")],
        [InlineKeyboardButton("💰 My Credits", callback_data="act_credits"), InlineKeyboardButton("❓ Help & Info", callback_data="act_help")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    welcome_text = (f"👋 <b>Welcome, {user.mention_html()}!</b>\n\n"
                    "I'm your all-in-one AI assistant, ready to bring your ideas to life. "
                    "👇 <b>Select a tool below to get started!</b>")
    if update.callback_query:
        await update.callback_query.answer()
        await update.callback_query.edit_message_text(text=welcome_text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)
    else:
        await update.message.reply_html(text=welcome_text, reply_markup=reply_markup)
    return SELECTING_ACTION

async def credits_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    user_data, _ = await get_or_create_user(user_id)
    credits = "♾️ Unlimited (Admin)" if is_admin(user_id) else user_data.get('credits', 0)
    text = (f"💰 <b>Your Credits</b>\n\n"
            f"You currently have: <b>{credits}</b> credits.\n\n"
            f"Normal users receive <b>{DAILY_CREDITS}</b> credits daily. "
            "You can get more by using a redeem code with the /redeem command.")

    reply_markup = InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Back to Main Menu", callback_data="main_menu")]])
    if update.callback_query:
        await update.callback_query.answer()
        await update.callback_query.edit_message_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)
    else:
        await update.message.reply_html(text, reply_markup=reply_markup)
    return SELECTING_ACTION

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    help_text = ("<b>❔ Help & Information</b>\n\n"
                 "<b>Available Commands:</b>\n"
                 "▫️ <code>/start</code> - Return to the main menu.\n"
                 "▫️ <code>/newchat</code> - Clear history for a fresh AI Chat.\n"
                 "▫️ <code>/mycredits</code> - Check your current credit balance.\n"
                 "▫️ <code>/redeem &lt;CODE&gt;</code> - Redeem a code for credits.\n"
                 "▫️ <code>/help</code> - Show this help message.")

    reply_markup = InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Back to Main Menu", callback_data="main_menu")]])
    if update.callback_query:
        await update.callback_query.answer()
        await update.callback_query.edit_message_text(help_text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)
    else:
        await update.message.reply_html(help_text, reply_markup=reply_markup)
    return SELECTING_ACTION

async def new_chat_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data.pop('chat_history', None)
    await update.message.reply_text("✅ Chat history cleared.")

async def cancel_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await cleanup_files(context.user_data.pop('image_edit_path', None), context.user_data.pop('temp_file_path', None))
    await update.message.reply_text("Action cancelled. Returning to the main menu.")
    await start_command(update, context)
    return ConversationHandler.END

async def action_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    _prefix, category = query.data.split('_', 1)
    if category == 'help': return await help_command(update, context)
    if category == 'credits': return await credits_handler(update, context)
    context.user_data['category'] = category
    last_model = context.user_data.get(f'last_model_{category}')
    if last_model:
        short_model_name = last_model.split('/')[-1]
        keyboard = [
            [InlineKeyboardButton(f"🚀 Use Last: {short_model_name}", callback_data=f"mr_{category}")],
            [InlineKeyboardButton("📋 Choose Another Model", callback_data=f"mc_{category}")]
        ]
        await query.edit_message_text(f"You previously used `{escape_markdown_v2(short_model_name)}`\\. Use it again?", reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.MARKDOWN_V2)
        return SELECTING_MODEL
    return await show_model_selection(query, context)

async def show_model_selection(update_or_query, context: ContextTypes.DEFAULT_TYPE, page=0) -> int:
    category = context.user_data['category']
    model_list = MODELS.get(category, [])
    reply_markup = create_paginated_keyboard(model_list, category, page)
    text = f"💎 *Select a Model for {category.replace('_', ' ').title()}*"
    message_to_edit = update_or_query.message if hasattr(update_or_query, 'message') else update_or_query
    await message_to_edit.edit_text(text=text, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN)
    return SELECTING_MODEL

async def model_page_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query; await query.answer()
    _prefix, data = query.data.split('_', 1)
    category, page_str = data.rsplit('_', 1)
    context.user_data['category'] = category
    return await show_model_selection(query, context, page=int(page_str))

async def model_choice_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query; await query.answer()
    _prefix, category = query.data.split('_', 1)
    context.user_data['category'] = category
    return await show_model_selection(query, context)

async def model_selection_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query; await query.answer()
    prefix, data = query.data.split('_', 1)
    model_name, category = None, None
    if prefix == 'mr':
        category = data
        model_name = context.user_data.get(f'last_model_{category}')
    elif prefix == 'ms':
        try:
            category, model_index_str = data.rsplit('_', 1)
            model_name = MODELS[category][int(model_index_str)]
        except (ValueError, IndexError) as e:
            logger.error(f"Error parsing model selection callback '{query.data}': {e}")
            await query.edit_message_text("Sorry, there was an error. Please try again.")
            return SELECTING_ACTION

    if not category or not model_name:
        await query.edit_message_text("Sorry, an error occurred. Returning to the main menu.")
        return SELECTING_ACTION

    context.user_data.update({'model': model_name, f'last_model_{category}': model_name, 'category': category})
    msg_text = f"✅ Model Selected: `{escape_markdown_v2(model_name.split('/')[-1])}`\n\n"

    if category == "image":
        keyboard = [[InlineKeyboardButton(name, callback_data=f"is_{size}") for name, size in IMAGE_SIZES.items()]]
        await query.edit_message_text(msg_text + "📏 Now, choose an aspect ratio\\.", reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.MARKDOWN_V2)
        return AWAITING_IMAGE_SIZE
    if category == "video":
        keyboard = [[InlineKeyboardButton(name, callback_data=f"vr_{ratio}") for name, ratio in VIDEO_RATIOS.items()]]
        await query.edit_message_text(msg_text + "🎬 Now, choose a video ratio\\.", reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.MARKDOWN_V2)
        return AWAITING_VIDEO_RATIO
    if category == "tts":
        keyboard = [[InlineKeyboardButton(v.capitalize(), callback_data=f"tv_{v}") for v in TTS_VOICES[:3]], [InlineKeyboardButton(v.capitalize(), callback_data=f"tv_{v}") for v in TTS_VOICES[3:]]]
        await query.edit_message_text(msg_text + "🗣️ Now, choose a voice\\.", reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.MARKDOWN_V2)
        return AWAITING_TTS_VOICE

    prompt_map = {"chat": "💬 What's on your mind?","transcription": "🎤 Send me a voice message or audio file.","image_edit": "🖼️ First, send the image you want to edit."}
    next_state_map = {"chat": AWAITING_PROMPT, "transcription": AWAITING_AUDIO, "image_edit": AWAITING_IMAGE_FOR_EDIT}
    await query.edit_message_text(msg_text + escape_markdown_v2(prompt_map[category]), parse_mode=ParseMode.MARKDOWN_V2)
    return next_state_map[category]

async def image_size_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query; await query.answer()
    context.user_data['image_size'] = query.data.split('_', 1)[1]
    await query.edit_message_text("✅ Size selected.\n\n✍️ Now, what should I create?")
    return AWAITING_PROMPT

async def video_ratio_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query; await query.answer()
    context.user_data['video_ratio'] = query.data.split('_', 1)[1]
    await query.edit_message_text("✅ Ratio selected.\n\n✍️ Now, what's the scene? Describe the video.")
    return AWAITING_PROMPT

async def tts_voice_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query; await query.answer()
    context.user_data['tts_voice'] = query.data.split('_', 1)[1]
    await query.edit_message_text("✅ Voice selected.\n\n✍️ Now, send me the text you want me to say.")
    return AWAITING_TTS_INPUT

async def process_task(update: Update, context: ContextTypes.DEFAULT_TYPE, task_type: str):
    user_id = update.effective_user.id
    if not await check_and_use_credit(user_id):
        await update.effective_message.reply_text("🚫 You are out of credits! Use /redeem to get more or wait for your daily refill.")
        return SELECTING_ACTION

    message = update.effective_message
    processing_message = await message.reply_text(LOADING_MESSAGES.get(task_type, "⏳ Working..."))
    
    try:
        async with httpx.AsyncClient() as client:
            headers = {"Authorization": f"Bearer {A4F_API_KEY}"}
            if task_type == 'chat':
                await context.bot.send_chat_action(message.chat_id, ChatAction.TYPING)
                if 'chat_history' not in context.user_data: context.user_data['chat_history'] = deque(maxlen=10)
                context.user_data['chat_history'].append({"role": "user", "content": message.text})
                data = {"model": context.user_data['model'], "messages": list(context.user_data['chat_history'])}
                response = await client.post(f"{A4F_API_BASE_URL}/chat/completions", headers=headers, json=data, timeout=120)
                response.raise_for_status()
                json_data = response.json()
                choices = json_data.get('choices')
                if not choices or not choices[0].get('message', {}).get('content'):
                    raise ValueError("API returned an empty or invalid chat response.")
                result_text = choices[0]['message']['content']
                context.user_data['chat_history'].append({"role": "assistant", "content": result_text})
                try:
                    await processing_message.edit_text(result_text, parse_mode=ParseMode.MARKDOWN)
                except error.BadRequest:
                    logger.warning("Markdown parsing failed, sending as plain text.")
                    await processing_message.edit_text(result_text)
                return AWAITING_PROMPT

            elif task_type in ['image', 'video', 'image_edit']:
                prompt = message.text
                data = {"model": context.user_data['model'], "prompt": prompt}
                files = None
                if task_type == 'image':
                    endpoint, action = 'images/generations', ChatAction.UPLOAD_PHOTO
                    data['size'] = context.user_data.get('image_size', '1024x1024')
                elif task_type == 'video':
                    endpoint, action = 'video/generations', ChatAction.UPLOAD_VIDEO
                    data.update({'ratio': context.user_data.get('video_ratio', '16:9'), 'quality': '480p', 'duration': 4})
                else: 
                    endpoint, action = 'images/edits', ChatAction.UPLOAD_PHOTO
                    image_path = context.user_data['image_edit_path']
                    files = {'image': open(image_path, 'rb')}

                await context.bot.send_chat_action(message.chat_id, action)
                response = await client.post(f"{A4F_API_BASE_URL}/{endpoint}", headers=headers, data=data, files=files if files else None, timeout=180)
                if files: files['image'].close()
                response.raise_for_status()
                json_data = response.json()
                data_list = json_data.get('data')
                if not data_list or not data_list[0].get('url'):
                    raise ValueError("API returned an empty or invalid result.")
                
                url = data_list[0]['url']
                caption_map = {"image": "🎨 Prompt:", "video": "🎬 Prompt:", "image_edit": "🖌️ Edit:"}
                caption = f"{caption_map[task_type]} {prompt}"
                
                if task_type in ['image', 'image_edit']:
                    await context.bot.send_photo(message.chat_id, photo=url, caption=caption)
                else:
                    await context.bot.send_video(message.chat_id, video=url, caption=caption)

            elif task_type == 'tts':
                await context.bot.send_chat_action(message.chat_id, ChatAction.RECORD_VOICE)
                data = {"model": context.user_data['model'], "input": message.text, "voice": context.user_data.get('tts_voice', 'alloy')}
                response = await client.post(f"{A4F_API_BASE_URL}/audio/speech", headers=headers, json=data, timeout=60)
                response.raise_for_status()
                await context.bot.send_voice(message.chat_id, voice=response.content, caption=f"🗣️ Voice: {context.user_data.get('tts_voice', 'alloy').capitalize()}")

            elif task_type == 'transcription':
                await context.bot.send_chat_action(message.chat_id, ChatAction.TYPING)
                file_obj = await (message.voice or message.audio).get_file()
                temp_filename = f"temp_{uuid.uuid4()}.ogg"
                await file_obj.download_to_drive(temp_filename)
                context.user_data['temp_file_path'] = temp_filename
                with open(temp_filename, 'rb') as f:
                    response = await client.post(f"{A4F_API_BASE_URL}/audio/transcriptions", headers=headers, files={'file': f}, data={'model': context.user_data['model']}, timeout=120)
                response.raise_for_status()
                json_data = response.json()
                transcribed_text = json_data.get('text')
                if transcribed_text is None:
                    raise ValueError("API did not return a transcription.")
                await processing_message.edit_text(f"<b>Transcription:</b>\n\n<i>{transcribed_text}</i>", parse_mode=ParseMode.HTML)
                await message.reply_text("✨ Task complete!", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]]))
                return SELECTING_ACTION

        await processing_message.delete()
        if task_type not in ['transcription', 'chat']:
            await message.reply_text("✨ Task complete!", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]]))

    except (httpx.RequestError, ValueError, KeyError, IndexError) as e:
        logger.error(f"An error occurred in process_task: {e}", exc_info=True)
        await refund_credit(user_id)
        if isinstance(e, httpx.RequestError):
            await handle_api_error(processing_message, e)
        else:
            error_message = f"❌ *API Response Error:*\n{escape_markdown_v2(str(e))}"
            await processing_message.edit_text(error_message, parse_mode=ParseMode.MARKDOWN_V2)
            
    except Exception as e:
        logger.error(f"A critical internal error occurred in process_task: {e}", exc_info=True)
        await refund_credit(user_id)
        await processing_message.edit_text("❌ A critical internal error occurred. Credit has been refunded.")
    
    finally:
        await cleanup_files(context.user_data.pop('image_edit_path', None), context.user_data.pop('temp_file_path', None))

    return SELECTING_ACTION

async def process_request(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    return await process_task(update, context, context.user_data.get('category'))
async def tts_input_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    return await process_task(update, context, 'tts')
async def audio_transcription_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    return await process_task(update, context, 'transcription')
async def edit_prompt_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    return await process_task(update, context, 'image_edit')

async def image_for_edit_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not update.message.photo:
        await update.message.reply_text("That's not an image. Please send a photo (not a file).")
        return AWAITING_IMAGE_FOR_EDIT
    await update.message.reply_text("✅ Image received! Now, tell me how to edit it.")
    photo_file = await update.message.photo[-1].get_file()
    temp_filename = f"temp_{uuid.uuid4()}.jpg"
    await photo_file.download_to_drive(temp_filename)
    context.user_data['image_edit_path'] = temp_filename
    return AWAITING_EDIT_PROMPT

async def redeem_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Please provide a code. Usage: `/redeem YOUR-CODE`")
        return

    code_to_redeem = context.args[0]
    user_id = update.effective_user.id
    
    code_doc = await redeem_codes_collection.find_one_and_update(
        {"code": code_to_redeem, "is_active": True},
        {"$set": {"is_active": False}}
    )

    if code_doc:
        credits_to_add = code_doc["credits"]
        user_update_result = await users_collection.find_one_and_update(
            {"user_id": user_id},
            {"$inc": {"credits": credits_to_add}},
            upsert=True,
            return_document=ReturnDocument.AFTER
        )
        new_balance = user_update_result["credits"]
        await update.message.reply_text(f"🎉 Success! *{credits_to_add}* credits have been added. You now have *{new_balance}* credits.", parse_mode=ParseMode.MARKDOWN_V2)
    else:
        await update.message.reply_text("❌ This code is invalid or has already been used.")

async def generate_code_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("⛔️ This command is for administrators only.")
        return

    try:
        credits, amount = int(context.args[0]), int(context.args[1])
        if credits <= 0 or amount <= 0: raise ValueError
    except (IndexError, ValueError):
        await update.message.reply_text("Invalid format. Usage: `/code {credits} {amount}`")
        return

    new_codes_docs = []
    new_codes_text = []
    for _ in range(amount):
        full_code = f"SYPNS-{uuid.uuid4().hex[:12].upper()}-BOT"
        new_codes_docs.append({"code": full_code, "credits": credits, "is_active": True})
        new_codes_text.append(f"`{full_code}`")
    
    if new_codes_docs:
        await redeem_codes_collection.insert_many(new_codes_docs)
    message_text = f"✅ Generated *{amount}* new code(s), each worth *{credits}* credits:\n\n" + "\n".join(new_codes_text)
    await update.message.reply_text(message_text, parse_mode=ParseMode.MARKDOWN_V2)

async def give_credits_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("⛔️ This command is for administrators only.")
        return

    try:
        target_user_id = int(context.args[0])
        amount = int(context.args[1])
    except (IndexError, ValueError):
        await update.message.reply_text("Invalid format. Usage: `/cred {user_id} {amount}`")
        return
    
    result = await users_collection.find_one_and_update(
        {"user_id": target_user_id},
        {"$inc": {"credits": amount}},
        upsert=True,
        return_document=ReturnDocument.AFTER
    )
    if result:
        await update.message.reply_text(f"✅ Successfully gave {amount} credits to user `{target_user_id}`. Their new balance is {result['credits']}.", parse_mode=ParseMode.MARKDOWN_V2)
    else:
        await update.message.reply_text(f"❌ Could not update credits for user `{target_user_id}`.", parse_mode=ParseMode.MARKDOWN_V2)

async def broadcast_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("⛔️ This command is for administrators only.")
        return
    
    if not context.args:
        await update.message.reply_text("Please provide a message to broadcast. Usage: `/brod Your message here`")
        return
    message_to_send = update.message.text.split(' ', 1)[1]
    
    users_cursor = users_collection.find({}, {"user_id": 1})
    success_count = 0
    fail_count = 0
    
    await update.message.reply_text(f"📢 Starting broadcast...")
    
    async for user in users_cursor:
        user_id = user["user_id"]
        try:
            await context.bot.send_message(chat_id=user_id, text=message_to_send)
            success_count += 1
        except (error.Forbidden, error.BadRequest):
            fail_count += 1
        except Exception as e:
            logger.error(f"Error broadcasting to {user_id}: {e}")
            fail_count += 1
        await asyncio.sleep(0.1) # Avoid rate limiting
        
    await update.message.reply_text(f"Broadcast finished.\n✅ Sent successfully: {success_count}\n❌ Failed to send: {fail_count}")

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("⛔️ This command is for administrators only.")
        return
    
    total_users = await users_collection.count_documents({})
    active_codes = await redeem_codes_collection.count_documents({"is_active": True})
    
    stats_text = (
        f"📊 <b>Bot Statistics</b> 📊\n\n"
        f"👥 Total Users: <b>{total_users}</b>\n"
        f"🎟️ Active Redeem Codes: <b>{active_codes}</b>"
    )
    await update.message.reply_html(stats_text)

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Exception while handling an update:", exc_info=context.error)
    if isinstance(update, Update) and update.effective_message:
        try:
            await update.effective_message.reply_text("An unexpected error occurred. Please try again later.")
        except Exception as e:
            logger.error(f"Failed to send error message to user: {e}")

async def run_bot():
    if not TELEGRAM_BOT_TOKEN: logger.critical("!!! ERROR: TELEGRAM_BOT_TOKEN not set. !!!"); return
    if not MONGO_DB_URI or "<password>" in MONGO_DB_URI:
        logger.critical("!!! ERROR: MONGO_DB_URI is not set correctly. Please paste the standard connection string and your password. !!!")
        return
    if not ADMIN_CHAT_ID: logger.warning("!!! WARNING: ADMIN_CHAT_ID not set. Admin features disabled. !!!")
    
    try:
        await setup_database()
    except Exception as e:
        logger.critical(f"Database setup failed, cannot start bot: {e}")
        return

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).concurrent_updates(True).build()
    
    application.add_error_handler(error_handler)

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start_command)],
        states={
            SELECTING_ACTION: [CallbackQueryHandler(action_handler, pattern="^act_"), CallbackQueryHandler(start_command, pattern="^main_menu$")],
            SELECTING_MODEL: [CallbackQueryHandler(model_page_handler, pattern="^mp_"), CallbackQueryHandler(model_selection_handler, pattern="^(mr|ms)_"), CallbackQueryHandler(model_choice_handler, pattern="^mc_"), CallbackQueryHandler(start_command, pattern="^main_menu$")],
            AWAITING_IMAGE_SIZE: [CallbackQueryHandler(image_size_handler, pattern="^is_")],
            AWAITING_VIDEO_RATIO: [CallbackQueryHandler(video_ratio_handler, pattern="^vr_")],
            AWAITING_TTS_VOICE: [CallbackQueryHandler(tts_voice_handler, pattern="^tv_")],
            AWAITING_PROMPT: [MessageHandler(filters.TEXT & ~filters.COMMAND, process_request)],
            AWAITING_TTS_INPUT: [MessageHandler(filters.TEXT & ~filters.COMMAND, tts_input_handler)],
            AWAITING_AUDIO: [MessageHandler(filters.VOICE | filters.AUDIO, audio_transcription_handler)],
            AWAITING_IMAGE_FOR_EDIT: [MessageHandler(filters.PHOTO, image_for_edit_handler)],
            AWAITING_EDIT_PROMPT: [MessageHandler(filters.TEXT & ~filters.COMMAND, edit_prompt_handler)],
        },
        fallbacks=[CommandHandler("start", start_command), CommandHandler("cancel", cancel_command)],
        conversation_timeout=600, name="main_conversation", persistent=False,
    )
    application.add_handler(conv_handler)
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("newchat", new_chat_command))
    application.add_handler(CommandHandler("mycredits", credits_handler))
    application.add_handler(CommandHandler("redeem", redeem_command))
    application.add_handler(CommandHandler("code", generate_code_command))
    application.add_handler(CommandHandler("cred", give_credits_command))
    application.add_handler(CommandHandler("brod", broadcast_command))
    application.add_handler(CommandHandler("stats", stats_command))

    logger.info("Starting bot polling...")
    await application.initialize()
    await application.start()
    await application.updater.start_polling()
    
    # Keep the script running
    while True:
        await asyncio.sleep(3600)

if __name__ == "__main__":
    try:
        asyncio.run(run_bot())
    except KeyboardInterrupt:
        logger.info("Bot stopped manually.")