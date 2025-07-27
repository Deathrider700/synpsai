import logging
import os
import httpx
import uuid
import re
import json
import psutil
import random
import traceback
from collections import deque
from datetime import date, datetime
import asyncio
from typing import Deque

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, error, Message
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler, MessageHandler,
    filters, ContextTypes, ConversationHandler
)
from telegram.constants import ChatAction, ParseMode

TELEGRAM_BOT_TOKEN = "8385126802:AAEqYo6r3IyteSnPgLHUTpAaxdNU1SfHlB4"
INITIAL_A4F_KEYS = [
    "ddc-a4f-14783cb2294142ebb17d4bbb0e55d88f",
    "ddc-a4f-3b6b2b21fd794959bb008593eba6b88b",
    "ddc-a4f-d59bfe903deb4c3f9b0b724493e3d190",
    "ddc-a4f-3f75074b54f646cf87fda35032e4690d",
    "ddc-a4f-ce49dddb591e4bf48589971994a57a74",
    "ddc-a4f-89e3e7a18a3e467d9ac2d9a38067ca3b",
    "ddc-a4f-4e580ec612a94f98b1fe344edb812ab0",
    "ddc-a4f-03a8b8ae52a841e2af8b81c6f02f5e15",
    "ddc-a4f-1f90259072ad4d5d9077d466f2df42ee",
    "ddc-a4f-003d19a80e85466ab58eca86eceabbf8", "ddc-a4f-4c0658a7764c432c9aa8e4a6d409afb3"
]
A4F_API_BASE_URL = "https://api.a4f.co/v1"
ADMIN_CHAT_ID = 7088711806

DATA_DIR = "data"
USERS_DIR = os.path.join(DATA_DIR, "users")
REDEEM_CODES_FILE = os.path.join(DATA_DIR, "redeem_codes.json")
ERROR_LOG_FILE = os.path.join(DATA_DIR, "error_log.txt")
API_KEYS_STATUS_FILE = os.path.join(DATA_DIR, "api_keys.json")
SETTINGS_FILE = os.path.join(DATA_DIR, "settings.json")
WATCHED_USERS_FILE = os.path.join(DATA_DIR, "watched_users.json")

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

DEFAULT_SETTINGS = {"daily_credits": 10, "new_user_bonus": 20, "referral_bonus": 5, "maintenance": False}

MODELS = {
    "chat": ["provider-3/gpt-4", "provider-3/gpt-4.1-mini", "provider-6/o4-mini-high", "provider-6/o4-mini-low", "provider-6/o3-high", "provider-6/o3-medium", "provider-6/o3-low", "provider-3/gpt-4o-mini-search-preview", "provider-6/gpt-4o", "provider-6/gpt-4.1-nano", "provider-6/gpt-4.1-mini", "provider-3/gpt-4.1-nano", "provider-6/gpt-4.1", "provider-6/o4-mini-medium", "provider-1/gemini-2.5-pro", "provider-3/deepseek-v3", "provider-1/deepseek-v3-0324", "provider-1/sonar", "provider-1/sonar-deep-research", "provider-2/mistral-small", "provider-6/minimax-m1-40k", "provider-6/kimi-k2", "provider-3/kimi-k2", "provider-6/qwen3-coder-480b-a35b", "provider-3/llama-3.1-405b", "provider-3/qwen-3-235b-a22b-2507", "provider-1/mistral-large", "provider-2/llama-4-scout", "provider-2/llama-4-maverick", "provider-6/gemini-2.5-flash-thinking", "provider-6/gemini-2.5-flash", "provider-1/gemma-3-12b-it", "provider-1/llama-3.3-70b-instruct-turbo", "provider-2/codestral", "provider-1/llama-3.1-405b-instruct-turbo", "provider-3/llama-3.1-70b", "provider-2/qwq-32b", "provider-3/qwen-2.5-coder-32b", "provider-6/kimi-k2-instruct", "provider-2/mistral-saba", "provider-6/r1-1776", "provider-6/deepseek-r1-uncensored", "provider-1/deepseek-r1-0528", "provider-1/sonar-reasoning-pro", "provider-1/sonar-reasoning", "provider-1/sonar-pro", "provider-3/mistral-small-latest", "provider-3/magistral-medium-latest"],
    "image": ["provider-4/imagen-3", "provider-6/FLUX.1-kontext-max", "provider-6/FLUX.1-kontext-pro", "provider-6/FLUX.1-kontext-dev", "provider-3/FLUX.1-schnell", "provider-6/sana-1.5", "provider-3/FLUX.1-dev", "provider-6/FLUX.1-dev", "provider-1/FLUX.1.1-pro", "provider-6/FLUX.1-pro", "provider-1/FLUX.1-kontext-pro", "provider-1/FLUX.1-schnell", "provider-6/FLUX.1-1-pro", "provider-2/FLUX.1-schnell-v2", "provider-6/sana-1.5-flash"],
    "image_edit": ["provider-6/black-forest-labs-flux-1-kontext-dev", "provider-6/black-forest-labs-flux-1-kontext-pro", "provider-6/black-forest-labs-flux-1-kontext-max", "provider-3/flux-kontext-dev"],
    "video": ["provider-6/wan-2.1"],
    "tts": ["provider-3/tts-1"],
    "transcription": ["provider-3/whisper-1", "provider-6/distil-whisper-large-v3-en"],
    "summarize": ["provider-1/sonar"]
}
MODELS_PER_PAGE = 5
TTS_VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
IMAGE_SIZES = {"Square ⏹️": "1024x1024", "Wide  widescreen": "1792x1024", "Tall 📲": "1024x1792"}
VIDEO_RATIOS = {"Wide 🎬": "16:9", "Vertical 📱": "9:16", "Square 🖼️": "1:1"}
LOADING_MESSAGES = {"chat": "🤔 Cogitating on a thoughtful response...", "image": "🎨 Painting your masterpiece...", "image_edit": "🖌️ Applying artistic edits...", "video": "🎬 Directing your short film...", "tts": "🎙️ Warming up the vocal cords...", "transcription": "👂 Listening closely to your audio...", "summarize": "📚 Summarizing the document..."}
REASONING_MESSAGES = {"image": "⚙️ Reasoning about the visual elements...", "video": "🎥 Planning the scene and action..."}

(USER_MAIN, SELECTING_MODEL, AWAITING_PROMPT, AWAITING_TTS_INPUT, AWAITING_AUDIO, AWAITING_IMAGE_FOR_EDIT, AWAITING_EDIT_PROMPT, AWAITING_TTS_VOICE, AWAITING_VIDEO_RATIO, AWAITING_PERSONALITY, AWAITING_BROADCAST_CONFIRMATION, AWAITING_IMAGE_SIZE, SELECTING_PRESET_PERSONALITY, ADMIN_MAIN, ADMIN_AWAITING_INPUT, ADMIN_USERS_LIST, ADMIN_KEYS_LIST, SELECTING_VOICE_FOR_MODE, AWAITING_VOICE_MODE_INPUT, AWAITING_MIXER_CONCEPT_1, AWAITING_MIXER_CONCEPT_2, AWAITING_WEB_PROMPT) = range(22)

_active_api_keys, _settings, _watched_users = [], {}, set()

def load_settings():
    global _settings
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'r') as f: _settings = json.load(f)
    else: _settings = DEFAULT_SETTINGS.copy()
    for key, value in DEFAULT_SETTINGS.items(): _settings.setdefault(key, value)
    save_settings()

def save_settings():
    with open(SETTINGS_FILE, 'w') as f: json.dump(_settings, f, indent=4)

def load_watched_users():
    global _watched_users
    if os.path.exists(WATCHED_USERS_FILE):
        with open(WATCHED_USERS_FILE, 'r') as f:
            _watched_users = set(json.load(f))
    else:
        _watched_users = set()

def save_watched_users():
    with open(WATCHED_USERS_FILE, 'w') as f:
        json.dump(list(_watched_users), f)

def load_api_keys():
    global _active_api_keys
    if not os.path.exists(API_KEYS_STATUS_FILE):
        all_keys = [{"key": k, "active": True} for k in INITIAL_A4F_KEYS]
        save_api_keys(all_keys)
        _active_api_keys = [k['key'] for k in all_keys]
        return all_keys
    with open(API_KEYS_STATUS_FILE, 'r') as f:
        all_keys_status = json.load(f)
    _active_api_keys = [k['key'] for k in all_keys_status if k.get('active', True)]
    return all_keys_status

def save_api_keys(keys_status):
    with open(API_KEYS_STATUS_FILE, 'w') as f:
        json.dump(keys_status, f, indent=4)
    load_api_keys()

def get_random_api_key():
    return random.choice(_active_api_keys) if _active_api_keys else None

def setup_data_directory():
    for path in [DATA_DIR, USERS_DIR]:
        if not os.path.exists(path): os.makedirs(path)
    if not os.path.exists(REDEEM_CODES_FILE):
        with open(REDEEM_CODES_FILE, 'w') as f: json.dump({}, f)
    load_api_keys()
    load_settings()
    load_watched_users()

def load_user_data(user_id):
    user_file = os.path.join(USERS_DIR, f"{user_id}.json")
    today = date.today().isoformat()
    if not os.path.exists(user_file):
        user_data = {"credits": _settings["daily_credits"] + _settings["new_user_bonus"], "last_login_date": today, "is_new": True, "personality": None, "banned": False, "referral_code": f"REF-{uuid.uuid4().hex[:8].upper()}", "referred_by": None, "referrals_made": 0, "stats": {k: 0 for k in LOADING_MESSAGES.keys()}}
        save_user_data(user_id, user_data)
        return user_data
    try:
        with open(user_file, 'r') as f: user_data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        user_data = {"credits": 0, "last_login_date": "1970-01-01", "is_new": True, "personality": None, "banned": False}
    if user_data.get("last_login_date") != today and user_id != ADMIN_CHAT_ID:
        user_data["credits"] = user_data.get("credits", 0) + _settings["daily_credits"]
        user_data["last_login_date"] = today
    
    defaults = {"is_new": False, "personality": None, "banned": False, "referral_code": f"REF-{uuid.uuid4().hex[:8].upper()}", "referred_by": None, "referrals_made": 0, "stats": {k: 0 for k in LOADING_MESSAGES.keys()}}
    for key, value in defaults.items(): user_data.setdefault(key, value)
    if "stats" not in user_data or not isinstance(user_data["stats"], dict): user_data["stats"] = {k: 0 for k in LOADING_MESSAGES.keys()}
    
    save_user_data(user_id, user_data)
    return user_data

def save_user_data(user_id, data):
    with open(os.path.join(USERS_DIR, f"{user_id}.json"), 'w') as f: json.dump(data, f, indent=4)

def load_redeem_codes():
    try:
        with open(REDEEM_CODES_FILE, 'r') as f: return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError): return {}

def save_redeem_codes(codes):
    with open(REDEEM_CODES_FILE, 'w') as f: json.dump(codes, f, indent=4)

def check_and_use_credit(user_id: int) -> bool:
    if is_admin(user_id): return True
    if _settings.get('maintenance', False): return False
    user_data = load_user_data(user_id)
    if user_data.get("banned", False): return False
    if user_data.get("credits", 0) > 0:
        user_data["credits"] -= 1
        save_user_data(user_id, user_data)
        return True
    return False

def refund_credit(user_id: int):
    if user_id != ADMIN_CHAT_ID:
        user_data = load_user_data(user_id)
        user_data["credits"] = user_data.get("credits", 0) + 1
        save_user_data(user_id, user_data)
        logger.info(f"Refunded 1 credit to user {user_id}")

def escape_markdown_v2(text: str) -> str:
    return re.sub(f'([{re.escape(r"_*[]()~>#+-=|{}.!")}])', r'\\\1', text)

async def cleanup_files(*files):
    for file_path in files:
        if file_path and os.path.exists(file_path):
            try: os.remove(file_path)
            except OSError as e: logger.error(f"Error removing file {file_path}: {e}")

def create_paginated_keyboard(model_list, category, page=0):
    buttons = []
    start_index = page * MODELS_PER_PAGE
    end_index = start_index + MODELS_PER_PAGE
    for i, model in enumerate(model_list[start_index:end_index], start=start_index):
        buttons.append([InlineKeyboardButton(f"⚙️ {model.split('/')[-1]}", callback_data=f"ms_{category}_{i}")])
    nav_buttons = []
    if page > 0: nav_buttons.append(InlineKeyboardButton("⬅️ Back", callback_data=f"mp_{category}_{page-1}"))
    if end_index < len(model_list): nav_buttons.append(InlineKeyboardButton("Next ➡️", callback_data=f"mp_{category}_{page+1}"))
    if nav_buttons: buttons.append(nav_buttons)
    buttons.append([InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")])
    return InlineKeyboardMarkup(buttons)

def format_error_message(error_obj: Exception) -> str:
    try:
        response = getattr(error_obj, 'response', None)
        if response:
            status = response.status_code
            if status >= 500:
                return f"❌ *API Server Error ({status}):* The server is currently unavailable or overloaded\\. Please try again later\\."
            else:
                details = response.json()
                return f"❌ *API Error ({status}):*\n{escape_markdown_v2(details.get('error', {}).get('message', 'No details from API.'))}"
        else:
            return f"❌ *Connection Error:*\n{escape_markdown_v2(str(error_obj))}"
    except Exception:
        return f"❌ *An unexpected API error occurred:*\n{escape_markdown_v2(str(error_obj))}"

def is_admin(user_id: int) -> bool: return user_id == ADMIN_CHAT_ID

async def forward_to_admin_if_watched(message: Message, context: ContextTypes.DEFAULT_TYPE):
    user_id = message.chat.id
    if user_id in _watched_users and ADMIN_CHAT_ID:
        await context.bot.forward_message(chat_id=ADMIN_CHAT_ID, from_chat_id=message.chat_id, message_id=message.message_id)

async def notify_admin_of_new_user(context: ContextTypes.DEFAULT_TYPE, user: Update.effective_user, referred_by=None):
    if not ADMIN_CHAT_ID: return
    base_text = (f"🎉 *New User Alert*\n\n"
                 f"*Name:* {escape_markdown_v2(user.full_name)}\n"
                 f"*ID:* {user.id}\n"
                 f"*Username:* {'@' + escape_markdown_v2(user.username) if user.username else 'N/A'}")
    if referred_by:
        base_text += f"\n*Referred By:* {referred_by}"
    try: await context.bot.send_message(chat_id=ADMIN_CHAT_ID, text=base_text, parse_mode=ParseMode.MARKDOWN_V2)
    except Exception as e: logger.error(f"Failed to send new user notification to admin: {e}")

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user = update.effective_user
    if _settings.get('maintenance', False) and not is_admin(user.id):
        await update.message.reply_text("🛠️ The bot is currently under maintenance. Please try again later.")
        return ConversationHandler.END

    user_data = load_user_data(user.id)
    if user_data.get("banned", False):
        await update.message.reply_text("🚫 You have been banned from using this bot.")
        return ConversationHandler.END

    if user_data.get("is_new"):
        referred_by_id = None
        if context.args and len(context.args) > 0:
            referral_code = context.args[0]
            all_users = [f for f in os.listdir(USERS_DIR) if f.endswith('.json')]
            for user_file in all_users:
                referrer_id = int(user_file.split('.')[0])
                if referrer_id == user.id: continue
                referrer_data = load_user_data(referrer_id)
                if referrer_data.get('referral_code') == referral_code:
                    referrer_data['credits'] += _settings.get('referral_bonus', 5)
                    referrer_data['referrals_made'] = referrer_data.get('referrals_made', 0) + 1
                    save_user_data(referrer_id, referrer_data)
                    user_data['referred_by'] = referrer_id
                    referred_by_id = referrer_id
                    await context.bot.send_message(chat_id=referrer_id, text=f"🎉 Someone used your referral code! You've earned {_settings.get('referral_bonus', 5)} credits.")
                    break
        await notify_admin_of_new_user(context, user, referred_by=referred_by_id)
        user_data["is_new"] = False
        save_user_data(user.id, user_data)
        await update.message.reply_markdown_v2(f"🎉 Welcome\\! As a new user, you've received a bonus of *{_settings['new_user_bonus']}* credits\\!")
    
    for key in ['chat_history', 'image_edit_path', 'temp_file_path', 'last_prompt', 'voice_mode_voice', 'voice_chat_history', 'mixer_concept_1']: context.user_data.pop(key, None)
    
    keyboard = [
        [InlineKeyboardButton("💬 AI Chat", callback_data="act_chat"), InlineKeyboardButton("🎨 Image Gen", callback_data="act_image")],
        [InlineKeyboardButton("🖼️ Image Edit", callback_data="act_image_edit"), InlineKeyboardButton("🎬 Video Gen", callback_data="act_video")],
        [InlineKeyboardButton("🎙️ TTS", callback_data="act_tts"), InlineKeyboardButton("✍️ Transcription", callback_data="act_transcription")],
        [InlineKeyboardButton("🎤 Voice Mode", callback_data="act_voice_mode"), InlineKeyboardButton("🎨 Image Mixer", callback_data="act_mixer")],
        [InlineKeyboardButton("🌐 Web Pilot", callback_data="act_web")],
        [InlineKeyboardButton("👤 My Profile", callback_data="act_me"), InlineKeyboardButton("🎭 Set Personality", callback_data="act_personality")],
        [InlineKeyboardButton("❓ Help & Info", callback_data="act_help")]
    ]
    if is_admin(user.id):
        keyboard.append([InlineKeyboardButton("👑 Admin Panel", callback_data="act_admin")])
        
    reply_markup = InlineKeyboardMarkup(keyboard)
    welcome_text = (f"👋 *Welcome, {escape_markdown_v2(user.first_name)}*\\!\n\n"
                    f"I'm your all\\-in\\-one AI assistant, ready to help\\. All tasks cost 1 credit\\.")
    if user_data.get("personality"):
        welcome_text += f"\n\n🎭 Current AI Personality: _{escape_markdown_v2(user_data['personality'])}_"
    welcome_text += "\n\n👇 Select a tool below to get started\\."
    
    if update.callback_query:
        await update.callback_query.answer()
        await update.callback_query.edit_message_text(text=welcome_text, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN_V2)
    else:
        await update.message.reply_markdown_v2(text=welcome_text, reply_markup=reply_markup)
    return USER_MAIN

async def profile_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query_or_message = update.callback_query or update.message
    user_id = query_or_message.from_user.id
    user_data = load_user_data(user_id)
    stats = user_data.get('stats', {})
    stats_text = "\n".join([f"  •  {k.title()}: {v}" for k, v in stats.items() if v > 0]) or "_No activity yet\\._"
    
    credits = "♾️ Unlimited (Admin)" if is_admin(user_id) else user_data.get('credits', 0)
    
    bot_username = (await context.bot.get_me()).username
    referral_code = user_data.get('referral_code', 'N/A')
    referral_link = f"https://t.me/{bot_username}?start={referral_code}"

    text = (f"👤 *My Profile*\n\n"
            f"💰 *Credits:* {credits}\n"
            f"📈 *Users Referred:* {user_data.get('referrals_made', 0)}\n\n"
            f"🤝 *Your Referral Link:*\n{escape_markdown_v2(referral_link)}\n\n"
            f"📊 *Usage Statistics*\n{stats_text}\n\n"
            f"Share your link to earn *{_settings['referral_bonus']}* credits for each new user who joins\\! You can also use /redeem\\.")
            
    reply_markup = InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Back to Main Menu", callback_data="main_menu")]])
    if update.callback_query:
        await update.callback_query.answer()
        await update.callback_query.edit_message_text(text, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN_V2)
    else:
        await update.message.reply_markdown_v2(text, reply_markup=reply_markup)
    return USER_MAIN

async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query_or_message = update.callback_query or update.message
    help_text = ("*❔ Help & Information*\n\n*Available Commands:*\n"
                 "/start \\- Return to the main menu\\.\n"
                 "/newchat \\- Clear history for a fresh AI Chat\\.\n"
                 "/me \\- Check your credits, referral code, and stats\\.\n"
                 "/personality \\- Set a custom personality for the chat AI\\.\n"
                 "/redeem <CODE> \\- Redeem a code for credits\\.\n"
                 "/help \\- Show this help message\\.\n"
                 "/exit \\- Stop the current mode \\(like Voice Mode\\)\\.")
    if is_admin(query_or_message.from_user.id):
        help_text += ("\n\n*Admin Commands:*\n"
                      "/admin \\- Open the admin dashboard\\.\n"
                      "/globalstats \\- View aggregate stats for all users\\.\n"
                      "/msg <id> <msg> \\- Send a message to a user\\.\n"
                      "/cred <id> <amt> \\- Give/deduct credits\\.\n"
                      "/watch <id> \\- Forward a user's messages to you\\.\n"
                      "/unwatch <id> \\- Stop watching a user\\.\n"
                      "/listwatched \\- List all watched users\\.\n"
                      "/gethistory <id> \\- Get a user's chat history\\.\n"
                      "/getdata <id> \\- Get a user's data file\\.")

    reply_markup = InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Back to Main Menu", callback_data="main_menu")]])
    if update.callback_query:
        await update.callback_query.answer()
        await update.callback_query.edit_message_text(help_text, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN_V2)
    else:
        await update.message.reply_markdown_v2(help_text, reply_markup=reply_markup)
    return USER_MAIN

async def new_chat_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data.pop('chat_history', None)
    await update.message.reply_text("✅ Chat history cleared. The AI will now have no memory of our previous conversation.")

async def cancel_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await cleanup_files(context.user_data.pop('image_edit_path', None), context.user_data.pop('temp_file_path', None))
    for key in ['voice_mode_voice', 'voice_chat_history', 'mixer_concept_1']: context.user_data.pop(key, None)
    await update.message.reply_text("Action cancelled. Returning to the main menu.")
    await start_command(update, context)
    return ConversationHandler.END

async def set_personality_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    buttons = [[InlineKeyboardButton("📝 Custom", callback_data="p_custom")]]
    await update.callback_query.edit_message_text(
        "🎭 How would you like to set the personality?",
        reply_markup=InlineKeyboardMarkup(buttons)
    )
    return AWAITING_PERSONALITY

async def personality_command_entry(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    buttons = [[InlineKeyboardButton("📝 Custom", callback_data="p_custom")],
               [InlineKeyboardButton("🤖 Presets", callback_data="p_presets")]]
    await update.message.reply_text("🎭 How would you like to set the personality?", reply_markup=InlineKeyboardMarkup(buttons))
    return AWAITING_PERSONALITY

async def personality_choice_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    choice = query.data.split("_")[1]
    
    current_text = query.message.text
    current_reply_markup = query.message.reply_markup

    if choice == "custom":
        user_id = update.effective_user.id
        user_data = load_user_data(user_id)
        current_personality = user_data.get("personality")
        text = "🎭 *Set Custom AI Personality*\n\n"
        if current_personality: text += f"Current personality: _{escape_markdown_v2(current_personality)}_\n\n"
        text += "Please send me the new personality prompt for the AI \\(e\\.g\\., 'You are a helpful pirate'\\)\\. To remove it, send /clear\\."
        
        if text != current_text:
            await query.edit_message_text(text, parse_mode=ParseMode.MARKDOWN_V2)
        return AWAITING_PERSONALITY
        
    elif choice == "presets":
        buttons = [[InlineKeyboardButton(name, callback_data=f"ps_{name}")] for name in PERSONALITY_PRESETS.keys()]
        buttons.append([InlineKeyboardButton("⬅️ Back", callback_data="p_back")])
        new_reply_markup = InlineKeyboardMarkup(buttons)
        text = "🎭 Choose a preset personality:"
        
        if text != current_text or new_reply_markup != current_reply_markup:
            await query.edit_message_text(text, reply_markup=new_reply_markup)
        return SELECTING_PRESET_PERSONALITY

async def receive_personality_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    user_data = load_user_data(user_id)
    if update.message.text.lower() == '/clear':
        user_data["personality"] = None
        await update.message.reply_text("✅ Personality cleared.")
    else:
        user_data["personality"] = update.message.text
        await update.message.reply_text("✅ Personality set successfully!")
    save_user_data(user_id, user_data)
    context.user_data.pop('chat_history', None)
    await update.message.reply_text("Chat history has been cleared to apply the new personality.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]]))
    return USER_MAIN

async def preset_personality_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    choice = query.data.split("_", 1)[1]
    user_id = query.from_user.id
    user_data = load_user_data(user_id)
    user_data["personality"] = PERSONALITY_PRESETS[choice]
    save_user_data(user_id, user_data)
    context.user_data.pop('chat_history', None)
    await query.edit_message_text(f"✅ Personality set to: *{choice}*. Chat history cleared.", parse_mode=ParseMode.MARKDOWN_V2)
    await query.message.reply_text("Returning to main menu...", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]]))
    return USER_MAIN

async def action_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    _prefix, category = query.data.split('_', 1)
    if category == 'help': return await help_handler(update, context)
    if category == 'me': return await profile_handler(update, context)
    if category == 'personality': return await set_personality_handler(update, context)
    if category == 'admin': return await admin_panel(update, context)
    if category == 'voice_mode':
        keyboard = [[InlineKeyboardButton(v.capitalize(), callback_data=f"vm_voice_{v}") for v in TTS_VOICES[:3]], [InlineKeyboardButton(v.capitalize(), callback_data=f"vm_voice_{v}") for v in TTS_VOICES[3:]]]
        await query.edit_message_text("🗣️ First, choose a voice for our conversation.", reply_markup=InlineKeyboardMarkup(keyboard))
        return SELECTING_VOICE_FOR_MODE
    if category == 'mixer':
        await query.edit_message_text("🎨 Welcome to the Image Mixer Studio!\n\nFirst, send me the primary concept or subject (e.g., 'a cat', 'a knight').")
        return AWAITING_MIXER_CONCEPT_1
    if category == 'web':
        await query.edit_message_text("🌐 Web Pilot Mode initiated. What would you like to know or which URL should I check?")
        return AWAITING_WEB_PROMPT

    context.user_data['category'] = category
    last_model = context.user_data.get(f'last_model_{category}')
    if last_model:
        short_model_name = last_model.split('/')[-1]
        keyboard = [[InlineKeyboardButton(f"🚀 Use Last: {short_model_name}", callback_data=f"mr_{category}")], [InlineKeyboardButton("📋 Choose Another Model", callback_data=f"mc_{category}")]]
        await query.edit_message_text(f"You previously used {escape_markdown_v2(short_model_name)}\\. Use it again?", reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.MARKDOWN_V2)
        return SELECTING_MODEL
    return await show_model_selection(query, context)

async def show_model_selection(update_or_query, context: ContextTypes.DEFAULT_TYPE, page=0) -> int:
    category = context.user_data['category']
    model_list = MODELS.get(category, [])
    reply_markup = create_paginated_keyboard(model_list, category, page)
    text = f"💎 *Select a Model for {category.replace('_', ' ').title()}*"
    message = update_or_query.message if hasattr(update_or_query, 'message') else update_or_query
    if text != message.text or reply_markup != message.reply_markup:
        await message.edit_text(text=text, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN_V2)
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
            await query.edit_message_text("Sorry, there was an error. Please try again."); return USER_MAIN
    if not category or not model_name:
        await query.edit_message_text("Sorry, an error occurred. Returning to the main menu."); return await start_command(update, context)
    context.user_data.update({'model': model_name, f'last_model_{category}': model_name, 'category': category})
    msg_text = f"✅ Model Selected: {escape_markdown_v2(model_name.split('/')[-1])}\n\n"
    
    if category == "image":
        await query.edit_message_text(msg_text + "📏 Now, choose an aspect ratio\\.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton(name, callback_data=f"is_{size}") for name, size in IMAGE_SIZES.items()]]), parse_mode=ParseMode.MARKDOWN_V2)
        return AWAITING_IMAGE_SIZE
    if category == "video":
        await query.edit_message_text(msg_text + "🎬 Now, choose a video ratio\\.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton(name, callback_data=f"vr_{ratio}") for name, ratio in VIDEO_RATIOS.items()]]), parse_mode=ParseMode.MARKDOWN_V2)
        return AWAITING_VIDEO_RATIO
    if category == "tts":
        await query.edit_message_text(msg_text + "🗣️ Now, choose a voice\\.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton(v.capitalize(), callback_data=f"tv_{v}") for v in TTS_VOICES[:3]], [InlineKeyboardButton(v.capitalize(), callback_data=f"tv_{v}") for v in TTS_VOICES[3:]]]), parse_mode=ParseMode.MARKDOWN_V2)
        return AWAITING_TTS_VOICE
    
    prompt_map = {"chat": "💬 What's on your mind?","transcription": "🎤 Send me a voice message or audio file.","image_edit": "🖼️ First, send the image you want to edit."}
    next_state_map = {"chat": AWAITING_PROMPT, "transcription": AWAITING_AUDIO, "image_edit": AWAITING_IMAGE_FOR_EDIT}
    await query.edit_message_text(msg_text + escape_markdown_v2(prompt_map[category]), parse_mode=ParseMode.MARKDOWN_V2)
    return next_state_map.get(category, USER_MAIN)

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
    if not check_and_use_credit(user_id):
        message_text = "🚫 You are out of credits! Use /redeem or refer friends to get more, or wait for your daily refill."
        if load_user_data(user_id).get("banned", False): message_text = "You are banned from using this bot."
        elif _settings.get('maintenance', False): message_text = "Bot is in maintenance, please try later."
        if update.callback_query: await update.callback_query.answer(message_text, show_alert=True)
        else: await update.effective_message.reply_text(message_text)
        return USER_MAIN

    message = update.effective_message
    processing_message = await message.reply_text(LOADING_MESSAGES.get(task_type, "⏳ Working..."))
    user_prompt = message.text
    context.user_data['last_prompt'] = user_prompt
    await forward_to_admin_if_watched(message, context)

    if reasoning_text := REASONING_MESSAGES.get(task_type):
        await asyncio.sleep(1)
        await processing_message.edit_text(reasoning_text)
        
        try:
            async with httpx.AsyncClient() as client:
                headers = {"Authorization": f"Bearer {get_random_api_key()}", "Content-Type": "application/json"}
                reasoning_payload = {
                    "model": "provider-1/sonar-reasoning-pro",
                    "messages": [{"role": "user", "content": f"Enhance this prompt for an AI generator, making it more descriptive and vivid: '{user_prompt}'"}]
                }
                response = await client.post(f"{A4F_API_BASE_URL}/chat/completions", headers=headers, json=reasoning_payload, timeout=120)
                response.raise_for_status()
                reasoned_prompt = response.json()['choices'][0]['message']['content']
                if reasoned_prompt:
                    user_prompt = reasoned_prompt
                    await processing_message.edit_text(f"⚙️ Reasoning complete.\n_New prompt: {user_prompt}_")
                    await asyncio.sleep(2)
        except Exception as e:
            logger.warning(f"Reasoning step failed, proceeding with original prompt. Error: {e}")

    max_retries = 3
    for attempt in range(max_retries):
        try:
            api_key = get_random_api_key()
            if not api_key:
                await processing_message.edit_text("❌ No active API keys available. Please contact the administrator."); break
            
            async with httpx.AsyncClient() as client:
                headers = {"Authorization": f"Bearer {api_key}"}
                response = None
                
                if task_type == 'chat':
                    await context.bot.send_chat_action(update.effective_chat.id, ChatAction.TYPING)
                    if 'chat_history' not in context.user_data:
                        context.user_data['chat_history'] = deque(maxlen=100)
                        if (personality := load_user_data(user_id).get("personality")):
                            context.user_data['chat_history'].append({"role": "system", "content": personality})
                    context.user_data['chat_history'].append({"role": "user", "content": user_prompt})
                    data = {"model": context.user_data['model'], "messages": list(context.user_data['chat_history'])}
                    user_data = load_user_data(user_id)
                    user_data['chat_history'] = list(context.user_data['chat_history'])
                    save_user_data(user_id, user_data)
                    headers["Content-Type"] = "application/json"
                    response = await client.post(f"{A4F_API_BASE_URL}/chat/completions", headers=headers, json=data, timeout=120)

                elif task_type in ['image', 'video', 'image_edit']:
                    data = {"model": context.user_data['model'], "prompt": user_prompt}
                    files = None
                    if task_type == 'image':
                        endpoint, action = 'images/generations', ChatAction.UPLOAD_PHOTO
                        data['size'] = context.user_data.get('image_size', '1024x1024')
                    elif task_type == 'video':
                        endpoint, action = 'video/generations', ChatAction.UPLOAD_VIDEO
                        data.update({'ratio': context.user_data.get('video_ratio', '16:9'), 'quality': '480p', 'duration': 4})
                    else: 
                        endpoint, action = 'images/edits', ChatAction.UPLOAD_PHOTO
                        files = {'image': open(context.user_data['image_edit_path'], 'rb')}
                    
                    await context.bot.send_chat_action(update.effective_chat.id, action)
                    
                    if files:
                        response = await client.post(f"{A4F_API_BASE_URL}/{endpoint}", headers=headers, data=data, files=files, timeout=180)
                    else:
                        headers["Content-Type"] = "application/json"
                        response = await client.post(f"{A4F_API_BASE_URL}/{endpoint}", headers=headers, json=data, timeout=180)
                    
                    if files: files['image'].close()
                
                elif task_type == 'tts':
                    await context.bot.send_chat_action(update.effective_chat.id, ChatAction.RECORD_VOICE)
                    data = {"model": context.user_data['model'], "input": user_prompt, "voice": context.user_data.get('tts_voice', 'alloy')}
                    headers["Content-Type"] = "application/json"
                    response = await client.post(f"{A4F_API_BASE_URL}/audio/speech", headers=headers, json=data, timeout=60)

                elif task_type in ['transcription', 'summarize']:
                    await context.bot.send_chat_action(update.effective_chat.id, ChatAction.TYPING)
                    if task_type == 'summarize':
                         data = {"model": context.user_data.get('model', 'provider-1/sonar'), "messages": [{"role": "system", "content": "You are a summarizing expert. Summarize the following document concisely and effectively."}, {"role": "user", "content": user_prompt}]}
                         headers["Content-Type"] = "application/json"
                         response = await client.post(f"{A4F_API_BASE_URL}/chat/completions", headers=headers, json=data, timeout=120)
                    else:
                        file_obj = await (message.voice or message.audio).get_file()
                        temp_filename = f"temp_{uuid.uuid4()}.ogg"
                        await file_obj.download_to_drive(temp_filename)
                        context.user_data['temp_file_path'] = temp_filename
                        with open(temp_filename, 'rb') as f:
                            response = await client.post(f"{A4F_API_BASE_URL}/audio/transcriptions", headers=headers, files={'file': f}, data={'model': context.user_data['model']}, timeout=120)

                response.raise_for_status()
                json_data = response.json() if task_type not in ['tts'] else None

                if task_type == 'chat' or task_type == 'summarize':
                    if not (choices := json_data.get('choices')) or not (result_text := choices[0].get('message', {}).get('content')): raise ValueError("API returned empty response.")
                    if task_type == 'chat':
                        context.user_data['chat_history'].append({"role": "assistant", "content": result_text})
                        user_data = load_user_data(user_id)
                        user_data['chat_history'] = list(context.user_data['chat_history'])
                        save_user_data(user_id, user_data)
                        keyboard = [[InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]]
                        text = result_text + f"\n\n_Conversation: {len(context.user_data['chat_history'])}/unlimited_"
                        try: 
                            final_message = await processing_message.edit_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.MARKDOWN)
                        except error.BadRequest: 
                            final_message = await processing_message.edit_text(text, reply_markup=InlineKeyboardMarkup(keyboard))
                        await forward_to_admin_if_watched(final_message, context)
                    else:
                        final_message = await processing_message.edit_text(f"*Summary:*\n\n{escape_markdown_v2(result_text)}", parse_mode=ParseMode.MARKDOWN_V2, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]]))
                        await forward_to_admin_if_watched(final_message, context)
                
                elif task_type in ['image', 'image_edit', 'video']:
                     if not (data_list := json_data.get('data')) or not data_list[0].get('url'): raise ValueError("API returned no media URL.")
                     media_url = data_list[0]['url']
                     caption = f"_{escape_markdown_v2(user_prompt)}_"
                     if task_type in ['image', 'image_edit']: 
                         sent_message = await context.bot.send_photo(update.effective_chat.id, photo=media_url, caption=caption, parse_mode=ParseMode.MARKDOWN)
                     else: 
                         sent_message = await context.bot.send_video(update.effective_chat.id, video=media_url, caption=caption, parse_mode=ParseMode.MARKDOWN)
                     await forward_to_admin_if_watched(sent_message, context)
                     await processing_message.delete()
                     await message.reply_text("✨ Task complete!", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]]))

                elif task_type == 'tts':
                    sent_message = await context.bot.send_voice(message.chat_id, voice=response.content, caption=f"🗣️ Voice: {context.user_data.get('tts_voice', 'alloy').capitalize()}")
                    await forward_to_admin_if_watched(sent_message, context)
                    await processing_message.delete()
                    await message.reply_text("✨ Task complete!", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]]))
                
                elif task_type == 'transcription':
                    if (transcribed_text := json_data.get('text')) is None: raise ValueError("API did not return a transcription.")
                    final_message = await processing_message.edit_text(f"*Transcription:*\n\n_{escape_markdown_v2(transcribed_text)}_", parse_mode=ParseMode.MARKDOWN_V2)
                    await forward_to_admin_if_watched(final_message, context)
                    await message.reply_text("✨ Task complete!", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]]))
                
                user_data = load_user_data(user_id)
                user_data['stats'][task_type] = user_data['stats'].get(task_type, 0) + 1
                save_user_data(user_id, user_data)
                
                if task_type == 'chat': return AWAITING_PROMPT
                else: return USER_MAIN

        except (httpx.RequestError, ValueError, KeyError, IndexError, json.JSONDecodeError) as e:
            logger.error(f"Error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                await processing_message.edit_text("⏳ Still thinking...")
                await asyncio.sleep(2)
                continue
            else:
                refund_credit(user_id)
                error_message = format_error_message(e)
                await processing_message.edit_text(error_message, parse_mode=ParseMode.MARKDOWN_V2)
                break
        except Exception as e:
            logger.error(f"A critical internal error occurred in process_task: {e}", exc_info=True)
            refund_credit(user_id)
            await processing_message.edit_text("❌ A critical internal error occurred. Credit has been refunded."); break

    await cleanup_files(context.user_data.pop('image_edit_path', None), context.user_data.pop('temp_file_path', None))
    return USER_MAIN

async def process_request(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int: return await process_task(update, context, context.user_data.get('category'))
async def tts_input_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int: return await process_task(update, context, 'tts')
async def audio_transcription_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int: return await process_task(update, context, 'transcription')
async def edit_prompt_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int: return await process_task(update, context, 'image_edit')
async def image_for_edit_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await forward_to_admin_if_watched(update.message, context)
    if not update.message.photo: await update.message.reply_text("That's not an image. Please send a photo."); return AWAITING_IMAGE_FOR_EDIT
    await update.message.reply_text("✅ Image received! Now, tell me how to edit it.")
    photo_file = await update.message.photo[-1].get_file()
    temp_filename = f"temp_{uuid.uuid4()}.jpg"
    await photo_file.download_to_drive(temp_filename)
    context.user_data['image_edit_path'] = temp_filename
    return AWAITING_EDIT_PROMPT

async def voice_mode_start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    voice = query.data.split('_')[-1]
    context.user_data['voice_mode_voice'] = voice
    context.user_data['voice_chat_history'] = deque(maxlen=100)
    context.user_data['voice_chat_history'].append({"role": "system", "content": "You are a voice assistant. Keep your responses concise and suitable for voice conversion."})
    await query.edit_message_text(f"🎤 Voice Mode Started with *{voice.capitalize()}* voice\\. Each voice message costs 1 credit\\.\n\nSend me a voice message to begin, or use /exit to stop\\.", parse_mode=ParseMode.MARKDOWN_V2)
    return AWAITING_VOICE_MODE_INPUT

async def voice_mode_input_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    await forward_to_admin_if_watched(update.message, context)
    if not check_and_use_credit(user_id):
        await update.message.reply_text("🚫 You're out of credits! Use /redeem or refer a friend. Voice mode stopped.")
        for key in ['voice_mode_voice', 'voice_chat_history']: context.user_data.pop(key, None)
        await start_command(update, context)
        return ConversationHandler.END

    processing_message = await update.message.reply_text("🎙️ Processing voice...")
    temp_filename = f"temp_{uuid.uuid4()}.ogg"
    ai_response_text = ""
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient() as client:
                api_key = get_random_api_key()
                if not api_key:
                    await processing_message.edit_text("❌ No active API keys. Please contact the administrator."); refund_credit(user_id); return AWAITING_VOICE_MODE_INPUT
                headers = {"Authorization": f"Bearer {api_key}"}

                await processing_message.edit_text(f"👂 Transcribing...")
                file_obj = await update.message.voice.get_file()
                await file_obj.download_to_drive(temp_filename)
                
                with open(temp_filename, 'rb') as f:
                    transcription_response = await client.post(f"{A4F_API_BASE_URL}/audio/transcriptions", headers=headers, files={'file': f}, data={'model': 'provider-6/distil-whisper-large-v3-en'}, timeout=120)
                transcription_response.raise_for_status()
                transcribed_text = transcription_response.json().get('text')
                if not transcribed_text: raise ValueError("Transcription failed or returned empty text.")
                
                context.user_data['voice_chat_history'].append({"role": "user", "content": transcribed_text})
                
                await processing_message.edit_text(f"🤔 Thinking...")
                chat_data = {"model": 'provider-3/gpt-4o-mini-search-preview', "messages": list(context.user_data['voice_chat_history'])}
                headers["Content-Type"] = "application/json"
                chat_response = await client.post(f"{A4F_API_BASE_URL}/chat/completions", headers=headers, json=chat_data, timeout=120)
                chat_response.raise_for_status()
                
                ai_response_text = chat_response.json().get('choices', [{}])[0].get('message', {}).get('content')
                if not ai_response_text: raise ValueError("Chat completion returned empty response.")
                context.user_data['voice_chat_history'].append({"role": "assistant", "content": ai_response_text})

                user_data = load_user_data(user_id)
                user_data['voice_chat_history'] = list(context.user_data['voice_chat_history'])
                save_user_data(user_id, user_data)

                await processing_message.edit_text(f"🗣️ Speaking...")
                tts_data = {"model": "provider-3/tts-1", "input": ai_response_text, "voice": context.user_data['voice_mode_voice']}
                tts_response = await client.post(f"{A4F_API_BASE_URL}/audio/speech", headers=headers, json=tts_data, timeout=60)
                tts_response.raise_for_status()

                sent_message = await context.bot.send_voice(chat_id=user_id, voice=tts_response.content)
                await forward_to_admin_if_watched(sent_message, context)
                await processing_message.delete()
                break
                
        except (httpx.RequestError, ValueError, KeyError, IndexError, json.JSONDecodeError) as e:
            logger.error(f"Error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                await processing_message.edit_text("⏳ Still thinking...")
                await asyncio.sleep(2)
                continue
            else:
                refund_credit(user_id)
                error_message = format_error_message(e)
                if "speech" in str(getattr(e, 'request', '') and getattr(e.request, 'url', '')):
                     await processing_message.edit_text("⚠️ Could not generate voice, sending response as text:")
                     await update.message.reply_text(ai_response_text)
                else:
                     await processing_message.edit_text(error_message, parse_mode=ParseMode.MARKDOWN_V2)
                break
        except Exception as e:
            logger.error(f"A critical error occurred in voice_mode_input_handler: {e}", exc_info=True)
            refund_credit(user_id)
            await processing_message.edit_text(f"❌ An error occurred during processing. Credit refunded. Please try again."); break
    
    await cleanup_files(temp_filename)
    return AWAITING_VOICE_MODE_INPUT

async def exit_voice_mode(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    for key in ['voice_mode_voice', 'voice_chat_history']: context.user_data.pop(key, None)
    await update.message.reply_text("🎤 Voice mode stopped. Returning to the main menu.")
    await start_command(update, context)
    return ConversationHandler.END

async def redeem_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args: await update.message.reply_text("Usage: /redeem YOUR-CODE"); return
    code_to_redeem, user_id, codes = context.args[0], update.effective_user.id, load_redeem_codes()
    if code_to_redeem in codes and codes[code_to_redeem]["is_active"]:
        credits_to_add, user_data = codes[code_to_redeem]["credits"], load_user_data(user_id)
        user_data["credits"] += credits_to_add
        codes[code_to_redeem]["is_active"] = False
        codes[code_to_redeem]["redeemed_by"] = user_id
        save_user_data(user_id, user_data)
        save_redeem_codes(codes)
        await update.message.reply_markdown_v2(f"🎉 Success\\! *{credits_to_add}* credits added\\. New balance: *{user_data['credits']}*\\.")
    else: await update.message.reply_text("❌ This code is invalid or has already been used.")

async def document_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if _settings.get('maintenance', False) and not is_admin(user_id): return
    doc = update.message.document
    if not doc or not doc.file_name.lower().endswith(('.txt', '.md', '.py', '.json', '.csv')): return
    
    await forward_to_admin_if_watched(update.message, context)
    if not check_and_use_credit(user_id):
        await update.message.reply_text("🚫 You are out of credits to summarize this document."); return
        
    processing_message = await update.message.reply_text("📥 Downloading document...")
    try:
        file = await doc.get_file()
        if file.file_size > 5 * 1024 * 1024:
            await processing_message.edit_text("❌ Document is too large (max 5MB)."); return
        
        temp_path = os.path.join(DATA_DIR, f"temp_{uuid.uuid4()}")
        await file.download_to_drive(temp_path)
        with open(temp_path, 'r', encoding='utf-8', errors='ignore') as f: content = f.read()
        os.remove(temp_path)
        
        if len(content) > 100000:
            await processing_message.edit_text("❌ Document content is too long to process."); return
        
        await processing_message.edit_text("📚 Summarizing document...")
        context.user_data.update({'category': 'summarize', 'model': 'provider-1/sonar'})
        context.user_data['last_prompt'] = content
        update.message.text = content
        await process_task(update, context, 'summarize')

    except Exception as e:
        refund_credit(user_id)
        await processing_message.edit_text(f"❌ Error processing document: {e}")
        logger.error(f"Error in document_handler: {e}", exc_info=True)

async def mixer_concept_1_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await forward_to_admin_if_watched(update.message, context)
    context.user_data['mixer_concept_1'] = update.message.text
    await update.message.reply_text("✅ Got it. Now, send the second concept, style, or modifier (e.g., 'a powerful wizard', 'in a cyberpunk city').")
    return AWAITING_MIXER_CONCEPT_2

async def mixer_concept_2_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    await forward_to_admin_if_watched(update.message, context)
    if not check_and_use_credit(user_id):
        await update.message.reply_text("🚫 You are out of credits for the Image Mixer.")
        return USER_MAIN

    concept_1 = context.user_data.pop('mixer_concept_1')
    concept_2 = update.message.text
    
    processing_message = await update.message.reply_text(LOADING_MESSAGES.get("image"))
    if reasoning_text := REASONING_MESSAGES.get("mixer"):
        await asyncio.sleep(1)
        await processing_message.edit_text(reasoning_text)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient() as client:
                api_key = get_random_api_key()
                if not api_key:
                    await processing_message.edit_text("❌ No active API keys."); return USER_MAIN
                
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                
                creative_brief = (
                    "You are a creative director AI. Your task is to merge two user-provided concepts into a single, highly detailed, "
                    "and visually descriptive prompt for an AI image generator. Combine the elements logically and creatively. "
                    f"Merge these concepts: '{concept_1}' and '{concept_2}'"
                )
                
                data = { "model": "provider-3/gpt-4o-mini-search-preview", "messages": [{"role": "user", "content": creative_brief}] }
                response = await client.post(f"{A4F_API_BASE_URL}/chat/completions", headers=headers, json=data, timeout=120)
                response.raise_for_status()
                
                final_prompt = response.json()['choices'][0]['message']['content']
                if not final_prompt: raise ValueError("Creative Director AI failed to generate a prompt.")
                
                await processing_message.edit_text(f"🎨 Painting your masterpiece...\n\n_Final Prompt: {final_prompt}_")
                
                image_data = { "model": "provider-4/imagen-3", "prompt": final_prompt, "size": "1024x1024" }
                image_response = await client.post(f"{A4F_API_BASE_URL}/images/generations", headers=headers, json=image_data, timeout=180)
                image_response.raise_for_status()
                
                image_url = image_response.json()['data'][0]['url']
                if not image_url: raise ValueError("Image generation failed to return a URL.")

                sent_message = await update.message.reply_photo(photo=image_url, caption=f"_{escape_markdown_v2(final_prompt)}_", parse_mode=ParseMode.MARKDOWN)
                await forward_to_admin_if_watched(sent_message, context)
                await processing_message.delete()

                user_data = load_user_data(user_id)
                user_data['stats']['image'] = user_data['stats'].get('image', 0) + 1
                save_user_data(user_id, user_data)

                return USER_MAIN
        except (httpx.RequestError, ValueError, KeyError, IndexError, json.JSONDecodeError) as e:
            logger.error(f"Error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                await processing_message.edit_text("⏳ Still thinking...")
                await asyncio.sleep(2)
                continue
            else:
                refund_credit(user_id)
                error_message = format_error_message(e)
                await processing_message.edit_text(error_message, parse_mode=ParseMode.MARKDOWN_V2)
                break
        except Exception as e:
            logger.error(f"Critical error in Image Mixer: {e}", exc_info=True)
            refund_credit(user_id)
            await processing_message.edit_text("❌ A critical internal error occurred. Credit refunded."); break
            
    return USER_MAIN

async def web_pilot_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    if not check_and_use_credit(user_id):
        await update.message.reply_text("🚫 You are out of credits for Web Pilot.")
        return USER_MAIN

    await forward_to_admin_if_watched(update.message, context)
    processing_message = await update.message.reply_text("🌐 Browsing the web...")
    prompt = update.message.text
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient() as client:
                api_key = get_random_api_key()
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                
                data = {
                    "model": "provider-3/gpt-4o-mini-search-preview",
                    "messages": [{"role": "user", "content": prompt}]
                }
                
                response = await client.post(f"{A4F_API_BASE_URL}/chat/completions", headers=headers, json=data, timeout=180)
                response.raise_for_status()

                json_data = response.json()
                if not (choices := json_data.get('choices')) or not (result_text := choices[0].get('message', {}).get('content')):
                    raise ValueError("Web Pilot returned an empty response.")

                final_message = await processing_message.edit_text(result_text)
                await forward_to_admin_if_watched(final_message, context)
                return AWAITING_WEB_PROMPT
                
        except (httpx.RequestError, ValueError, KeyError, IndexError, json.JSONDecodeError) as e:
            logger.error(f"Error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                await processing_message.edit_text("⏳ Still thinking...")
                await asyncio.sleep(2)
                continue
            else:
                refund_credit(user_id)
                error_message = format_error_message(e)
                await processing_message.edit_text(error_message, parse_mode=ParseMode.MARKDOWN_V2)
                break
        except Exception as e:
            logger.error(f"A critical error occurred in Web Pilot Mode: {e}", exc_info=True)
            refund_credit(user_id)
            await processing_message.edit_text("❌ A critical internal error occurred. Credit refunded."); break
    
    return AWAITING_WEB_PROMPT

async def admin_panel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("⛔️ This command is for administrators only.")
        return ConversationHandler.END

    text = f"⚙️ *Admin Dashboard*\n\nSelect an action to perform\\."
    maintenance_text = "✅ ON" if _settings.get('maintenance') else "❌ OFF"
    
    keyboard = [
        [InlineKeyboardButton(f"🛠️ Maintenance: {maintenance_text}", callback_data="admin_maintenance")],
        [InlineKeyboardButton("📊 Bot Stats", callback_data="admin_stats"), InlineKeyboardButton("⚙️ Bot Settings", callback_data="admin_settings")],
        [InlineKeyboardButton("🔑 API Keys", callback_data="admin_keys"), InlineKeyboardButton("👥 Users", callback_data="admin_users")],
        [InlineKeyboardButton("📢 Broadcast", callback_data="admin_broadcast"), InlineKeyboardButton("🎁 Generate Codes", callback_data="admin_gen_codes")],
        [InlineKeyboardButton("📜 Error Log", callback_data="admin_errors")]
    ]
    if update.callback_query:
        await update.callback_query.answer()
        await update.callback_query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.MARKDOWN_V2)
    else:
        await update.message.reply_markdown_v2(text, reply_markup=InlineKeyboardMarkup(keyboard))
    return ADMIN_MAIN

async def admin_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    action = query.data.split('_', 1)[1]
    
    if action == "maintenance":
        _settings['maintenance'] = not _settings.get('maintenance', False)
        save_settings()
        await query.message.reply_text(f"Maintenance mode is now {'ON' if _settings['maintenance'] else 'OFF'}.")
        return await admin_panel(update, context)

    elif action == "stats":
        total_users = len(os.listdir(USERS_DIR))
        active_codes = sum(1 for code in load_redeem_codes().values() if code.get("is_active"))
        try:
            cpu = psutil.cpu_percent(interval=1)
            ram = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            cpu_text = f"{cpu}%"
            ram_text = f"{ram.used/1e9:.2f}/{ram.total/1e9:.2f} GB ({ram.percent}%)"
            disk_text = f"{disk.used/1e9:.2f}/{disk.total/1e9:.2f} GB ({disk.percent}%)"
        except (PermissionError, FileNotFoundError):
            cpu_text = ram_text = disk_text = "N/A (Permission Denied)"

        text = (f"📊 *Bot & System Statistics*\n\n"
                f"*Users:* {total_users}\n"
                f"*Active Redeem Codes:* {active_codes}\n\n"
                f"💻 *CPU:* {escape_markdown_v2(cpu_text)}\n"
                f"🧠 *RAM:* {escape_markdown_v2(ram_text)}\n"
                f"💽 *Disk:* {escape_markdown_v2(disk_text)}")
        await query.edit_message_text(text, parse_mode=ParseMode.MARKDOWN_V2, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back", callback_data="admin_back")]]))

    elif action == "settings":
        text = "*⚙️ Bot Settings*\n\n"
        for key, value in _settings.items():
            text += f"{key}: {value}\n"
        text += "\nSend setting to change in key value format \\(e\\.g\\., daily_credits 15\\)\\."
        context.user_data['admin_action'] = 'change_setting'
        await query.edit_message_text(text, parse_mode=ParseMode.MARKDOWN_V2, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back", callback_data="admin_back")]]))
        return ADMIN_AWAITING_INPUT
        
    elif action == 'users':
        await query.edit_message_text("Enter User ID to manage, or send /all to list users\\.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back", callback_data="admin_back")]]))
        context.user_data['admin_action'] = 'manage_user'
        return ADMIN_AWAITING_INPUT

    elif action == 'keys':
        keys_status = load_api_keys()
        text = "🔑 *API Key Status*\n\n"
        for i, key_info in enumerate(keys_status):
            status = "✅ Active" if key_info.get('active', True) else "❌ Disabled"
            key_display = escape_markdown_v2(f"{key_info['key'][:12]}...{key_info['key'][-4:]}")
            text += f"{i}: {key_display} \\- {status}\n"
        text += "\nSend key index to toggle its status\\."
        context.user_data['admin_action'] = 'toggle_key'
        await query.edit_message_text(text, parse_mode=ParseMode.MARKDOWN_V2, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back", callback_data="admin_back")]]))
        return ADMIN_AWAITING_INPUT

    elif action == "broadcast":
        await query.edit_message_text("Send the message to broadcast to all users\\.")
        context.user_data['admin_action'] = 'broadcast'
        return ADMIN_AWAITING_INPUT
        
    elif action == "gen_codes":
        await query.edit_message_text("Send details in credits amount format \\(e\\.g\\., 10 5 to generate 5 codes worth 10 credits each\\)\\.")
        context.user_data['admin_action'] = 'gen_codes'
        return ADMIN_AWAITING_INPUT
        
    elif action == "errors":
        if not os.path.exists(ERROR_LOG_FILE) or os.path.getsize(ERROR_LOG_FILE) == 0:
            await query.edit_message_text("No errors logged yet\\.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back", callback_data="admin_back")]]))
        else:
            await query.message.reply_document(document=open(ERROR_LOG_FILE, 'rb'), filename="error_log.txt")
            await query.message.reply_text("Error log sent\\. Do you want to clear it?", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("✅ Yes, clear", callback_data="admin_clear_errors"), InlineKeyboardButton("⬅️ Back", callback_data="admin_back")]]))

    elif action == "clear_errors":
        if os.path.exists(ERROR_LOG_FILE):
             os.remove(ERROR_LOG_FILE)
             await query.edit_message_text("✅ Error log cleared\\.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back to Admin Panel", callback_data="admin_back")]]))
        return ADMIN_MAIN
        
    elif action == "back":
        return await admin_panel(update, context)

    return ADMIN_MAIN

async def admin_input_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    action = context.user_data.pop('admin_action', None)
    text = update.message.text
    
    if action == "change_setting":
        try:
            key, value = text.split(maxsplit=1)
            if key not in _settings:
                await update.message.reply_text("Invalid setting key.")
            else:
                _settings[key] = type(DEFAULT_SETTINGS[key])(value)
                save_settings()
                await update.message.reply_text(f"✅ Setting {key} updated to {_settings[key]}.")
        except Exception as e:
            await update.message.reply_text(f"Invalid format. Use key value. Error: {e}")

    elif action == "manage_user":
        if text.lower() == '/all':
             all_users = [f.split('.')[0] for f in os.listdir(USERS_DIR)]
             user_list = "\n".join([f"{uid}" for uid in all_users])
             await update.message.reply_markdown_v2(f"*All User IDs:*\n{user_list}")
        else:
            try:
                target_user_id = int(text)
                if not os.path.exists(os.path.join(USERS_DIR, f"{target_user_id}.json")):
                    await update.message.reply_text(f"No data for user ID {target_user_id}."); return await admin_panel(update, context)
                
                user_data = load_user_data(target_user_id)
                info_text = (f"ℹ️ *User Info: {target_user_id}*\n\n"
                             f"*Credits:* {user_data.get('credits', 'N/A')}\n*Last Login:* {user_data.get('last_login_date', 'N/A')}\n"
                             f"*Personality:* _{escape_markdown_v2(user_data.get('personality') or 'Not set')}_\n*Banned:* {'Yes' if user_data.get('banned') else 'No'}")
                
                keyboard = [
                    [InlineKeyboardButton("💰 Add/Deduct Credits", callback_data=f"admin_user_cred_{target_user_id}"),
                     InlineKeyboardButton("🚫 Ban" if not user_data.get('banned') else "✅ Unban", callback_data=f"admin_user_ban_{target_user_id}")],
                    [InlineKeyboardButton("⬅️ Back to Admin", callback_data="admin_back")]
                ]
                await update.message.reply_markdown_v2(info_text, reply_markup=InlineKeyboardMarkup(keyboard))

            except ValueError: await update.message.reply_text("Invalid User ID.")
            
    elif action == "toggle_key":
        try:
            key_index = int(text)
            keys_status = load_api_keys()
            if not 0 <= key_index < len(keys_status):
                await update.message.reply_text("Invalid key index.")
            else:
                keys_status[key_index]['active'] = not keys_status[key_index].get('active', True)
                save_api_keys(keys_status)
                await update.message.reply_text(f"✅ Key {key_index} has been {'enabled' if keys_status[key_index]['active'] else 'disabled'}.")
        except ValueError: await update.message.reply_text("Invalid index.")

    elif action == "broadcast":
        user_count = len(os.listdir(USERS_DIR))
        context.user_data['broadcast_message_text'] = text
        keyboard = [[InlineKeyboardButton("✅ Yes, send it", callback_data="brod_confirm_yes"), InlineKeyboardButton("❌ No, cancel", callback_data="brod_confirm_no")]]
        await update.message.reply_markdown_v2(f"You are about to send this message to *{user_count}* users\\. Are you sure?", reply_markup=InlineKeyboardMarkup(keyboard))
        return AWAITING_BROADCAST_CONFIRMATION

    elif action == "gen_codes":
        try:
            credits, amount = map(int, text.split())
            codes, new_codes_text = load_redeem_codes(), []
            for _ in range(amount):
                full_code = f"SYPNS-{uuid.uuid4().hex[:12].upper()}-BOT"
                codes[full_code] = {"credits": credits, "is_active": True}
                new_codes_text.append(f"{escape_markdown_v2(full_code)}")
            save_redeem_codes(codes)
            await update.message.reply_markdown_v2(f"✅ Generated *{amount}* codes worth *{credits}* credits:\n\n" + "\n".join(new_codes_text))
        except ValueError: await update.message.reply_text("Invalid format. Use credits amount.")

    await admin_panel(update, context)
    return ADMIN_MAIN

async def admin_user_actions_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    _prefix, _cat, action, user_id_str = query.data.split('_')
    target_user_id = int(user_id_str)
    
    if action == "cred":
        context.user_data['admin_action'] = f'add_credits_{target_user_id}'
        await query.edit_message_text(f"How many credits to add/deduct from user {target_user_id}? (Use a negative number to deduct)")
        return ADMIN_AWAITING_INPUT
        
    elif action == "ban":
        user_data = load_user_data(target_user_id)
        user_data['banned'] = not user_data.get('banned', False)
        save_user_data(target_user_id, user_data)
        status = 'banned' if user_data['banned'] else 'unbanned'
        await query.edit_message_text(f"✅ User {target_user_id} has been {status}.")
        try:
            await context.bot.send_message(chat_id=target_user_id, text=f"You have been {status} by an administrator.")
        except Exception as e:
            logger.warning(f"Could not notify user {target_user_id} of status change: {e}")
            
    await admin_panel(update, context)
    return ADMIN_MAIN

async def broadcast_confirm_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    if query.data == "brod_confirm_no":
        await query.edit_message_text("Broadcast cancelled.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back to Admin Panel", callback_data="admin_back")]]))
        return ADMIN_MAIN
    
    await query.edit_message_text("📢 Sending broadcast...")
    user_files = os.listdir(USERS_DIR)
    success_count, fail_count = 0, 0
    text = context.user_data.pop('broadcast_message_text', None)
    if not text:
        await query.edit_message_text("Error: Broadcast message not found.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back to Admin Panel", callback_data="admin_back")]])); return ADMIN_MAIN
    
    for filename in user_files:
        user_id = int(filename.split('.')[0])
        try:
            await context.bot.send_message(chat_id=user_id, text=text, parse_mode=ParseMode.MARKDOWN_V2)
            success_count += 1
            await asyncio.sleep(0.1) 
        except (error.Forbidden, error.BadRequest):
            fail_count += 1
        except Exception as e:
            logger.error(f"Error broadcasting to {user_id}: {e}")
            fail_count += 1
            
    final_text = f"Broadcast finished\\.\n✅ Sent: {success_count}\n❌ Failed: {fail_count}"
    await query.edit_message_text(final_text, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back to Admin Panel", callback_data="admin_back")]]), parse_mode=ParseMode.MARKDOWN_V2)
    return ADMIN_MAIN

async def admin_add_credits_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    action = context.user_data.pop('admin_action', None)
    if not action or not action.startswith('add_credits_'): return
    
    try:
        target_user_id = int(action.split('_')[2])
        amount = int(update.message.text)
        user_data = load_user_data(target_user_id)
        user_data['credits'] += amount
        save_user_data(target_user_id, user_data)
        operation = "Gave" if amount >= 0 else "Deducted"
        await update.message.reply_markdown_v2(f"✅ {operation} *{abs(amount)}* credits to/from user {target_user_id}\\. New balance: *{user_data['credits']}*\\.")
    except (IndexError, ValueError):
        await update.message.reply_text("Invalid amount.")
        
    await admin_panel(update, context)
    return ADMIN_MAIN

async def admin_command_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("⛔️ This command is for administrators only.")
        return

    command = update.message.text.split()[0][1:].lower()
    
    if command == "msg":
        try:
            target_user_id = int(context.args[0])
            message_text = " ".join(context.args[1:])
            if not message_text:
                await update.message.reply_text("Usage: /msg <user_id> <message>")
                return
            await context.bot.send_message(chat_id=target_user_id, text=f"✉️ A message from the admin:\n\n{message_text}")
            await update.message.reply_text(f"✅ Message sent to user {target_user_id}.")
        except (IndexError, ValueError):
            await update.message.reply_text("Usage: /msg <user_id> <message>")
    
    elif command == "cred":
        try:
            target_user_id, amount = int(context.args[0]), int(context.args[1])
            user_data = load_user_data(target_user_id)
            user_data['credits'] += amount
            save_user_data(target_user_id, user_data)
            operation = "Gave" if amount >= 0 else "Deducted"
            await update.message.reply_markdown_v2(f"✅ {operation} *{abs(amount)}* credits to/from user {target_user_id}\\. New balance: *{user_data['credits']}*\\.")
        except (IndexError, ValueError):
            await update.message.reply_text("Usage: /cred <user_id> <amount> (use negative for deduction)")

    elif command in ["watch", "unwatch"]:
        try:
            target_user_id = int(context.args[0])
            if command == "watch":
                _watched_users.add(target_user_id)
                await update.message.reply_text(f"👀 Now watching user {target_user_id}. All their interactions will be forwarded here.")
            else: # unwatch
                _watched_users.discard(target_user_id)
                await update.message.reply_text(f"✅ Stopped watching user {target_user_id}.")
            save_watched_users()
        except (IndexError, ValueError):
            await update.message.reply_text(f"Usage: /{command} <user_id>")

    elif command == "listwatched":
        if not _watched_users:
            await update.message.reply_text("No users are currently being watched.")
        else:
            text = "*👀 Watched Users:*\n" + "\n".join([f"{uid}" for uid in _watched_users])
            await update.message.reply_markdown_v2(text)
            
    elif command == "getdata":
        try:
            target_user_id = int(context.args[0])
            user_file = os.path.join(USERS_DIR, f"{target_user_id}.json")
            if os.path.exists(user_file):
                await update.message.reply_document(document=open(user_file, 'rb'))
            else:
                await update.message.reply_text(f"No data file found for user {target_user_id}.")
        except (IndexError, ValueError):
            await update.message.reply_text("Usage: /getdata <user_id>")

    elif command == "gethistory":
        try:
            target_user_id = int(context.args[0])
            user_data = load_user_data(target_user_id)
            chat_history = user_data.get('chat_history', [])
            voice_history = user_data.get('voice_chat_history', [])
            
            if not chat_history and not voice_history:
                await update.message.reply_text(f"No saved history found for user {target_user_id}.")
                return

            history_text = f"--- Chat History for {target_user_id} ---\n\n"
            if chat_history:
                history_text += "--- Text Chat ---\n"
                for msg in chat_history:
                    history_text += f"[{msg['role'].upper()}]: {msg['content']}\n"
            
            if voice_history:
                history_text += "\n--- Voice Chat ---\n"
                for msg in voice_history:
                    history_text += f"[{msg['role'].upper()}]: {msg['content']}\n"

            history_file = f"history_{target_user_id}.txt"
            with open(history_file, "w", encoding="utf-8") as f:
                f.write(history_text)
            
            await update.message.reply_document(document=open(history_file, 'rb'))
            os.remove(history_file)

        except (IndexError, ValueError):
            await update.message.reply_text("Usage: /gethistory <user_id>")
            
    elif command == "globalstats":
        total_credits = 0
        total_stats = {k: 0 for k in LOADING_MESSAGES.keys()}
        user_files = [f for f in os.listdir(USERS_DIR) if f.endswith('.json')]
        user_activity = {}

        for user_file in user_files:
            user_id_str = user_file.split('.')[0]
            with open(os.path.join(USERS_DIR, user_file), 'r') as f:
                data = json.load(f)
                total_credits += data.get('credits', 0)
                user_stats = data.get('stats', {})
                activity_count = 0
                for key in total_stats:
                    usage = user_stats.get(key, 0)
                    total_stats[key] += usage
                    activity_count += usage
                if activity_count > 0:
                    user_activity[user_id_str] = activity_count

        sorted_users = sorted(user_activity.items(), key=lambda item: item[1], reverse=True)
        top_5_users = sorted_users[:5]

        text = (f"🌐 *Global Bot Statistics*\n\n"
                f"💰 *Total Credits in Circulation:* {total_credits}\n\n"
                f"📊 *Aggregate Tool Usage:*\n")
        
        for key, value in sorted(total_stats.items(), key=lambda item: item[1], reverse=True):
            if value > 0:
                text += f"  •  {key.title()}: {value}\n"
        
        text += "\n🏆 *Top 5 Most Active Users:*\n"
        if top_5_users:
            for i, (user_id, count) in enumerate(top_5_users):
                text += f"{i+1}\\. {user_id} \\({count} actions\\)\n"
        else:
            text += "_No user activity recorded yet\\._"
        
        await update.message.reply_markdown_v2(text)

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Exception while handling an update:", exc_info=context.error)
    try:
        with open(ERROR_LOG_FILE, 'a') as f: f.write(f"--- Time: {datetime.now()} ---\n{traceback.format_exc()}\n\n")
    except Exception as e: logger.error(f"Could not write to error log file: {e}")
    if isinstance(update, Update) and update.effective_message:
        try: await update.effective_message.reply_text("An unexpected error occurred. The admin has been notified.")
        except Exception as e: logger.error(f"Failed to send error message to user: {e}")

def main() -> None:
    setup_data_directory()
    if not TELEGRAM_BOT_TOKEN: logger.critical("TOKEN not set!"); return
    if not INITIAL_A4F_KEYS: logger.critical("API KEYS not set!"); return
    if not ADMIN_CHAT_ID: logger.warning("ADMIN_CHAT_ID not set!");
    
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).concurrent_updates(True).build()
    
    user_conv_handler = ConversationHandler(
        entry_points=[
            CommandHandler("start", start_command),
            CommandHandler("personality", personality_command_entry),
            CallbackQueryHandler(start_command, pattern="^main_menu$")
        ],
        states={
            USER_MAIN: [
                CallbackQueryHandler(action_handler, pattern="^act_"),
                CallbackQueryHandler(start_command, pattern="^main_menu$")
            ],
            SELECTING_MODEL: [
                CallbackQueryHandler(model_page_handler, pattern="^mp_"),
                CallbackQueryHandler(model_selection_handler, pattern="^(mr|ms)_"),
                CallbackQueryHandler(model_choice_handler, pattern="^mc_"),
            ],
            AWAITING_PROMPT: [MessageHandler(filters.TEXT & ~filters.COMMAND, process_request)],
            AWAITING_TTS_INPUT: [MessageHandler(filters.TEXT & ~filters.COMMAND, tts_input_handler)],
            AWAITING_AUDIO: [MessageHandler(filters.VOICE | filters.AUDIO, audio_transcription_handler)],
            AWAITING_IMAGE_FOR_EDIT: [MessageHandler(filters.PHOTO, image_for_edit_handler)],
            AWAITING_EDIT_PROMPT: [MessageHandler(filters.TEXT & ~filters.COMMAND, edit_prompt_handler)],
            AWAITING_VIDEO_RATIO: [CallbackQueryHandler(video_ratio_handler, pattern="^vr_")],
            AWAITING_TTS_VOICE: [CallbackQueryHandler(tts_voice_handler, pattern="^tv_")],
            AWAITING_IMAGE_SIZE: [CallbackQueryHandler(image_size_handler, pattern="^is_")],
            AWAITING_PERSONALITY: [
                CallbackQueryHandler(personality_choice_handler, pattern="^p_"),
                MessageHandler(filters.TEXT & ~filters.COMMAND, receive_personality_handler)
            ],
            SELECTING_PRESET_PERSONALITY: [
                CallbackQueryHandler(preset_personality_handler, pattern="^ps_"),
                CallbackQueryHandler(personality_choice_handler, pattern="^p_back")
            ],
            SELECTING_VOICE_FOR_MODE: [CallbackQueryHandler(voice_mode_start_handler, pattern="^vm_voice_")],
            AWAITING_VOICE_MODE_INPUT: [MessageHandler(filters.VOICE, voice_mode_input_handler)],
            AWAITING_MIXER_CONCEPT_1: [MessageHandler(filters.TEXT & ~filters.COMMAND, mixer_concept_1_handler)],
            AWAITING_MIXER_CONCEPT_2: [MessageHandler(filters.TEXT & ~filters.COMMAND, mixer_concept_2_handler)],
            AWAITING_WEB_PROMPT: [MessageHandler(filters.TEXT & ~filters.COMMAND, web_pilot_handler)],
        },
        fallbacks=[CommandHandler("start", start_command), CommandHandler("cancel", cancel_handler), CommandHandler("exit", exit_voice_mode)],
        conversation_timeout=1800, name="user_conversation", persistent=False, allow_reentry=True
    )
    
    admin_conv_handler = ConversationHandler(
        entry_points=[CommandHandler("admin", admin_panel)],
        states={
            ADMIN_MAIN: [
                CallbackQueryHandler(admin_callback_handler, pattern="^admin_"),
                CallbackQueryHandler(admin_user_actions_handler, pattern="^admin_user_"),
            ],
            ADMIN_AWAITING_INPUT: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, admin_input_handler),
                CallbackQueryHandler(admin_callback_handler, pattern="^admin_back"),
            ],
            AWAITING_BROADCAST_CONFIRMATION: [CallbackQueryHandler(broadcast_confirm_handler, pattern="^brod_confirm_")],
        },
        fallbacks=[CommandHandler("admin", admin_panel)],
        conversation_timeout=600, name="admin_conversation", persistent=False, allow_reentry=True
    )

    application.add_handler(user_conv_handler)
    application.add_handler(admin_conv_handler)
    
    application.add_handler(CommandHandler("help", help_handler))
    application.add_handler(CommandHandler("newchat", new_chat_command))
    application.add_handler(CommandHandler(["me", "mycredits"], profile_handler))
    application.add_handler(CommandHandler("redeem", redeem_command))
    
    admin_cmds = ["msg", "cred", "watch", "unwatch", "listwatched", "getdata", "gethistory", "globalstats"]
    application.add_handler(CommandHandler(admin_cmds, admin_command_handler))

    application.add_handler(MessageHandler(filters.Document.ALL, document_handler))
    
    application.add_error_handler(error_handler)
    logger.info("Bot is running...")
    application.run_polling()

if __name__ == "__main__":
    main()