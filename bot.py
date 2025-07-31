import logging
import os
import httpx
import uuid
import re
import json
import psutil
import random
import traceback
import requests
import signal
import sys
from collections import deque
from datetime import date, datetime
import asyncio
from typing import Deque
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, error, Message, InputMediaAnimation
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler, MessageHandler,
    filters, ContextTypes, ConversationHandler
)
from telegram.constants import ChatAction, ParseMode
from telegram.request import HTTPXRequest
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, DuplicateKeyError

TELEGRAM_BOT_TOKEN = "7649463897:AAE_oz0r72b-tRwZVWty18xZl2ku_IbqiDY"
MONGODB_URL = "mongodb+srv://kuntaldebnath588:Qz4WfNVFVfpoLjOk@cluster0.c9losyl.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
INITIAL_A4F_KEYS = [
    "ddc-a4f-14783cb2294142ebb17d4bbb0e55d88f", "ddc-a4f-3b6b2b21fd794959bb008593eba6b88b",
    "ddc-a4f-d59bfe903deb4c3f9b0b724493e3d190", "ddc-a4f-3f75074b54f646cf87fda35032e4690d",
    "ddc-a4f-ce49dddb591e4bf48589971994a57a74", "ddc-a4f-89e3e7a18a3e467d9ac2d9a38067ca3b",
    "ddc-a4f-4e580ec612a94f98b1fe344edb812ab0", "ddc-a4f-03a8b8ae52a841e2af8b81c6f02f5e15",
    "ddc-a4f-1f90259072ad4d5d9077d466f2df42ee", "ddc-a4f-003d19a80e85466ab58eca86eceabbf8",
    "ddc-a4f-4c0658a7764c432c9aa8e4a6d409afb3"
]
A4F_API_BASE_URL = "https://api.a4f.co/v1"
ADMIN_CHAT_ID = 7088711806
DEFAULT_VOICE_MODE_MODEL = "provider-6/gpt-4.1"
WORKFLOW_DESIGNER_MODEL = "provider-3/gpt-4.1-mini"

PRO_REASONING_MODEL = "provider-6/gpt-4.1"
PRO_MODEL_MAPPING = {
    "web_search": "provider-3/gpt-4o-mini-search-preview",
    "coding": "provider-3/qwen-2.5-coder-32b",
    "reasoning": "provider-6/o4-mini-high",
    "general_chat": "provider-6/gpt-4.1",
    "image_generation": "provider-4/imagen-3",
    "image_editing": "provider-6/black-forest-labs-flux-1-kontext-max",
    "video_generation": "provider-6/wan-2.1"
}

DATA_DIR = "data"
USERS_DIR = os.path.join(DATA_DIR, "users")
TEMP_DIR = os.path.join(DATA_DIR, "temp")
WORKFLOWS_DIR = os.path.join(DATA_DIR, "workflows")
MARKETPLACE_DIR = os.path.join(DATA_DIR, "marketplace")
REDEEM_CODES_FILE = os.path.join(DATA_DIR, "redeem_codes.json")
ERROR_LOG_FILE = os.path.join(DATA_DIR, "error_log.txt")
API_KEYS_STATUS_FILE = os.path.join(DATA_DIR, "api_keys.json")
SETTINGS_FILE = os.path.join(DATA_DIR, "settings.json")
WATCHED_USERS_FILE = os.path.join(DATA_DIR, "watched_users.json")

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

DEFAULT_SETTINGS = {"daily_credits": 10, "new_user_bonus": 20, "referral_bonus": 10, "maintenance": False}

MODELS = {
    "chat": ["provider-3/gpt-4", "provider-3/gpt-4.1-mini", "provider-6/o4-mini-high", "provider-6/o4-mini-low", "provider-6/o3-high", "provider-6/o3-medium", "provider-6/o3-low", "provider-3/gpt-4o-mini-search-preview", "provider-6/gpt-4o", "provider-6/gpt-4.1-nano", "provider-6/gpt-4.1-mini", "provider-3/gpt-4.1-nano", "provider-6/gpt-4.1", "provider-6/o4-mini-medium", "provider-1/deepseek-v3-0324", "provider-6/minimax-m1-40k", "provider-6/kimi-k2", "provider-3/kimi-k2", "provider-6/qwen3-coder-480b-a35b", "provider-3/llama-3.1-405b", "provider-3/qwen-3-235b-a22b-2507", "provider-6/gemini-2.5-flash-thinking", "provider-6/gemini-2.5-flash", "provider-1/llama-3.1-405b-instruct-turbo", "provider-3/llama-3.1-70b", "provider-3/qwen-2.5-coder-32b", "provider-6/kimi-k2-instruct", "provider-6/r1-1776", "provider-6/deepseek-r1-uncensored", "provider-1/deepseek-r1-0528"],
    "image": ["provider-4/imagen-3", "provider-6/FLUX.1-kontext-max", "provider-6/FLUX.1-kontext-pro", "provider-6/FLUX.1-kontext-dev", "provider-3/FLUX.1-schnell", "provider-6/sana-1.5", "provider-3/FLUX.1-dev", "provider-6/FLUX.1-dev", "provider-1/FLUX.1.1-pro", "provider-6/FLUX.1-pro", "provider-1/FLUX.1-kontext-pro", "provider-1/FLUX.1-schnell", "provider-6/FLUX.1-1-pro", "provider-2/FLUX.1-schnell-v2", "provider-6/sana-1.5-flash"],
    "image_edit": ["provider-6/black-forest-labs-flux-1-kontext-max", "provider-6/black-forest-labs-flux-1-kontext-dev", "provider-6/black-forest-labs-flux-1-kontext-pro", "provider-3/flux-kontext-dev"],
    "video": ["provider-6/wan-2.1"],
    "tts": ["provider-3/tts-1"],
    "transcription": ["provider-3/whisper-1", "provider-6/distil-whisper-large-v3-en"]
}
MODELS_PER_PAGE = 5
TTS_VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

PERSONALITY_PRESETS = {
    "Friendly": "You are a friendly and helpful AI assistant. Always be warm, supportive, and encouraging in your responses.",
    "Professional": "You are a professional and formal AI assistant. Provide clear, concise, and business-like responses.",
    "Creative": "You are a creative and imaginative AI assistant. Think outside the box and provide innovative solutions.",
    "Humorous": "You are a witty and humorous AI assistant. Use humor and light-heartedness in your responses when appropriate.",
    "Educational": "You are an educational AI assistant. Focus on teaching, explaining concepts clearly, and providing learning opportunities.",
    "Technical": "You are a technical and precise AI assistant. Provide detailed, accurate, and technically sound responses."
}
IMAGE_SIZES = {"Square ⏹️": "1024x1024", "Wide  widescreen": "1792x1024", "Tall 📲": "1024x1792"}
VIDEO_RATIOS = {"Wide 🎬": "16:9", "Vertical 📱": "9:16", "Square 🖼️": "1:1"}
LOADING_MESSAGES = {"chat": "🤔 Cogitating on a thoughtful response...", "image": "🎨 Painting your masterpiece...", "image_edit": "🖌️ Applying artistic edits...", "video": "🎬 Directing your short film...", "tts": "🎙️ Warming up the vocal cords...", "transcription": "👂 Listening closely to your audio..."}
REASONING_MESSAGES = {"image": "⚙️ Reasoning about the visual elements...", "video": "🎥 Planning the scene and action..."}

AVAILABLE_MODELS_PROMPT_CONTEXT = """
You are a workflow design AI. Your task is to convert a user's natural language request into a structured JSON workflow.
The JSON output must be a single JSON object with a "steps" key containing a list of operations.
The output of one step can be used as input for the next using `prompt_from` or `input_from` which refers to the `output_variable` of a previous step.
If a workflow does not require user input to start, set `requires_input` to `false`. Otherwise, set it to `true`.
The initial user text input is always available as the variable `{initial_input}`. The initial user file input is available as `{initial_file_path}`.

Here are the available model types and their parameters:

1.  `"type": "chat"`
    - `model`: (string) Model name.
    - `prompt_template`: (string) The prompt to send. Use `{variable_name}` for placeholders.
    - `output_variable`: (string) The name to store the text output.
    - Available Models: "provider-3/gpt-4", "provider-3/gpt-4.1-mini", "provider-6/o4-mini-high", "provider-6/o4-mini-low", "provider-6/o3-high", "provider-6/o3-medium", "provider-6/o3-low", "provider-3/gpt-4o-mini-search-preview", "provider-6/gpt-4o", "provider-6/gpt-4.1-nano", "provider-6/gpt-4.1-mini", "provider-3/gpt-4.1-nano", "provider-6/o4-mini-medium", "provider-1/deepseek-v3-0324", "provider-6/minimax-m1-40k", "provider-6/kimi-k2", "provider-3/kimi-k2", "provider-6/qwen3-coder-480b-a35b", "provider-3/llama-3.1-405b", "provider-3/qwen-3-235b-a22b-2507", "provider-6/gemini-2.5-flash-thinking", "provider-6/gemini-2.5-flash", "provider-1/llama-3.1-405b-instruct-turbo", "provider-3/llama-3.1-70b", "provider-3/qwen-2.5-coder-32b", "provider-6/kimi-k2-instruct", "provider-6/r1-1776", "provider-6/deepseek-r1-uncensored", "provider-1/deepseek-r1-0528"

2.  `"type": "image_generation"`
    - `output_variable`: (string) The name to store the output image URL.
    - `prompt_from`: (string) Variable containing the prompt text.
    - **Standard Models (Support `n` and `size`):**
        - `model`: (string) One of: "provider-4/imagen-3", "provider-3/FLUX.1-schnell", "provider-1/FLUX.1-schnell", "provider-2/FLUX.1-schnell-v2", "provider-6/sana-1.5", "provider-6/sana-1.5-flash"
        - `n`: (integer, optional) Number of images.
        - `size`: (string, optional) e.g., "1024x1024", "1792x1024", "1024x1792".
    - **Advanced FLUX Models (Do NOT use `n` parameter):**
        - `model`: (string) One of: "provider-6/FLUX.1-kontext-max", "provider-6/FLUX.1-kontext-pro", "provider-6/FLUX.1-kontext-dev", "provider-3/FLUX.1-dev", "provider-6/FLUX.1-dev", "provider-1/FLUX.1.1-pro", "provider-6/FLUX.1-pro", "provider-1/FLUX.1-kontext-pro", "provider-6/FLUX.1-1-pro"
        - `size`: (string, optional) e.g., "1024x1024", "1792x1024", "1024x1792".

3.  `"type": "image_edit"`
    - `model`: (string) Model name.
    - `prompt_from`: (string) Variable containing the edit instruction.
    - `image_from`: (string) Variable containing the URL of the image to edit.
    - `output_variable`: (string) The name to store the edited image URL.
    - Available Models: "provider-6/black-forest-labs-flux-1-kontext-pro", "provider-3/flux-kontext-dev", "provider-6/black-forest-labs-flux-1-kontext-max"

4.  `"type": "video_generation"`
    - `model`: (string) Model name.
    - `prompt_from`: (string) Variable containing the prompt text.
    - `ratio`: (string) e.g., "9:16", "16:9".
    - `duration`: (integer, optional, default: 4) Max 4.
    - `quality`: (string, optional, default: "480p") Max "480p".
    - `output_variable`: (string) The name to store the output video URL.
    - Available Models: "provider-6/wan-2.1"

5.  `"type": "tts"`
    - `model`: (string) Model name.
    - `input_from`: (string) Variable containing the text to convert to speech.
    - `voice`: (string) One of: alloy, echo, fable, onyx, nova, shimmer.
    - `output_variable`: (string) The name to store the output audio file path.
    - Available Models: "provider-3/tts-1", "provider-6/sonic-2", "provider-6/sonic"

6.  `"type": "transcription"`
    - `model`: (string) Model name.
    - `file_path_from`: (string) Variable containing the path of the audio file to transcribe (e.g., `{initial_file_path}`).
    - `output_variable`: (string) The name to store the transcribed text.
    - Available Models: "provider-2/whisper-1", "provider-6/distil-whisper-large-v3-en"

7.  `"type": "custom_api"`
    - `url`: (string) The endpoint URL.
    - `method`: (string) "POST", "GET", etc.
    - `headers`: (dict) e.g., {"Authorization": "Bearer {api_key_variable}", "Content-Type": "application/json"}.
    - `body_template`: (dict/string) The request body as a JSON template.
    - `output_variable`: (string) The name to store the result.
    - `output_parser_path`: (string, optional) e.g., "choices.0.message.content".
"""

(USER_MAIN, SELECTING_MODEL, AWAITING_PROMPT, AWAITING_TTS_INPUT, AWAITING_AUDIO, AWAITING_IMAGE_FOR_EDIT, AWAITING_EDIT_PROMPT, AWAITING_TTS_VOICE, AWAITING_VIDEO_RATIO, AWAITING_PERSONALITY, AWAITING_BROADCAST_CONFIRMATION, AWAITING_IMAGE_SIZE, SELECTING_PRESET_PERSONALITY, ADMIN_MAIN, ADMIN_AWAITING_INPUT, SELECTING_VOICE_FOR_MODE, AWAITING_VOICE_MODE_INPUT, AWAITING_MIXER_CONCEPT_1, AWAITING_MIXER_CONCEPT_2, AWAITING_WEB_PROMPT, SELECTING_VOICE_MODEL_CHOICE, AWAITING_VOICE_MODE_PRO_INPUT, GET_WORKFLOW_DESCRIPTION, GET_WORKFLOW_NAME, CONFIRM_WORKFLOW, WORKFLOW_PRIVACY_SETTINGS, WORKFLOW_PRICING_SETTINGS, EDIT_WORKFLOW_DESCRIPTION, MARKETPLACE_BROWSE, AWAITING_AI_EDIT_INSTRUCTIONS) = range(30)

_active_api_keys, _settings, _watched_users = [], {}, set()

# MongoDB connection
mongo_client = None
db = None

def init_mongodb():
    """Initialize MongoDB connection"""
    global mongo_client, db
    try:
        # Add timeout parameters to prevent hanging
        mongo_client = MongoClient(
            MONGODB_URL,
            serverSelectionTimeoutMS=10000,  # 10 seconds timeout
            connectTimeoutMS=10000,
            socketTimeoutMS=10000,
            maxPoolSize=10,
            retryWrites=True
        )
        
        # Test the connection with timeout
        mongo_client.admin.command('ping', serverSelectionTimeoutMS=5000)
        db = mongo_client.telegram_bot
        
        # Create indexes with timeout handling
        try:
            db.users.create_index("user_id", unique=True, background=True)
            logger.info("Created users index")
        except Exception as e:
            logger.warning(f"Could not create users index: {e}")
            
        try:
            db.workflows.create_index([("user_id", 1), ("name", 1)], unique=True, background=True)
            logger.info("Created workflows index")
        except Exception as e:
            logger.warning(f"Could not create workflows index: {e}")
            
        try:
            db.api_keys.create_index("key", unique=True, background=True)
            logger.info("Created api_keys index")
        except Exception as e:
            logger.warning(f"Could not create api_keys index: {e}")
            
        try:
            db.marketplace.create_index("workflow_id", unique=True, background=True)
            logger.info("Created marketplace index")
        except Exception as e:
            logger.warning(f"Could not create marketplace index: {e}")
            
        try:
            db.redeem_codes.create_index("code", unique=True, background=True)
            logger.info("Created redeem_codes index")
        except Exception as e:
            logger.warning(f"Could not create redeem_codes index: {e}")
        
        logger.info("MongoDB connected successfully")
        return True
    except ConnectionFailure as e:
        logger.error(f"MongoDB connection failed: {e}")
        return False
    except Exception as e:
        logger.error(f"MongoDB initialization error: {e}")
        return False

def load_settings():
    global _settings
    try:
        if db is not None:
            settings_doc = db.settings.find_one({"_id": "bot_settings"})
            if settings_doc:
                _settings = settings_doc.get("data", DEFAULT_SETTINGS.copy())
            else:
                _settings = DEFAULT_SETTINGS.copy()
        else:
            # Fallback to file system if MongoDB is not available
            if os.path.exists(SETTINGS_FILE):
                with open(SETTINGS_FILE, 'r') as f: _settings = json.load(f)
            else: _settings = DEFAULT_SETTINGS.copy()
        
        for key, value in DEFAULT_SETTINGS.items(): _settings.setdefault(key, value)
        save_settings()
    except Exception as e:
        logger.error(f"Error loading settings: {e}")
        _settings = DEFAULT_SETTINGS.copy()

def save_settings():
    try:
        if db is not None:
            db.settings.replace_one(
                {"_id": "bot_settings"},
                {"_id": "bot_settings", "data": _settings},
                upsert=True
            )
        else:
            # Fallback to file system
            with open(SETTINGS_FILE, 'w') as f: json.dump(_settings, f, indent=4)
    except Exception as e:
        logger.error(f"Error saving settings: {e}")
        # Fallback to file system
        with open(SETTINGS_FILE, 'w') as f: json.dump(_settings, f, indent=4)

def load_watched_users():
    global _watched_users
    try:
        if db is not None:
            watched_doc = db.settings.find_one({"_id": "watched_users"})
            if watched_doc:
                _watched_users = set(watched_doc.get("data", []))
            else:
                _watched_users = set()
        else:
            # Fallback to file system
            if os.path.exists(WATCHED_USERS_FILE):
                with open(WATCHED_USERS_FILE, 'r') as f:
                    _watched_users = set(json.load(f))
            else:
                _watched_users = set()
    except Exception as e:
        logger.error(f"Error loading watched users: {e}")
        _watched_users = set()

def save_watched_users():
    try:
        if db is not None:
            db.settings.replace_one(
                {"_id": "watched_users"},
                {"_id": "watched_users", "data": list(_watched_users)},
                upsert=True
            )
        else:
            # Fallback to file system
            with open(WATCHED_USERS_FILE, 'w') as f:
                json.dump(list(_watched_users), f)
    except Exception as e:
        logger.error(f"Error saving watched users: {e}")
        # Fallback to file system
        with open(WATCHED_USERS_FILE, 'w') as f:
            json.dump(list(_watched_users), f)

def load_api_keys():
    global _active_api_keys
    try:
        if db is not None:
            keys_doc = db.settings.find_one({"_id": "api_keys"})
            if keys_doc:
                all_keys_status = keys_doc.get("data", [])
            else:
                all_keys_status = [{"key": k, "active": True} for k in INITIAL_A4F_KEYS]
                save_api_keys(all_keys_status)
        else:
            # Fallback to file system
            if not os.path.exists(API_KEYS_STATUS_FILE):
                all_keys_status = [{"key": k, "active": True} for k in INITIAL_A4F_KEYS]
                save_api_keys(all_keys_status)
            else:
                with open(API_KEYS_STATUS_FILE, 'r') as f:
                    all_keys_status = json.load(f)
        
        _active_api_keys = [k['key'] for k in all_keys_status if k.get('active', True)]
        return all_keys_status
    except Exception as e:
        logger.error(f"Error loading API keys: {e}")
        all_keys_status = [{"key": k, "active": True} for k in INITIAL_A4F_KEYS]
        _active_api_keys = [k['key'] for k in all_keys_status]
        return all_keys_status

def save_api_keys(keys_status):
    try:
        if db is not None:
            db.settings.replace_one(
                {"_id": "api_keys"},
                {"_id": "api_keys", "data": keys_status},
                upsert=True
            )
        else:
            # Fallback to file system
            with open(API_KEYS_STATUS_FILE, 'w') as f:
                json.dump(keys_status, f, indent=4)
        load_api_keys()
    except Exception as e:
        logger.error(f"Error saving API keys: {e}")
        # Fallback to file system
        with open(API_KEYS_STATUS_FILE, 'w') as f:
            json.dump(keys_status, f, indent=4)
        load_api_keys()

def get_random_api_key():
    return random.choice(_active_api_keys) if _active_api_keys else None

def is_admin(user_id: int) -> bool:
    return user_id == ADMIN_CHAT_ID

async def make_api_request_with_retry(client, url, headers, json_data=None, files=None, data=None, timeout=120, max_retries=3):
    """Make an API request with silent retry logic for handling temporary server issues."""
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            logger.info(f"=== API REQUEST ATTEMPT {attempt + 1}/{max_retries} ===")
            logger.info(f"URL: {url}")
            logger.info(f"Method: POST")
            logger.info(f"Headers: {headers}")
            if json_data:
                logger.info(f"JSON Data: {json_data}")
                response = await client.post(url, headers=headers, json=json_data, timeout=timeout)
            elif files:
                logger.info(f"Files: {list(files.keys())}")
                logger.info(f"Data: {data}")
                response = await client.post(url, headers=headers, data=data, files=files, timeout=timeout)
            else:
                logger.info(f"Data: {data}")
                response = await client.post(url, headers=headers, data=data, timeout=timeout)
            
            logger.info(f"Response Status: {response.status_code}")
            logger.info(f"Response Headers: {dict(response.headers)}")
            
            response.raise_for_status()
            return response
            
        except httpx.HTTPStatusError as e:
            last_exception = e
            logger.error(f"HTTP Status Error: {e.response.status_code}")
            logger.error(f"Response Headers: {dict(e.response.headers)}")
            try:
                error_body = e.response.json()
                logger.error(f"Response Body: {error_body}")
            except:
                logger.error(f"Response Text: {e.response.text}")
            
            # Silent retry for 500 errors and other server errors
            if e.response.status_code >= 500 and attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                logger.debug(f"Silent retry {attempt + 1}/{max_retries} for {url} due to {e.response.status_code}")
                continue
            elif e.response.status_code == 429 and attempt < max_retries - 1:  # Rate limit
                await asyncio.sleep(2 ** attempt)
                logger.debug(f"Silent retry {attempt + 1}/{max_retries} for {url} due to rate limit")
                continue
            else:
                raise e
        except httpx.RequestError as e:
            last_exception = e
            logger.error(f"Request Error: {str(e)}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                logger.debug(f"Silent retry {attempt + 1}/{max_retries} for {url} due to connection error")
                continue
            else:
                raise e
        except Exception as e:
            last_exception = e
            logger.error(f"Unexpected Error: {str(e)}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
                logger.debug(f"Silent retry {attempt + 1}/{max_retries} for {url} due to unexpected error")
                continue
            else:
                raise e
    
    # If all retries failed, raise the last exception
    if last_exception:
        raise last_exception
    else:
        raise httpx.RequestError("All retry attempts failed")

def setup_data_directory():
    # Initialize MongoDB first with timeout protection
    try:
        mongo_connected = init_mongodb()
        if mongo_connected:
            logger.info("Using MongoDB for data storage")
        else:
            logger.warning("MongoDB connection failed, falling back to file system")
    except Exception as e:
        logger.error(f"MongoDB initialization failed: {e}")
        mongo_connected = False
    
    if not mongo_connected:
        # Create directories for file system fallback
        try:
            for path in [DATA_DIR, USERS_DIR, TEMP_DIR, WORKFLOWS_DIR, MARKETPLACE_DIR]:
                if not os.path.exists(path): 
                    os.makedirs(path)
            if not os.path.exists(REDEEM_CODES_FILE):
                with open(REDEEM_CODES_FILE, 'w') as f: 
                    json.dump({}, f)
            logger.info("File system fallback initialized")
        except Exception as e:
            logger.error(f"Error setting up file system fallback: {e}")
    
    # Load data with error handling
    try:
        load_api_keys()
    except Exception as e:
        logger.error(f"Error loading API keys: {e}")
    
    try:
        load_settings()
    except Exception as e:
        logger.error(f"Error loading settings: {e}")
    
    try:
        load_watched_users()
    except Exception as e:
        logger.error(f"Error loading watched users: {e}")

def load_user_data(user_id):
    today = date.today().isoformat()
    try:
        if db is not None:
            user_data = db.users.find_one({"user_id": user_id})
            if not user_data:
                user_data = {"user_id": user_id, "credits": _settings["daily_credits"] + _settings["new_user_bonus"], "last_login_date": today, "is_new": True, "personality": None, "banned": False, "referral_code": f"REF-{uuid.uuid4().hex[:8].upper()}", "referred_by": None, "referrals_made": 0, "stats": {k: 0 for k in LOADING_MESSAGES.keys()}}
                save_user_data(user_id, user_data)
                return user_data
        else:
            # Fallback to file system
            user_file = os.path.join(USERS_DIR, f"{user_id}.json")
            if not os.path.exists(user_file):
                user_data = {"user_id": user_id, "credits": _settings["daily_credits"] + _settings["new_user_bonus"], "last_login_date": today, "is_new": True, "personality": None, "banned": False, "referral_code": f"REF-{uuid.uuid4().hex[:8].upper()}", "referred_by": None, "referrals_made": 0, "stats": {k: 0 for k in LOADING_MESSAGES.keys()}}
                save_user_data(user_id, user_data)
                return user_data
            try:
                with open(user_file, 'r') as f: user_data = json.load(f)
                user_data["user_id"] = user_id  # Ensure user_id is set
            except (json.JSONDecodeError, FileNotFoundError):
                user_data = {"user_id": user_id, "credits": 0, "last_login_date": "1970-01-01", "is_new": True, "personality": None, "banned": False}
        
        if user_data.get("last_login_date") != today and user_id != ADMIN_CHAT_ID:
            user_data["credits"] = user_data.get("credits", 0) + _settings["daily_credits"]
            user_data["last_login_date"] = today

        defaults = {"is_new": False, "personality": None, "banned": False, "referral_code": f"REF-{uuid.uuid4().hex[:8].upper()}", "referred_by": None, "referrals_made": 0, "stats": {k: 0 for k in LOADING_MESSAGES.keys()}}
        for key, value in defaults.items(): user_data.setdefault(key, value)
        if "stats" not in user_data or not isinstance(user_data["stats"], dict): user_data["stats"] = {k: 0 for k in LOADING_MESSAGES.keys()}

        save_user_data(user_id, user_data)
        return user_data
    except Exception as e:
        logger.error(f"Error loading user data for {user_id}: {e}")
        user_data = {"user_id": user_id, "credits": _settings["daily_credits"], "last_login_date": today, "is_new": True, "personality": None, "banned": False, "referral_code": f"REF-{uuid.uuid4().hex[:8].upper()}", "referred_by": None, "referrals_made": 0, "stats": {k: 0 for k in LOADING_MESSAGES.keys()}}
        return user_data

def save_user_data(user_id, data):
    try:
        data["user_id"] = user_id  # Ensure user_id is always set
        if db is not None:
            db.users.replace_one(
                {"user_id": user_id},
                data,
                upsert=True
            )
        else:
            # Fallback to file system
            with open(os.path.join(USERS_DIR, f"{user_id}.json"), 'w') as f: 
                json.dump(data, f, indent=4)
    except Exception as e:
        logger.error(f"Error saving user data for {user_id}: {e}")
        # Fallback to file system
        with open(os.path.join(USERS_DIR, f"{user_id}.json"), 'w') as f: 
            json.dump(data, f, indent=4)

def load_redeem_codes():
    try:
        if db is not None:
            codes_doc = db.settings.find_one({"_id": "redeem_codes"})
            if codes_doc:
                return codes_doc.get("data", {})
            else:
                return {}
        else:
            # Fallback to file system
            with open(REDEEM_CODES_FILE, 'r') as f: 
                return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {}
    except Exception as e:
        logger.error(f"Error loading redeem codes: {e}")
        return {}

def save_redeem_codes(codes):
    try:
        if db is not None:
            db.settings.replace_one(
                {"_id": "redeem_codes"},
                {"_id": "redeem_codes", "data": codes},
                upsert=True
            )
        else:
            # Fallback to file system
            with open(REDEEM_CODES_FILE, 'w') as f: 
                json.dump(codes, f, indent=4)
    except Exception as e:
        logger.error(f"Error saving redeem codes: {e}")
        # Fallback to file system
        with open(REDEEM_CODES_FILE, 'w') as f: 
            json.dump(codes, f, indent=4)

def load_workflows(user_id):
    """Load workflows for a specific user."""
    try:
        if db is not None:
            workflows_doc = db.workflows.find_one({"user_id": user_id})
            if workflows_doc:
                workflows = workflows_doc.get("workflows", {})
                logger.info(f"Loaded {len(workflows)} workflows from MongoDB for user {user_id}")
                return workflows
            else:
                logger.info(f"No workflows found in MongoDB for user {user_id}")
                return {}
        else:
            # Fallback to file system
            user_workflow_file = os.path.join(WORKFLOWS_DIR, f"user_{user_id}_workflows.json")
            if os.path.exists(user_workflow_file):
                try:
                    with open(user_workflow_file, 'r') as f:
                        workflows = json.load(f)
                        logger.info(f"Loaded {len(workflows)} workflows from file for user {user_id}")
                        return workflows
                except (json.JSONDecodeError, FileNotFoundError):
                    logger.warning(f"Error reading workflow file for user {user_id}")
                    return {}
            logger.info(f"No workflow file found for user {user_id}")
            return {}
    except Exception as e:
        logger.error(f"Error loading workflows for user {user_id}: {e}")
        return {}

def save_workflows(user_id, workflows):
    """Save workflows for a specific user."""
    try:
        if db is not None:
            db.workflows.replace_one(
                {"user_id": user_id},
                {"user_id": user_id, "workflows": workflows},
                upsert=True
            )
            logger.info(f"Saved {len(workflows)} workflows to MongoDB for user {user_id}")
        else:
            # Fallback to file system
            user_workflow_file = os.path.join(WORKFLOWS_DIR, f"user_{user_id}_workflows.json")
            with open(user_workflow_file, 'w') as f:
                json.dump(workflows, f, indent=4)
            logger.info(f"Saved {len(workflows)} workflows to file for user {user_id}")
    except Exception as e:
        logger.error(f"Error saving workflows for user {user_id}: {e}")
        # Fallback to file system
        user_workflow_file = os.path.join(WORKFLOWS_DIR, f"user_{user_id}_workflows.json")
        with open(user_workflow_file, 'w') as f:
            json.dump(workflows, f, indent=4)
        logger.info(f"Saved {len(workflows)} workflows to file (fallback) for user {user_id}")

def substitute_variables(template, variables):
    """Substitute variables in templates for workflow execution."""
    if isinstance(template, str):
        for key, value in variables.items():
            template = template.replace(f"{{{key}}}", str(value))
        return template
    elif isinstance(template, dict):
        return {k: substitute_variables(v, variables) for k, v in template.items()}
    elif isinstance(template, list):
        return [substitute_variables(i, variables) for i in template]
    return template

def create_workflow_metadata(name, workflow_data, creator_id, is_public=False, price=0):
    """Create a workflow with metadata."""
    return {
        "name": name,
        "workflow": workflow_data,
        "creator_id": creator_id,
        "created_at": datetime.now().isoformat(),
        "is_public": is_public,
        "price": price,  # 0 = free, >0 = paid
        "downloads": 0,
        "rating": 0.0,
        "ratings_count": 0
    }

def calculate_workflow_credits(workflow):
    """Calculate credits needed to run a workflow based on steps."""
    credits = 0
    for step in workflow.get("steps", []):
        step_type = step.get("type")
        # External API calls don't require credits
        if step_type == "custom_api":
            continue
        # All other step types require 1 credit each
        elif step_type in ["chat", "image_generation", "image_edit", "video_generation", "tts", "transcription"]:
            credits += 1
    return max(credits, 1)  # Minimum 1 credit per workflow run

def load_marketplace_workflows():
    """Load all public workflows from marketplace."""
    try:
        if db is not None:
            marketplace_doc = db.marketplace.find_one({"_id": "public_workflows"})
            if marketplace_doc:
                return marketplace_doc.get("workflows", {})
            else:
                return {}
        else:
            # Fallback to file system
            marketplace_file = os.path.join(MARKETPLACE_DIR, "public_workflows.json")
            if os.path.exists(marketplace_file):
                try:
                    with open(marketplace_file, 'r') as f:
                        return json.load(f)
                except (json.JSONDecodeError, FileNotFoundError):
                    return {}
            return {}
    except Exception as e:
        logger.error(f"Error loading marketplace workflows: {e}")
        return {}

def save_marketplace_workflows(workflows):
    """Save public workflows to marketplace."""
    try:
        if db is not None:
            db.marketplace.replace_one(
                {"_id": "public_workflows"},
                {"_id": "public_workflows", "workflows": workflows},
                upsert=True
            )
        else:
            # Fallback to file system
            marketplace_file = os.path.join(MARKETPLACE_DIR, "public_workflows.json")
            with open(marketplace_file, 'w') as f:
                json.dump(workflows, f, indent=4)
    except Exception as e:
        logger.error(f"Error saving marketplace workflows: {e}")
        # Fallback to file system
        marketplace_file = os.path.join(MARKETPLACE_DIR, "public_workflows.json")
        with open(marketplace_file, 'w') as f:
            json.dump(workflows, f, indent=4)

def add_workflow_to_marketplace(workflow_id, workflow_metadata):
    """Add a workflow to the public marketplace."""
    marketplace_workflows = load_marketplace_workflows()
    marketplace_workflows[workflow_id] = workflow_metadata
    save_marketplace_workflows(marketplace_workflows)

def remove_workflow_from_marketplace(workflow_id):
    """Remove a workflow from the public marketplace."""
    marketplace_workflows = load_marketplace_workflows()
    if workflow_id in marketplace_workflows:
        del marketplace_workflows[workflow_id]
        save_marketplace_workflows(marketplace_workflows)

def load_user_owned_workflows(user_id):
    """Load workflows that the user has purchased or downloaded."""
    try:
        if db is not None:
            owned_doc = db.user_owned.find_one({"user_id": user_id})
            if owned_doc:
                return owned_doc.get("owned_workflows", [])
            else:
                return []
        else:
            # Fallback to file system
            user_owned_file = os.path.join(WORKFLOWS_DIR, f"user_{user_id}_owned.json")
            if os.path.exists(user_owned_file):
                try:
                    with open(user_owned_file, 'r') as f:
                        return json.load(f)
                except (json.JSONDecodeError, FileNotFoundError):
                    return []
            return []
    except Exception as e:
        logger.error(f"Error loading owned workflows for user {user_id}: {e}")
        return []

def save_user_owned_workflows(user_id, owned_workflows):
    """Save workflows that the user owns/has downloaded."""
    try:
        if db is not None:
            db.user_owned.replace_one(
                {"user_id": user_id},
                {"user_id": user_id, "owned_workflows": owned_workflows},
                upsert=True
            )
        else:
            # Fallback to file system
            user_owned_file = os.path.join(WORKFLOWS_DIR, f"user_{user_id}_owned.json")
            with open(user_owned_file, 'w') as f:
                json.dump(owned_workflows, f, indent=4)
    except Exception as e:
        logger.error(f"Error saving owned workflows for user {user_id}: {e}")
        # Fallback to file system
        user_owned_file = os.path.join(WORKFLOWS_DIR, f"user_{user_id}_owned.json")
        with open(user_owned_file, 'w') as f:
            json.dump(owned_workflows, f, indent=4)

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

def use_multiple_credits(user_id: int, count: int) -> bool:
    """Use multiple credits at once. Returns True if successful, False if insufficient credits."""
    if is_admin(user_id): 
        return True
    if _settings.get('maintenance', False): 
        return False
    user_data = load_user_data(user_id)
    if user_data.get("banned", False): 
        return False
    if user_data.get("credits", 0) >= count:
        user_data["credits"] -= count
        save_user_data(user_id, user_data)
        return True
    return False

def refund_multiple_credits(user_id: int, count: int):
    """Refund multiple credits to a user."""
    if user_id != ADMIN_CHAT_ID:
        user_data = load_user_data(user_id)
        user_data["credits"] = user_data.get("credits", 0) + count
        save_user_data(user_id, user_data)
        logger.info(f"Refunded {count} credits to user {user_id}")

async def handle_api_error_with_retry_and_refund(user_id: int, error: Exception, credit_cost: int = 1, context=None, voice_mode=False):
    """Comprehensive error handler that logs, refunds credits, and provides user feedback."""
    error_type = type(error).__name__
    error_msg = str(error)
    
    # Log the error
    logger.error(f"API Error for user {user_id}: {error_type} - {error_msg}", exc_info=True)
    
    # Refund credits
    if credit_cost == 1:
        refund_credit(user_id)
    else:
        refund_multiple_credits(user_id, credit_cost)
    
    # Determine user-friendly message based on error type
    if isinstance(error, httpx.HTTPStatusError):
        if error.response.status_code >= 500:
            user_message = "The service is currently experiencing issues. Please try again in a few minutes."
        elif error.response.status_code == 429:
            user_message = "The service is currently busy. Please try again in a moment."
        elif error.response.status_code == 401:
            user_message = "Authentication error. Please contact support."
        elif error.response.status_code == 400:
            user_message = "Invalid request. Please try again with different parameters."
        else:
            user_message = f"Service error (Status: {error.response.status_code}). Please try again."
    elif isinstance(error, httpx.RequestError):
        user_message = "Connection error. Please check your internet connection and try again."
    elif isinstance(error, httpx.TimeoutException):
        user_message = "Request timed out. Please try again."
    else:
        user_message = "An unexpected error occurred. Please try again."
    
    # Provide user feedback
    if voice_mode and context:
        await _speak(context, user_id, user_message, voice=context.user_data.get('voice_mode_voice'))
    elif context:
        try:
            await context.message.reply_text(f"❌ {user_message}")
        except:
            pass
    
    return user_message

async def make_api_request_for_workflow(method, url, headers=None, json_data=None, data=None, files=None):
    """Make an API request for workflow operations using httpx for consistency."""
    try:
        # Increase timeout for A4F API calls
        timeout = 300 if "api.a4f.co" in url else 180
        async with httpx.AsyncClient() as client:
            if json_data:
                response = await client.post(url, headers=headers, json=json_data, timeout=timeout)
            elif files:
                response = await client.post(url, headers=headers, data=data, files=files, timeout=timeout)
            else:
                response = await client.post(url, headers=headers, data=data, timeout=timeout)
            
            response.raise_for_status()
            
            # Handle different content types properly
            content_type = response.headers.get('Content-Type', '')
            if 'application/json' in content_type:
                return response.json()
            elif 'audio' in content_type or 'image' in content_type or 'video' in content_type:
                return response.content  # Return bytes for binary content
            else:
                # For text content, return as string
                return response.text
    except httpx.RequestException as e:
        logger.error(f"API Request failed: {e}")
        error_details = e.response.text if e.response else "No response from server."
        return {"error": str(e), "details": error_details}

async def design_workflow(user_description: str) -> dict:
    """Design a workflow using AI based on user description."""
    headers = {"Authorization": f"Bearer {get_random_api_key()}", "Content-Type": "application/json"}
    data = {
        "model": WORKFLOW_DESIGNER_MODEL,
        "messages": [{"role": "system", "content": AVAILABLE_MODELS_PROMPT_CONTEXT}, {"role": "user", "content": user_description}],
        "temperature": 0.1, "max_tokens": 2048,
    }
    
    # Use the same retry logic as the direct tools
    async with httpx.AsyncClient() as client:
        response = await make_api_request_with_retry(
            client, 
            f"{A4F_API_BASE_URL}/chat/completions", 
            headers=headers, 
            json_data=data,
            timeout=120
        )
        response_data = response.json()
    
    if "error" in response_data: 
        return response_data
    try:
        content = response_data['choices'][0]['message']['content']
        match = re.search(r'```json\\s*(.*?)\\s*```', content, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            json_str = content[content.find('{'):content.rfind('}')+1] or content[content.find('['):content.rfind(']')+1]
        
        parsed_json = json.loads(json_str)
        if isinstance(parsed_json, list):
            return {"requires_input": True, "steps": parsed_json}
        return parsed_json
    except Exception as e:
        logger.error(f"Failed to parse workflow from AI response: {e}\nResponse was: {content}")
        return {"error": "Could not design a valid workflow. The AI's response was malformed."}

async def execute_workflow(workflow: dict, initial_input: str, chat_id: int, context: ContextTypes.DEFAULT_TYPE, initial_file_path: str = None):
    """Execute a complete workflow with multiple steps."""
    variables = {"initial_input": initial_input, "initial_file_path": initial_file_path}
    if "api_keys" in workflow:
        variables.update(workflow["api_keys"])
    temp_files = []
    try:
        for i, step in enumerate(workflow["steps"]):
            step_type = step.get("type")
            if not step_type:
                raise ValueError(f"Step {i+1} is missing the 'type' attribute.")
            
            logger.info(f"=== PROCESSING STEP {i+1} ===")
            logger.info(f"Step type: {step_type}")
            logger.info(f"Step data: {step}")
            
            await context.bot.send_message(chat_id, f"⚙️ Step {i+1}/{len(workflow['steps'])}: Running `{escape_markdown_v2(step_type)}`\\.\\.\\.", parse_mode=ParseMode.MARKDOWN_V2)
            output_variable = step.get("output_variable")
            output = None

            if step_type == "chat":
                logger.info(f"=== ENTERING CHAT STEP ===")
                logger.info(f"Step {i+1}: {step}")
                model = step.get("model")
                if not model: 
                    raise ValueError(f"Chat step {i+1} is missing 'model'.")
                prompt_template = step.get("prompt_template")
                if not prompt_template: 
                    raise ValueError(f"Chat step {i+1} is missing 'prompt_template'.")
                prompt = substitute_variables(prompt_template, variables)
                
                logger.info(f"=== CHAT STEP DEBUG ===")
                logger.info(f"Model: {model}")
                logger.info(f"Prompt template: {prompt_template}")
                logger.info(f"Substituted prompt: {prompt}")
                logger.info(f"Variables: {variables}")
                
                data = {"model": model, "messages": [{"role": "user", "content": prompt}]}
                headers = {"Authorization": f"Bearer {get_random_api_key()}", "Content-Type": "application/json"}
                
                # Use the same retry logic as the direct tools
                async with httpx.AsyncClient() as client:
                    response = await make_api_request_with_retry(
                        client, 
                        f"{A4F_API_BASE_URL}/chat/completions", 
                        headers=headers, 
                        json_data=data,
                        timeout=120
                    )
                    result = response.json()
                output = result['choices'][0]['message']['content']
                
                logger.info(f"Chat response: {output}")
                logger.info(f"Output variable: {output_variable}")
            
            elif step_type == "image_generation":
                model = step.get("model")
                if not model: 
                    raise ValueError(f"Image generation step {i+1} is missing 'model'.")
                prompt_from = step.get("prompt_from")
                if not prompt_from: 
                    raise ValueError("Image generation step is missing 'prompt_from'.")
                prompt = variables.get(prompt_from)
                if not prompt: 
                    raise Exception(f"Input for image generation (from variable '{prompt_from}') is empty or missing.")
                data = {"model": model, "prompt": prompt}
                if "n" in step: 
                    data["n"] = step["n"]
                if "size" in step: 
                    data["size"] = step["size"]
                headers = {"Authorization": f"Bearer {get_random_api_key()}", "Content-Type": "application/json"}
                
                # Use the same retry logic as the direct tools
                async with httpx.AsyncClient() as client:
                    response = await make_api_request_with_retry(
                        client, 
                        f"{A4F_API_BASE_URL}/images/generations", 
                        headers=headers, 
                        json_data=data,
                        timeout=180
                    )
                    result = response.json()
                output = result['data'][0]['url']

            elif step_type == "image_edit":
                model = step.get("model")
                if not model: 
                    raise ValueError(f"Image edit step {i+1} is missing 'model'.")
                prompt_from = step.get("prompt_from")
                if not prompt_from: 
                    raise ValueError("Image edit step is missing 'prompt_from'.")
                prompt = variables.get(prompt_from)
                logger.info(f"=== IMAGE EDIT STEP DEBUG ===")
                logger.info(f"Prompt from variable: {prompt_from}")
                logger.info(f"All variables: {variables}")
                logger.info(f"Retrieved prompt: {prompt}")
                if not prompt: 
                    raise Exception(f"Prompt for image edit (from variable '{prompt_from}') is empty or missing.")
                image_from = step.get("image_from")
                if not image_from: 
                    raise ValueError("Image edit step is missing 'image_from'.")
                
                # Handle case where image_from might be a literal string with braces
                if image_from.startswith("{") and image_from.endswith("}"):
                    # Extract the variable name from braces
                    actual_variable = image_from[1:-1]
                    logger.info(f"Extracted image variable name: {actual_variable}")
                    image_path = variables.get(actual_variable)
                else:
                    image_path = variables.get(image_from)
                
                if not image_path: 
                    raise Exception(f"Image path for image edit (from variable '{image_from}') is empty or missing.")
                
                # Check if it's a local file path or URL
                if os.path.exists(image_path):
                    # Local file - read directly
                    with open(image_path, 'rb') as f:
                        image_content = f.read()
                    headers = {"Authorization": f"Bearer {get_random_api_key()}"}
                    data = {"model": model, "prompt": prompt}
                    files = {"image": ("image.jpg", image_content, "image/jpeg")}
                else:
                    # URL - download first
                    image_response = requests.get(image_path)
                    image_response.raise_for_status()
                    headers = {"Authorization": f"Bearer {get_random_api_key()}"}
                    data = {"model": model, "prompt": prompt}
                    files = {"image": ("image.png", image_response.content, "image/png")}
                
                # Use the same retry logic as the direct tools
                async with httpx.AsyncClient() as client:
                    response = await make_api_request_with_retry(
                        client, 
                        f"{A4F_API_BASE_URL}/images/edits", 
                        headers=headers, 
                        data=data, 
                        files=files,
                        timeout=180
                    )
                    result = response.json()
                output = result['data'][0]['url']

            elif step_type == "video_generation":
                model = step.get("model")
                if not model: 
                    raise ValueError(f"Video generation step {i+1} is missing 'model'.")
                prompt_from = step.get("prompt_from")
                if not prompt_from: 
                    raise ValueError("Video generation step is missing 'prompt_from'.")
                prompt = variables.get(prompt_from)
                if not prompt: 
                    raise Exception(f"Input for video generation (from variable '{prompt_from}') is empty or missing.")
                ratio = step.get("ratio")
                if not ratio: 
                    raise ValueError("Video generation step is missing required 'ratio' parameter.")
                data = {"model": model, "prompt": prompt, "ratio": ratio}
                # Add required fields with defaults if not specified
                data["duration"] = step.get("duration", 4)  # Default 4 seconds
                data["quality"] = step.get("quality", "480p")  # Default 480p
                headers = {"Authorization": f"Bearer {get_random_api_key()}", "Content-Type": "application/json"}
                
                # Log the request details
                logger.info(f"=== VIDEO GENERATION REQUEST ===")
                logger.info(f"URL: {A4F_API_BASE_URL}/video/generations")
                logger.info(f"Headers: {headers}")
                logger.info(f"Data: {data}")
                logger.info(f"Step config: {step}")
                
                # Use the same retry logic as the direct video tool
                async with httpx.AsyncClient() as client:
                    try:
                        response = await make_api_request_with_retry(
                            client, 
                            f"{A4F_API_BASE_URL}/video/generations", 
                            headers=headers, 
                            json_data=data,
                            timeout=300  # Longer timeout for video generation
                        )
                        result = response.json()
                        logger.info(f"=== VIDEO GENERATION RESPONSE ===")
                        logger.info(f"Status Code: {response.status_code}")
                        logger.info(f"Response Headers: {dict(response.headers)}")
                        logger.info(f"Response Body: {result}")
                        output = result['data'][0]['url']
                    except Exception as e:
                        logger.error(f"=== VIDEO GENERATION ERROR ===")
                        logger.error(f"Error type: {type(e).__name__}")
                        logger.error(f"Error message: {str(e)}")
                        if hasattr(e, 'response'):
                            logger.error(f"Response status: {e.response.status_code}")
                            logger.error(f"Response headers: {dict(e.response.headers)}")
                            try:
                                error_body = e.response.json()
                                logger.error(f"Response body: {error_body}")
                            except:
                                logger.error(f"Response text: {e.response.text}")
                        raise
            
            elif step_type == "tts":
                model = step.get("model")
                if not model: 
                    raise ValueError(f"TTS step {i+1} is missing 'model'.")
                input_from = step.get("input_from")
                if not input_from: 
                    raise ValueError("TTS step is missing 'input_from'.")
                text_input = variables.get(input_from)
                if not text_input: 
                    raise Exception(f"Input for TTS (from variable '{input_from}') is empty or missing.")
                voice = step.get("voice")
                if not voice: 
                    raise ValueError("TTS step is missing required 'voice' parameter.")
                data = {"model": model, "input": text_input, "voice": voice}
                headers = {"Authorization": f"Bearer {get_random_api_key()}", "Content-Type": "application/json"}
                
                # Use the same retry logic as the direct tools
                async with httpx.AsyncClient() as client:
                    response = await make_api_request_with_retry(
                        client, 
                        f"{A4F_API_BASE_URL}/audio/speech", 
                        headers=headers, 
                        json_data=data,
                        timeout=60
                    )
                    result = response.content  # TTS returns bytes directly
                temp_file_path = f"{uuid.uuid4()}.mp3"
                
                # Write the audio bytes to file
                with open(temp_file_path, "wb") as f: 
                    f.write(result)
                temp_files.append(temp_file_path)
                output = temp_file_path
            
            elif step_type == "transcription":
                model = step.get("model")
                if not model: 
                    raise ValueError(f"Transcription step {i+1} is missing 'model'.")
                file_path_from = step.get("file_path_from")
                if not file_path_from: 
                    raise ValueError("Transcription step is missing 'file_path_from'.")
                
                # Debug logging
                logger.info(f"=== TRANSCRIPTION STEP DEBUG ===")
                logger.info(f"file_path_from: {file_path_from}")
                logger.info(f"variables: {variables}")
                logger.info(f"initial_file_path: {initial_file_path}")
                
                # Handle case where file_path_from might be a literal string with braces
                if file_path_from.startswith("{") and file_path_from.endswith("}"):
                    # Extract the variable name from braces
                    actual_variable = file_path_from[1:-1]
                    logger.info(f"Extracted variable name: {actual_variable}")
                    file_path = variables.get(actual_variable)
                else:
                    file_path = variables.get(file_path_from)
                
                logger.info(f"resolved file_path: {file_path}")
                
                if not file_path or not os.path.exists(file_path): 
                    raise Exception(f"File path for transcription (from variable '{file_path_from}') is invalid or file does not exist.")
                headers = {"Authorization": f"Bearer {get_random_api_key()}"}
                
                # Use the exact same format as single model transcription
                logger.info(f"=== WORKFLOW TRANSCRIPTION DEBUG ===")
                logger.info(f"File path: {file_path}")
                logger.info(f"File exists: {os.path.exists(file_path)}")
                logger.info(f"File size: {os.path.getsize(file_path) if os.path.exists(file_path) else 'N/A'}")
                
                async with httpx.AsyncClient() as client:
                    with open(file_path, 'rb') as f:
                        logger.info(f"File opened successfully")
                        response = await client.post(
                            f"{A4F_API_BASE_URL}/audio/transcriptions", 
                            headers=headers, 
                            files={'file': f}, 
                            data={'model': model}, 
                            timeout=1200
                        )
                        logger.info(f"Response status: {response.status_code}")
                        logger.info(f"Response headers: {dict(response.headers)}")
                        result = response.json()
                        logger.info(f"=== TRANSCRIPTION RESPONSE ===")
                        logger.info(f"Response JSON: {result}")
                        logger.info(f"Response keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
                        
                        # Handle different possible response structures
                        if isinstance(result, dict):
                            if 'text' in result:
                                output = result['text']
                            elif 'transcription' in result:
                                output = result['transcription']
                            elif 'content' in result:
                                output = result['content']
                            else:
                                # If no expected field, use the entire result as string
                                output = str(result)
                        else:
                            # If result is not a dict, convert to string
                            output = str(result)

            elif step_type == "custom_api":
                url = substitute_variables(step.get('url'), variables)
                if not url: 
                    raise ValueError("Custom API step is missing 'url'.")
                method = step.get('method', 'GET')
                headers = substitute_variables(step.get('headers', {}), variables)
                body = substitute_variables(step.get('body_template', {}), variables)
                
                # Log the request details for debugging
                logger.info(f"Custom API Request: {method} {url}")
                logger.info(f"Headers: {headers}")
                logger.info(f"Body: {body}")
                
                # Use the same retry logic as the direct tools
                async with httpx.AsyncClient() as client:
                    response = await make_api_request_with_retry(
                        client, 
                        url, 
                        headers=headers, 
                        json_data=body,
                        timeout=300  # Longer timeout for external APIs
                    )
                    result = response.json()
                logger.info(f"Custom API Raw Result Type: {type(result)}")
                logger.info(f"Custom API Raw Result: {result}")
                
                # Handle different response types from Segmind
                if isinstance(result, dict):
                    # If it's a dictionary, look for common video URL fields
                    if "video_url" in result:
                        output = result["video_url"]
                    elif "url" in result:
                        output = result["url"]
                    elif "data" in result and isinstance(result["data"], dict):
                        if "video_url" in result["data"]:
                            output = result["data"]["video_url"]
                        elif "url" in result["data"]:
                            output = result["data"]["url"]
                        else:
                            output = result
                    else:
                        output = result
                elif isinstance(result, str):
                    # If it's a string, it might be a direct URL
                    if result.startswith("http"):
                        output = result
                    else:
                        output = result
                else:
                    output = result
                
                # Apply output parser if specified
                if step.get("output_parser_path"):
                    try:
                        path_keys = step["output_parser_path"].split('.')
                        temp_output = output
                        for key in path_keys:
                            if key.isdigit(): 
                                temp_output = temp_output[int(key)]
                            else: 
                                temp_output = temp_output[key]
                        output = temp_output
                    except Exception as e:
                        logger.warning(f"Output parser failed: {e}, using original output")
                
                logger.info(f"Custom API Final Output: {output}")
            else:
                await context.bot.send_message(chat_id, f"⚠️ Unknown or unsupported step type: {step_type}")
                return
            
            if output_variable:
                variables[output_variable] = output
                logger.info(f"=== VARIABLE ASSIGNMENT ===")
                logger.info(f"Assigned '{output_variable}' = '{output}'")
                logger.info(f"All variables after assignment: {variables}")
                
            if i == len(workflow["steps"]) - 1:
                await context.bot.send_message(chat_id, "✅ Workflow complete\\! Here is your final result:", parse_mode=ParseMode.MARKDOWN_V2)
                logger.info(f"Final output type: {type(output)}, value: {output}")
                
                if isinstance(output, str) and output.startswith("http"):
                    # Handle A4F video URLs specially since they require auth
                    if "api.a4f.co" in output and "/videos/serve/" in output:
                        try:
                            # Download the video with authentication
                            headers = {"Authorization": f"Bearer {get_random_api_key()}"}
                            response = requests.get(output, headers=headers, stream=True, timeout=60)
                            response.raise_for_status()
                            
                            # Save to temporary file
                            temp_video_path = os.path.join(TEMP_DIR, f"temp_video_{uuid.uuid4()}.mp4")
                            with open(temp_video_path, 'wb') as f:
                                for chunk in response.iter_content(chunk_size=8192):
                                    f.write(chunk)
                            
                            # Send the video file
                            with open(temp_video_path, 'rb') as video_file:
                                await context.bot.send_video(chat_id, video=video_file, caption="🎬 Here's your video\\!")
                            
                            # Clean up
                            os.remove(temp_video_path)
                        except Exception as e:
                            logger.error(f"Failed to download and send video: {e}")
                            # Fallback to URL if download fails
                            await context.bot.send_message(chat_id, f"🎬 Here's your video\\!\n\n🔗 Video URL: {escape_markdown_v2(output)}", parse_mode=ParseMode.MARKDOWN_V2)
                    else:
                        try:
                            response = requests.get(output, stream=True, timeout=15, allow_redirects=True)
                            response.raise_for_status()
                            content_type = response.headers.get('Content-Type', '')
                            response.close()

                            if 'image' in content_type:
                                await context.bot.send_photo(chat_id, photo=output, caption="🖼️ Here's your image!")
                            elif 'video' in content_type:
                                await context.bot.send_video(chat_id, video=output, caption="🎬 Here's your video!")
                            elif 'audio' in content_type:
                                await context.bot.send_audio(chat_id, audio=output, caption="🎤 Here's your audio!")
                            else:
                                await context.bot.send_message(chat_id, f"🔗 Result URL: {escape_markdown_v2(output)}", parse_mode=ParseMode.MARKDOWN_V2)
                        except requests.exceptions.RequestException as e:
                            logger.warning(f"Could not determine content type for URL {output}: {e}")
                            await context.bot.send_message(chat_id, f"🔗 Result URL (could not fetch preview): {escape_markdown_v2(output)}", parse_mode=ParseMode.MARKDOWN_V2)
                elif isinstance(output, str) and os.path.exists(output):
                    try:
                        with open(output, 'rb') as f: 
                            await context.bot.send_audio(chat_id, audio=f, caption="🎤 Here's your audio!")
                    except Exception as e:
                        logger.error(f"Error sending file: {e}")
                        await context.bot.send_message(chat_id, f"📄 File path: {escape_markdown_v2(output)}", parse_mode=ParseMode.MARKDOWN_V2)
                elif output is not None:
                    try:
                        await context.bot.send_message(chat_id, f"📄 Result:\n\n`{escape_markdown_v2(str(output))}`", parse_mode=ParseMode.MARKDOWN_V2)
                    except Exception as e:
                        logger.error(f"Error sending result message: {e}")
                        await context.bot.send_message(chat_id, "✅ Workflow completed successfully!")
    except Exception as e:
        logger.error(f"Error executing step: {e}")
        error_message = f"❌ Error during workflow execution:\n`{escape_markdown_v2(str(e))}`\nAborting\\."
        await context.bot.send_message(chat_id, error_message, parse_mode=ParseMode.MARKDOWN_V2)
    finally:
        for f in temp_files:
            if os.path.exists(f): 
                os.remove(f)
        if initial_file_path and os.path.exists(initial_file_path):
             os.remove(initial_file_path)

def escape_markdown_v2(text: str) -> str:
    return re.sub(f'([{re.escape(r"_*[]()~`>#+-=|{}.!")}])', r'\\\1', text)

async def safe_edit_message(query, text, reply_markup=None, parse_mode=None):
    """Safely edit a message, handling both text and media messages."""
    try:
        await query.edit_message_text(text=text, reply_markup=reply_markup, parse_mode=parse_mode)
    except error.BadRequest as e:
        if "Message to edit not found" in str(e) or "Message is not modified" in str(e):
            # Message was deleted or is too old, send a new message
            try:
                await query.message.reply_text(text=text, reply_markup=reply_markup, parse_mode=parse_mode)
            except Exception:
                # If reply also fails, try to send a new message to the chat
                await query.message.chat.send_message(text=text, reply_markup=reply_markup, parse_mode=parse_mode)
        else:
            # If the current message is a media message, send a new text message
            try:
                await query.message.reply_text(text=text, reply_markup=reply_markup, parse_mode=parse_mode)
                await query.message.delete()
            except Exception:
                # If delete fails, just send a new message
                await query.message.reply_text(text=text, reply_markup=reply_markup, parse_mode=parse_mode)

async def safe_edit_message_for_message(message, text, reply_markup=None, parse_mode=None):
    """Safely edit a message when we have a Message object instead of CallbackQuery."""
    try:
        await message.edit_text(text=text, reply_markup=reply_markup, parse_mode=parse_mode)
    except error.BadRequest:
        # If the current message is a media message, send a new text message
        await message.reply_text(text=text, reply_markup=reply_markup, parse_mode=parse_mode)
        await message.delete()

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
        elif isinstance(error_obj, httpx.ReadTimeout):
            return f"❌ *Connection Timeout:* The request to the API timed out\\. Please try again\\."
        else:
            return f"❌ *Connection Error:*\n{escape_markdown_v2(str(error_obj))}"
    except Exception:
        return f"❌ *An unexpected API error occurred:*\n{escape_markdown_v2(str(error_obj))}"



async def forward_to_admin_if_watched(message: Message, context: ContextTypes.DEFAULT_TYPE):
    user_id = message.chat.id
    if user_id in _watched_users and ADMIN_CHAT_ID:
        await context.bot.forward_message(chat_id=ADMIN_CHAT_ID, from_chat_id=message.chat_id, message_id=message.message_id)

async def notify_admin_of_new_user(context: ContextTypes.DEFAULT_TYPE, user: Update.effective_user, referred_by=None):
    if not ADMIN_CHAT_ID: return
    base_text = (f"🎉 *New User Alert*\n\n"
                 f"*Name:* {escape_markdown_v2(user.full_name)}\n"
                 f"*ID:* `{user.id}`\n"
                 f"*Username:* {'@' + escape_markdown_v2(user.username) if user.username else 'N/A'}")
    if referred_by:
        base_text += f"\n*Referred By:* `{referred_by}`"
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

    keys_to_clear = ['chat_history', 'image_edit_path', 'temp_file_path', 'last_prompt',
                     'voice_mode_voice', 'voice_chat_history', 'mixer_concept_1',
                     'voice_mode_pro_active', 'voice_mode_pro_context', 'voice_mode_pro_history']
    for key in keys_to_clear:
        context.user_data.pop(key, None)

    keyboard = [
        [InlineKeyboardButton("💬 AI Chat", callback_data="act_chat"), InlineKeyboardButton("🎨 Image Gen", callback_data="act_image")],
        [InlineKeyboardButton("🖼️ Image Edit", callback_data="act_image_edit"), InlineKeyboardButton("🎬 Video Gen", callback_data="act_video")],
        [InlineKeyboardButton("🎙️ TTS", callback_data="act_tts"), InlineKeyboardButton("✍️ Transcription", callback_data="act_transcription")],
        [InlineKeyboardButton("🎤 Voice Mode", callback_data="act_voice_mode"), InlineKeyboardButton("✨ Voice Mode Pro", callback_data="act_voice_mode_pro")],
        [InlineKeyboardButton("🎨 Image Mixer", callback_data="act_mixer"), InlineKeyboardButton("🌐 Web Pilot", callback_data="act_web")],
        [InlineKeyboardButton("🚀 Create Agent", callback_data="act_create_workflow"), InlineKeyboardButton("📚 My Agents", callback_data="act_list_workflows")],
        [InlineKeyboardButton("🏪 Agent Hub", callback_data="act_marketplace"), InlineKeyboardButton("💼 My Owned Agents", callback_data="act_owned_workflows")],
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

    # GIF URL from the channel
    gif_url = "https://t.me/sypnsai/3"
    
    if update.callback_query:
        await update.callback_query.answer()
        await update.callback_query.edit_message_media(
            media=InputMediaAnimation(
                media=gif_url,
                caption=welcome_text,
                parse_mode=ParseMode.MARKDOWN_V2
            ),
            reply_markup=reply_markup
        )
    else:
        await update.message.reply_animation(
            animation=gif_url,
            caption=welcome_text,
            reply_markup=reply_markup,
            parse_mode=ParseMode.MARKDOWN_V2
        )
    return USER_MAIN

async def profile_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query_or_message = update.callback_query or update.message
    user_id = query_or_message.from_user.id
    user_data = load_user_data(user_id)
    stats = user_data.get('stats', {})
    stats_text = "\n".join([f"  •  `{k.title()}`: {v}" for k, v in stats.items() if v > 0]) or "_No activity yet\\._"

    credits = "♾️ Unlimited (Admin)" if is_admin(user_id) else user_data.get('credits', 0)

    bot_username = (await context.bot.get_me()).username
    referral_code = user_data.get('referral_code', 'N/A')
    referral_link = f"https://t.me/{bot_username}?start={referral_code}"

    text = (f"👤 *My Profile*\n\n"
            f"💰 *Credits:* `{credits}`\n"
            f"📈 *Users Referred:* `{user_data.get('referrals_made', 0)}`\n\n"
            f"🤝 *Your Referral Link:*\n`{escape_markdown_v2(referral_link)}`\n\n"
            f"📊 *Usage Statistics*\n{stats_text}\n\n"
            f"Share your link to earn *{_settings['referral_bonus']}* credits for each new user who joins\\! You can also use /redeem\\.")

    reply_markup = InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Back to Main Menu", callback_data="main_menu")]])
    if update.callback_query:
        await update.callback_query.answer()
        await safe_edit_message(update.callback_query, text, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN_V2)
    else:
        await update.message.reply_markdown_v2(text, reply_markup=reply_markup)
    return USER_MAIN

async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query_or_message = update.callback_query or update.message
    help_text = ("*❔ Help & Information*\n\n*Available Commands:*\n"
                 "`/start` \\- Return to the main menu\\.\n"
                 "`/newchat` \\- Clear history for a fresh AI Chat\\.\n"
                 "`/me` \\- Check your credits, referral code, and stats\\.\n"
                 "`/personality` \\- Set a custom personality for the chat AI\\.\n"
                 f"`/redeem {escape_markdown_v2('<CODE>')}` \\- Redeem a code for credits\\.\n"
                 "`/help` \\- Show this help message\\.\n"
                 "`/exit` \\- Stop the current mode \\(like Voice Mode\\)\\.\n")
    if is_admin(query_or_message.from_user.id):
        help_text += ("\n\n*Admin Commands:*\n"
                      "`/admin` \\- Open the admin dashboard\\.\n"
                      "`/globalstats` \\- View aggregate stats for all users\\.\n"
                      f"`/msg {escape_markdown_v2('<id> <msg>')}` \\- Send a message to a user\\.\n"
                      f"`/cred {escape_markdown_v2('<id> <amt>')}` \\- Give/deduct credits\\.\n"
                      f"`/watch {escape_markdown_v2('<id>')}` \\- Forward a user's messages to you\\.\n"
                      f"`/unwatch {escape_markdown_v2('<id>')}` \\- Stop watching a user\\.\n"
                      "`/listwatched` \\- List all watched users\\.\n"
                      f"`/gethistory {escape_markdown_v2('<id>')}` \\- Get a user's chat history\\.\n"
                      f"`/getdata {escape_markdown_v2('<id>')}` \\- Get a user's data file\\.")

    reply_markup = InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Back to Main Menu", callback_data="main_menu")]])
    if update.callback_query:
        await update.callback_query.answer()
        await safe_edit_message(update.callback_query, help_text, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN_V2)
    else:
        await update.message.reply_markdown_v2(help_text, reply_markup=reply_markup)
    return USER_MAIN

async def new_chat_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data.pop('chat_history', None)
    await update.message.reply_text("✅ Chat history cleared. The AI will now have no memory of our previous conversation.")

async def cancel_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await cleanup_files(context.user_data.pop('image_edit_path', None), context.user_data.pop('temp_file_path', None))
    keys_to_clear = ['voice_mode_voice', 'voice_chat_history', 'mixer_concept_1',
                     'voice_mode_pro_active', 'voice_mode_pro_context', 'voice_mode_pro_history']
    for key in keys_to_clear:
        context.user_data.pop(key, None)
    await update.message.reply_text("Action cancelled. You can start again with /start.")
    return await start_command(update, context)

async def _speak(context: ContextTypes.DEFAULT_TYPE, chat_id: int, text: str, voice: str = "onyx"):
    try:
        async with httpx.AsyncClient() as client:
            api_key = get_random_api_key()
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            data = {"model": "provider-3/tts-1", "input": text, "voice": voice}
            response = await client.post(f"{A4F_API_BASE_URL}/audio/speech", headers=headers, json=data, timeout=60)
            response.raise_for_status()
            await context.bot.send_voice(chat_id=chat_id, voice=response.content)
            return True
    except Exception as e:
        logger.error(f"Error in _speak helper: {e}")
        await context.bot.send_message(chat_id=chat_id, text=f"_(Could not generate voice, an error occurred.)_")
        return False

async def _analyze_and_deliver(update: Update, context: ContextTypes.DEFAULT_TYPE, response_text: str):
    is_code = "```" in response_text
    is_list = any(line.strip().startswith(("- ", "* ", "1. ", "2. ")) for line in response_text.split('\n'))
    is_long = len(response_text) > 400

    voice = context.user_data.get('voice_mode_voice', 'onyx')

    if is_code or is_list or is_long:
        heads_up_text = "I have the answer for you. Because it contains code or is quite long, I'm sending it as a text message."
        await _speak(context, update.effective_chat.id, heads_up_text, voice)
        try:
            await update.message.reply_text(response_text, parse_mode=ParseMode.MARKDOWN)
        except error.BadRequest:
            await update.message.reply_text(response_text)
    else:
        await _speak(context, update.effective_chat.id, response_text, voice)

async def _get_pro_voice_intent(text: str) -> dict:
    cognitive_prompt = (
        "You are the cognitive routing engine for an advanced voice AI assistant. Your primary goal is to analyze the user's request and select the most appropriate tool for the job. "
        f"Analyze the request: '{text}' and respond ONLY with a valid JSON object. Do not add any explanation. "
        "1. Determine the 'intent': Is it 'chat', 'create_image', 'create_video', or 'edit_image'? For edits, look for keywords like 'edit', 'change', 'make it'. For videos, look for keywords like 'video', 'movie', 'animation', 'create a video'.\n"
        "2. If the intent is 'chat', determine the 'category':\n"
        " - 'web_search': Prioritize this. Applies to requests for current info, news, real-time data, or explicit search commands.\n"
        " - 'coding': Involves code, programming languages, or algorithms.\n"
        " - 'reasoning': Requires complex, step-by-step logical problem-solving.\n"
        " - 'general_chat': Any other conversational request.\n"
        "3. Select the appropriate 'model' from the provided mapping.\n"
        "4. Extract the core 'prompt' for the final action.\n\n"
        f"Model Mapping: {json.dumps(PRO_MODEL_MAPPING)}\n\n"
        "Example 1: User says 'Create a picture of a robot fighting a dinosaur.'\n"
        '{"intent": "create_image", "model": "provider-4/imagen-3", "prompt": "a robot fighting a dinosaur"}\n'
        "Example 2: User says 'Who won the last Super Bowl?'\n"
        '{"intent": "chat", "category": "web_search", "model": "provider-3/gpt-4o-mini-search-preview", "prompt": "Who won the last Super Bowl?"}\n'
        "Example 3: User says 'Now make it look like a cartoon.'\n"
        '{"intent": "edit_image", "model": "provider-6/black-forest-labs-flux-1-kontext-pro", "prompt": "make it look like a cartoon"}\n'
        "Example 4: User says 'Create a video of a cat playing with a ball.'\n"
        '{"intent": "create_video", "model": "provider-6/wan-2.1", "prompt": "a cat playing with a ball"}'
    )

    try:
        async with httpx.AsyncClient() as client:
            api_key = get_random_api_key()
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            data = {"model": PRO_REASONING_MODEL, "messages": [{"role": "user", "content": cognitive_prompt}], "temperature": 0.1}
            response = await client.post(f"{A4F_API_BASE_URL}/chat/completions", headers=headers, json=data, timeout=1200)
            response.raise_for_status()
            json_response = response.json()
            choices = json_response.get('choices', [])
            content = None
            if choices and len(choices) > 0:
                first_choice = choices[0]
                content = first_choice.get('message', {}).get('content')
            if not content:
                raise ValueError("API returned an empty content response.")
            cleaned_content = re.sub(r'```json\n(.*?)\n```', r'\1', content, flags=re.DOTALL)
            return json.loads(cleaned_content)
    except Exception as e:
        logger.error(f"Cognitive routing failed: {e}. Falling back to general chat.")
        return {"intent": "chat", "category": "general_chat", "model": PRO_MODEL_MAPPING["general_chat"], "prompt": text}

async def voice_mode_pro_start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    context.user_data['voice_mode_pro_active'] = True
    context.user_data['voice_mode_pro_history'] = deque(maxlen=100)
    context.user_data['voice_mode_pro_context'] = {}
    keyboard = [[InlineKeyboardButton(v.capitalize(), callback_data=f"vm_pro_voice_{v}") for v in TTS_VOICES[:3]],
                [InlineKeyboardButton(v.capitalize(), callback_data=f"vm_pro_voice_{v}") for v in TTS_VOICES[3:]]]
    
    message_text = ("✨*Voice Mode Pro* activated\\. This advanced mode understands complex commands it can code, create images, videos and edit images Just instruct it via voice message\\.\n\n"
                   "You can send photos\\. I'll ask you how to edit them\\.\n\n"
                   "First, please choose a voice for our conversation\\.")
    
    try:
        await query.edit_message_text(message_text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.MARKDOWN_V2)
    except error.BadRequest:
        # If the current message is a media message, send a new text message
        await query.message.reply_text(message_text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.MARKDOWN_V2)
        await query.message.delete()
    
    return AWAITING_VOICE_MODE_PRO_INPUT

async def voice_mode_pro_voice_select_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    voice = query.data.split('_')[-1]
    context.user_data['voice_mode_voice'] = voice
    
    message_text = (f"✅ Voice set to *{voice.capitalize()}*\\.\n\n"
                   "You can now send me voice messages or photos\\. To stop, use the /exit command\\.")
    
    try:
        await query.edit_message_text(message_text, parse_mode=ParseMode.MARKDOWN_V2)
    except error.BadRequest:
        # If the current message is a media message, send a new text message
        await query.message.reply_text(message_text, parse_mode=ParseMode.MARKDOWN_V2)
        await query.message.delete()
    
    return AWAITING_VOICE_MODE_PRO_INPUT

async def voice_mode_pro_input_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    await forward_to_admin_if_watched(update.message, context)
    if not context.user_data.get('voice_mode_pro_active'):
        await update.message.reply_text("Voice Mode Pro isn't active. Please start it from the main menu.")
        return await start_command(update, context)
    
    if not use_multiple_credits(user_id, 2):
        await _speak(context, user_id, "You need at least 2 credits for Voice Mode Pro! Please redeem a code or wait for your daily refill.")
        return await cancel_handler(update, context)
    processing_message = await update.message.reply_text("🎙️ Processing...")
    temp_audio_path = os.path.join(TEMP_DIR, f"pro_{uuid.uuid4()}.ogg")
    try:
        await processing_message.edit_text("👂 Transcribing...")
        file_obj = await update.message.voice.get_file()
        await file_obj.download_to_drive(temp_audio_path)
        transcribed_text = None
        with open(temp_audio_path, 'rb') as f:
            async with httpx.AsyncClient() as client:
                headers = {"Authorization": f"Bearer {get_random_api_key()}"}
                trans_res = await make_api_request_with_retry(
                    client,
                    f"{A4F_API_BASE_URL}/audio/transcriptions",
                    headers=headers,
                    files={'file': f},
                    data={'model': 'provider-6/distil-whisper-large-v3-en'},
                    timeout=120
                )
                transcribed_text = trans_res.json().get('text')
                if not transcribed_text: raise ValueError("Transcription failed.")
        await processing_message.edit_text("🧠 Thinking...")
        intent_data = await _get_pro_voice_intent(transcribed_text)
        intent = intent_data.get("intent")
        model = intent_data.get("model")
        prompt = intent_data.get("prompt")
        context.user_data['voice_mode_pro_history'].append({"role": "user", "content": transcribed_text})
        if intent == "create_image":
            await processing_message.edit_text("🎨 Painting...")
            try:
                async with httpx.AsyncClient() as client:
                    api_response = await make_api_request_with_retry(
                        client,
                        f"{A4F_API_BASE_URL}/images/generations",
                        headers={"Authorization": f"Bearer {get_random_api_key()}", "Content-Type": "application/json"},
                        json_data={"model": model, "prompt": prompt, "size": "1024x1792"},
                        timeout=180
                    )
                    json_data = api_response.json()
                    data_list = json_data.get('data', [])
                    if not data_list or len(data_list) == 0:
                        raise ValueError("Image generation failed to return data.")
                    media_url = data_list[0].get('url')
                    if not media_url:
                        raise ValueError("Image generation failed to return a URL.")
                sent_message = await update.message.reply_photo(photo=media_url)
                photo_file = await sent_message.photo[-1].get_file()
                edit_path = os.path.join(TEMP_DIR, f"edit_{uuid.uuid4()}.jpg")
                await photo_file.download_to_drive(edit_path)
                context.user_data['voice_mode_pro_context'] = {'last_media_type': 'image', 'last_media_path': edit_path}
                await _speak(context, user_id, "Here is the image you requested. You can now tell me how you'd like to edit it.", voice=context.user_data.get('voice_mode_voice'))
            except Exception as e:
                await handle_api_error_with_retry_and_refund(user_id, e, credit_cost=2, context=context, voice_mode=True)
        elif intent == "edit_image":
            pro_context = context.user_data.get('voice_mode_pro_context', {})
            if pro_context.get('last_media_type') != 'image' or not os.path.exists(pro_context.get('last_media_path', '')):
                await _speak(context, user_id, "I don't have an image to edit. Please create one first.", voice=context.user_data.get('voice_mode_voice'))
            else:
                await processing_message.edit_text("🖌️ Editing...")
                try:
                    image_path = pro_context['last_media_path']
                    with open(image_path, 'rb') as img_file:
                        async with httpx.AsyncClient() as client:
                            api_response = await make_api_request_with_retry(
                                client,
                                f"{A4F_API_BASE_URL}/images/edits",
                                headers={"Authorization": f"Bearer {get_random_api_key()}"},
                                data={"prompt": prompt, "model": model},
                                files={"image": img_file},
                                timeout=180
                            )
                            json_data = api_response.json()
                            data_list = json_data.get('data', [])
                            if not data_list or len(data_list) == 0:
                                raise ValueError("Image editing failed to return data.")
                            media_url = data_list[0].get('url')
                            if not media_url:
                                raise ValueError("Image editing failed to return a URL.")
                    sent_message = await update.message.reply_photo(photo=media_url)
                    await cleanup_files(pro_context['last_media_path'])
                    photo_file = await sent_message.photo[-1].get_file()
                    edit_path = os.path.join(TEMP_DIR, f"edit_{uuid.uuid4()}.jpg")
                    await photo_file.download_to_drive(edit_path)
                    context.user_data['voice_mode_pro_context'] = {'last_media_type': 'image', 'last_media_path': edit_path}
                    await _speak(context, user_id, "Here is the edited version. What would you like to do next?", voice=context.user_data.get('voice_mode_voice'))
                except Exception as e:
                    await handle_api_error_with_retry_and_refund(user_id, e, credit_cost=2, context=context, voice_mode=True)
        elif intent == "create_video":
            await processing_message.edit_text("🎬 Creating video...")
            try:
                async with httpx.AsyncClient() as client:
                    api_response = await make_api_request_with_retry(
                        client,
                        f"{A4F_API_BASE_URL}/video/generations",
                        headers={"Authorization": f"Bearer {get_random_api_key()}", "Content-Type": "application/json"},
                        json_data={"model": model, "prompt": prompt, "ratio": "16:9", "quality": "480p", "duration": 4},
                        timeout=300
                    )
                    json_data = api_response.json()
                    data_list = json_data.get('data', [])
                    if not data_list or len(data_list) == 0:
                        raise ValueError("Video generation failed to return data.")
                    media_url = data_list[0].get('url')
                    if not media_url:
                        raise ValueError("Video generation failed to return a URL.")
                sent_message = await update.message.reply_video(video=media_url)
                await _speak(context, user_id, "Here is your video! What would you like to do next?", voice=context.user_data.get('voice_mode_voice'))
            except Exception as e:
                await handle_api_error_with_retry_and_refund(user_id, e, credit_cost=2, context=context, voice_mode=True)
        elif intent == "chat":
            await processing_message.edit_text("💬 Chatting...")
            try:
                messages = list(context.user_data['voice_mode_pro_history'])
                data = {"model": model, "messages": messages}
                async with httpx.AsyncClient() as client:
                    headers = {"Authorization": f"Bearer {get_random_api_key()}", "Content-Type": "application/json"}
                    api_response = await make_api_request_with_retry(
                        client,
                        f"{A4F_API_BASE_URL}/chat/completions",
                        headers=headers,
                        json_data=data,
                        timeout=120
                    )
                    json_data = api_response.json()
                    choices = json_data.get('choices', [])
                    if choices and len(choices) > 0:
                        first_choice = choices[0]
                        response_text = first_choice.get('message', {}).get('content', "I'm sorry, I couldn't process that.")
                    else:
                        response_text = "I'm sorry, I couldn't process that."
                context.user_data['voice_mode_pro_history'].append({"role": "assistant", "content": response_text})
                await _analyze_and_deliver(update, context, response_text)
            except Exception as e:
                await handle_api_error_with_retry_and_refund(user_id, e, credit_cost=2, context=context, voice_mode=True)
        else:
            await _speak(context, user_id, f"I understood that you want to '{intent}', but that feature is not fully connected in Voice Mode Pro yet.", voice=context.user_data.get('voice_mode_voice'))
        await processing_message.delete()
    except Exception as e:
        await handle_api_error_with_retry_and_refund(user_id, e, credit_cost=2, context=context, voice_mode=True)
        if 'processing_message' in locals() and processing_message:
            try: await processing_message.delete()
            except error.TelegramError as te: logger.warning(f"Could not delete processing message: {te}")
    finally:
        await cleanup_files(temp_audio_path)
    return AWAITING_VOICE_MODE_PRO_INPUT

async def voice_mode_pro_photo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    await forward_to_admin_if_watched(update.message, context)
    
    if not context.user_data.get('voice_mode_pro_active'):
        await update.message.reply_text("Voice Mode Pro isn't active. Please start it from the main menu.")
        return await start_command(update, context)
    
    if not update.message.photo:
        await update.message.reply_text("Please send a photo to edit.")
        return AWAITING_VOICE_MODE_PRO_INPUT
    
    # Voice Mode Pro costs 2 credits
    if not use_multiple_credits(user_id, 2):
        await _speak(context, user_id, "You need at least 2 credits for Voice Mode Pro! Please redeem a code or wait for your daily refill.")
        return await cancel_handler(update, context)
    
    processing_message = await update.message.reply_text("📸 Processing photo...")
    
    try:
        # Download the photo
        photo_file = await update.message.photo[-1].get_file()
        photo_path = os.path.join(TEMP_DIR, f"pro_photo_{uuid.uuid4()}.jpg")
        await photo_file.download_to_drive(photo_path)
        
        # Store the photo path in context for editing
        context.user_data['voice_mode_pro_context'] = {
            'last_media_type': 'image', 
            'last_media_path': photo_path
        }
        
        await processing_message.edit_text("✅ Photo received! Now send me a voice message telling me how you'd like to edit it.")
        await _speak(context, user_id, "I've received your photo. Please send me a voice message describing how you'd like to edit it.", voice=context.user_data.get('voice_mode_voice'))
        
    except Exception as e:
        await handle_api_error_with_retry_and_refund(user_id, e, credit_cost=2, context=context, voice_mode=True)
        await cleanup_files(photo_path) if 'photo_path' in locals() else None
    
    return AWAITING_VOICE_MODE_PRO_INPUT

async def set_personality_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    buttons = [[InlineKeyboardButton("📝 Custom", callback_data="p_custom")]]
    await safe_edit_message(
        update.callback_query, 
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
    choice = query.data.split("_")[-1]
    if choice == "custom":
        user_id = update.effective_user.id
        user_data = load_user_data(user_id)
        current_personality = user_data.get("personality")
        text = "🎭 *Set Custom AI Personality*\n\n"
        if current_personality: text += f"Current personality: _{escape_markdown_v2(current_personality)}_\n\n"
        text += "Please send me the new personality prompt for the AI \\(e\\.g\\., 'You are a helpful pirate'\\)\\. To remove it, send /clear\\."
        await safe_edit_message(query, text, parse_mode=ParseMode.MARKDOWN_V2)
        return AWAITING_PERSONALITY
    elif choice == "presets":
        buttons = [[InlineKeyboardButton(name, callback_data=f"ps_{name}")] for name in PERSONALITY_PRESETS.keys()]
        buttons.append([InlineKeyboardButton("⬅️ Back", callback_data="p_back")])
        await safe_edit_message(query, "🎭 Choose a preset personality:", reply_markup=InlineKeyboardMarkup(buttons))
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
    choice = query.data.split("_", 1)[-1]
    user_id = query.from_user.id
    user_data = load_user_data(user_id)
    user_data["personality"] = PERSONALITY_PRESETS[choice]
    save_user_data(user_id, user_data)
    context.user_data.pop('chat_history', None)
    await safe_edit_message(query, f"✅ Personality set to: *{choice}*. Chat history cleared.", parse_mode=ParseMode.MARKDOWN_V2)
    await query.message.reply_text("Returning to main menu...", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]]))
    return USER_MAIN

async def action_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    _prefix, category = query.data.split('_', 1)
    if category == 'voice_mode_pro':
        return await voice_mode_pro_start_handler(update, context)
    if category == 'create_workflow': 
        return await new_workflow_entry(update, context)
    if category == 'list_workflows': 
        return await list_workflows(update, context)
    if category == 'marketplace': 
        return await marketplace_browse(update, context)
    if category == 'owned_workflows': 
        return await list_owned_workflows(update, context)
    if category == 'help': return await help_handler(update, context)
    if category == 'me': return await profile_handler(update, context)
    if category == 'personality': return await set_personality_handler(update, context)
    if category == 'admin': return await admin_panel(update, context)
    if category == 'voice_mode':
        keyboard = [[InlineKeyboardButton(v.capitalize(), callback_data=f"vm_voice_{v}") for v in TTS_VOICES[:3]], [InlineKeyboardButton(v.capitalize(), callback_data=f"vm_voice_{v}") for v in TTS_VOICES[3:]]]
        await safe_edit_message(query, "🗣️ First, choose a voice for our conversation.", reply_markup=InlineKeyboardMarkup(keyboard))
        return SELECTING_VOICE_FOR_MODE
    if category == 'mixer':
        await safe_edit_message(query, "🎨 Welcome to the Image Mixer Studio!\n\nFirst, send me the primary concept or subject (e.g., 'a cat', 'a knight').")
        return AWAITING_MIXER_CONCEPT_1
    if category == 'web':
        await safe_edit_message(query, "🌐 Web Pilot Mode initiated. What would you like to know or which URL should I check?")
        return AWAITING_WEB_PROMPT

    context.user_data['category'] = category
    last_model = context.user_data.get(f'last_model_{category}')
    if last_model:
        short_model_name = last_model.split('/')[-1]
        keyboard = [[InlineKeyboardButton(f"🚀 Use Last: {short_model_name}", callback_data=f"mr_{category}")], [InlineKeyboardButton("📋 Choose Another Model", callback_data=f"mc_{category}")]]
        await safe_edit_message(query, f"You previously used `{escape_markdown_v2(short_model_name)}`\\. Use it again?", reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.MARKDOWN_V2)
        return SELECTING_MODEL
    return await show_model_selection(query, context)

async def show_model_selection(update_or_query, context: ContextTypes.DEFAULT_TYPE, page=0) -> int:
    category = context.user_data['category']
    model_list = MODELS.get(category, [])
    reply_markup = create_paginated_keyboard(model_list, category, page)
    text = f"💎 *Select a Model for {category.replace('_', ' ').title()}*"
    
    # Check if it's a CallbackQuery or Message
    if hasattr(update_or_query, 'message'):
        # It's a CallbackQuery
        await safe_edit_message(update_or_query, text, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN_V2)
    else:
        # It's a Message
        await safe_edit_message_for_message(update_or_query, text, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN_V2)
    
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
    query = update.callback_query
    await query.answer()
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
            await safe_edit_message(query, "Sorry, there was an error. Please try again.")
            return USER_MAIN
    if not category or not model_name:
        await safe_edit_message(query, "Sorry, an error occurred. Returning to the main menu.")
        return await start_command(update, context)
    
    if context.user_data.get('voice_mode_setup'):
        context.user_data['voice_mode_model'] = model_name
        model_display_name = escape_markdown_v2(model_name.split('/')[-1])
        voice_display_name = context.user_data['voice_mode_voice'].capitalize()
        await safe_edit_message(query, f"🎤 Voice Mode Started with *{voice_display_name}* voice & *{model_display_name}* model\\.\n\nSend a voice message to begin, or use /exit to stop\\.", parse_mode=ParseMode.MARKDOWN_V2)
        context.user_data.pop('voice_mode_setup')
        return AWAITING_VOICE_MODE_INPUT
    context.user_data.update({'model': model_name, f'last_model_{category}': model_name, 'category': category})
    msg_text = f"✅ Model Selected: `{escape_markdown_v2(model_name.split('/')[-1])}`\n\n"
    if category == "image":
        await safe_edit_message(query, msg_text + "📏 Now, choose an aspect ratio\\.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton(name, callback_data=f"is_{size}") for name, size in IMAGE_SIZES.items()]]), parse_mode=ParseMode.MARKDOWN_V2)
        return AWAITING_IMAGE_SIZE
    if category == "video":
        await safe_edit_message(query, msg_text + "🎬 Now, choose a video ratio\\.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton(name, callback_data=f"vr_{ratio}") for name, ratio in VIDEO_RATIOS.items()]]), parse_mode=ParseMode.MARKDOWN_V2)
        return AWAITING_VIDEO_RATIO
    if category == "tts":
        await safe_edit_message(query, msg_text + "🗣️ Now, choose a voice\\.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton(v.capitalize(), callback_data=f"tv_{v}") for v in TTS_VOICES[:3]], [InlineKeyboardButton(v.capitalize(), callback_data=f"tv_{v}") for v in TTS_VOICES[3:]]]), parse_mode=ParseMode.MARKDOWN_V2)
        return AWAITING_TTS_VOICE
    prompt_map = {"chat": "💬 What's on your mind?","transcription": "🎤 Send me a voice message or audio file.","image_edit": "🖼️ First, send the image you want to edit."}
    next_state_map = {"chat": AWAITING_PROMPT, "transcription": AWAITING_AUDIO, "image_edit": AWAITING_IMAGE_FOR_EDIT}
    await safe_edit_message(query, msg_text + escape_markdown_v2(prompt_map[category]), parse_mode=ParseMode.MARKDOWN_V2)
    return next_state_map.get(category, USER_MAIN)

async def image_size_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query; await query.answer()
    context.user_data['image_size'] = query.data.split('_', 1)[-1]
    await safe_edit_message(query, "✅ Size selected.\n\n✍️ Now, what should I create?")
    return AWAITING_PROMPT

async def video_ratio_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query; await query.answer()
    context.user_data['video_ratio'] = query.data.split('_', 1)[-1]
    await safe_edit_message(query, "✅ Ratio selected.\n\n✍️ Now, what's the scene? Describe the video.")
    return AWAITING_PROMPT

async def tts_voice_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query; await query.answer()
    context.user_data['tts_voice'] = query.data.split('_', 1)[-1]
    await safe_edit_message(query, "✅ Voice selected.\n\n✍️ Now, send me the text you want me to say.")
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
                response = await client.post(f"{A4F_API_BASE_URL}/chat/completions", headers=headers, json=reasoning_payload, timeout=1200)
                response.raise_for_status()
                json_data = response.json()
                choices = json_data.get('choices', [])
                if choices and len(choices) > 0 and (reasoned_prompt := choices[0].get('message', {}).get('content')):
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
                    response = await client.post(f"{A4F_API_BASE_URL}/chat/completions", headers=headers, json=data, timeout=1200)
                elif task_type in ['image', 'video', 'image_edit']:
                    data = {"model": context.user_data['model'], "prompt": user_prompt}
                    files = None
                    if task_type == 'image':
                        endpoint, action = 'images/generations', ChatAction.UPLOAD_PHOTO
                        data['size'] = context.user_data.get('image_size', '1024x1024')
                    elif task_type == 'video':
                        endpoint, action = 'video/generations', ChatAction.UPLOAD_VIDEO
                        data.update({'ratio': context.user_data.get('video_ratio', '16:9'), 'quality': '480p', 'duration': 4})
                        # Log the direct video tool request
                        logger.info(f"=== DIRECT VIDEO TOOL REQUEST ===")
                        logger.info(f"URL: {A4F_API_BASE_URL}/{endpoint}")
                        logger.info(f"Headers: {headers}")
                        logger.info(f"Data: {data}")
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
                elif task_type == 'transcription':
                    await context.bot.send_chat_action(update.effective_chat.id, ChatAction.TYPING)
                    file_obj = await (message.voice or message.audio).get_file()
                    temp_filename = os.path.join(TEMP_DIR, f"temp_{uuid.uuid4()}.ogg")
                    await file_obj.download_to_drive(temp_filename)
                    context.user_data['temp_file_path'] = temp_filename
                    with open(temp_filename, 'rb') as f:
                        response = await client.post(f"{A4F_API_BASE_URL}/audio/transcriptions", headers=headers, files={'file': f}, data={'model': context.user_data['model']}, timeout=1200)
                
                # Handle regular single-model tasks
                response.raise_for_status()
                
                # Log response for video tool
                if task_type == 'video':
                    logger.info(f"=== DIRECT VIDEO TOOL RESPONSE ===")
                    logger.info(f"Status Code: {response.status_code}")
                    logger.info(f"Response Headers: {dict(response.headers)}")
                    try:
                        json_data = response.json()
                        logger.info(f"Response Body: {json_data}")
                    except Exception as e:
                        logger.error(f"Failed to parse response JSON: {e}")
                        logger.error(f"Response Text: {response.text}")
                else:
                    json_data = response.json() if task_type not in ['tts'] else None
                
                if task_type == 'chat':
                    choices = json_data.get('choices', [])
                    result_text = None
                    if choices and len(choices) > 0:
                        result_text = choices[0].get('message', {}).get('content')
                    if not result_text:
                        raise ValueError("API returned empty response.")
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
                elif task_type in ['image', 'image_edit', 'video']:
                     data_list = json_data.get('data')
                     if not data_list:
                         raise ValueError("API returned no data.")
                     
                     # Handle both list and dictionary formats
                     if isinstance(data_list, list):
                         if not data_list or not (media_url := data_list[0].get('url')):
                             raise ValueError("API returned no media URL.")
                     else:
                         if not (media_url := data_list.get('url')):
                             raise ValueError("API returned no media URL.")
                     
                     caption = f"_{escape_markdown_v2(user_prompt)}_"
                     if task_type in ['image', 'image_edit']:
                         sent_message = await context.bot.send_photo(update.effective_chat.id, photo=media_url, caption=caption, parse_mode=ParseMode.MARKDOWN_V2)
                     else:
                         sent_message = await context.bot.send_video(update.effective_chat.id, video=media_url, caption=caption, parse_mode=ParseMode.MARKDOWN_V2)
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
                    final_message = await processing_message.edit_text(f"*Transcription:*\n\n_{escape_markdown_v2(transcribed_text)}_", parse_mode=ParseMode.MARKDOWN_V2, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]]))
                    await forward_to_admin_if_watched(final_message, context)
                
                user_data = load_user_data(user_id)
                user_data['stats'][task_type] = user_data['stats'].get(task_type, 0) + 1
                save_user_data(user_id, user_data)
                if task_type == 'chat': return AWAITING_PROMPT
                else: return USER_MAIN
        except (httpx.RequestError, ValueError, KeyError, IndexError, json.JSONDecodeError, error.TimedOut) as e:
            logger.error(f"Error on attempt {attempt + 1}: {e}")
            if task_type == 'video':
                logger.error(f"=== DIRECT VIDEO TOOL ERROR ===")
                logger.error(f"Error type: {type(e).__name__}")
                logger.error(f"Error message: {str(e)}")
                if hasattr(e, 'response'):
                    logger.error(f"Response status: {e.response.status_code}")
                    logger.error(f"Response headers: {dict(e.response.headers)}")
                    try:
                        error_body = e.response.json()
                        logger.error(f"Response body: {error_body}")
                    except:
                        logger.error(f"Response text: {e.response.text}")
            if attempt < max_retries - 1:
                await processing_message.edit_text("⏳ Hold on a sec!")
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
    try:
        photo_file = await update.message.photo[-1].get_file()
        temp_filename = os.path.join(TEMP_DIR, f"temp_{uuid.uuid4()}.jpg")
        await photo_file.download_to_drive(temp_filename)
        context.user_data['image_edit_path'] = temp_filename
        return AWAITING_EDIT_PROMPT
    except error.TimedOut:
        await update.message.reply_text("⏳ Telegram is taking a while to process the image. Please try sending it again.")
        return AWAITING_IMAGE_FOR_EDIT

async def voice_mode_start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    voice = query.data.split('_')[-1]
    context.user_data['voice_mode_voice'] = voice
    keyboard = [[InlineKeyboardButton("🧠 Choose Model", callback_data="vm_choose_model")],
                [InlineKeyboardButton("🚀 Use Default", callback_data="vm_use_default")]]
    await safe_edit_message(query, "Next, choose your AI thinking model for this session, or use the default.", reply_markup=InlineKeyboardMarkup(keyboard))
    return SELECTING_VOICE_MODEL_CHOICE

async def voice_model_choice_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    choice = query.data.split('_')[-1]
    if choice == "default":
        context.user_data['voice_mode_model'] = DEFAULT_VOICE_MODE_MODEL
        await safe_edit_message(query, f"🎤 Voice Mode Started with *{context.user_data['voice_mode_voice'].capitalize()}* voice & default model\\.\n\nSend a voice message to begin, or use /exit to stop\\.", parse_mode=ParseMode.MARKDOWN_V2)
        return AWAITING_VOICE_MODE_INPUT
    else:
        context.user_data['voice_mode_setup'] = True
        context.user_data['category'] = 'chat'
        return await show_model_selection(query, context)

async def voice_mode_input_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    await forward_to_admin_if_watched(update.message, context)
    if 'voice_chat_history' not in context.user_data:
        context.user_data['voice_chat_history'] = deque(maxlen=100)
        context.user_data['voice_chat_history'].append({"role": "system", "content": "You are a voice assistant. Your responses must be concise, suitable for voice conversion, and contain no Markdown, code blocks, lists, or special characters. Respond only with plain, spoken-word text."})
    if not check_and_use_credit(user_id):
        await update.message.reply_text("🚫 You're out of credits! Use /redeem or refer a friend. Voice mode stopped.")
        for key in ['voice_mode_voice', 'voice_chat_history']: context.user_data.pop(key, None)
        await start_command(update, context)
        return ConversationHandler.END
    processing_message = await update.message.reply_text("🎙️ Processing voice...")
    temp_filename = os.path.join(TEMP_DIR, f"temp_{uuid.uuid4()}.ogg")
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
                    transcription_response = await client.post(f"{A4F_API_BASE_URL}/audio/transcriptions", headers=headers, files={'file': f}, data={'model': 'provider-6/distil-whisper-large-v3-en'}, timeout=1200)
                transcription_response.raise_for_status()
                transcribed_text = transcription_response.json().get('text')
                if not transcribed_text: raise ValueError("Transcription failed or returned empty text.")
                context.user_data['voice_chat_history'].append({"role": "user", "content": transcribed_text})
                await processing_message.edit_text(f"🤔 Thinking...")
                chat_data = {"model": context.user_data.get('voice_mode_model', DEFAULT_VOICE_MODE_MODEL), "messages": list(context.user_data['voice_chat_history'])}
                headers["Content-Type"] = "application/json"
                chat_response = await client.post(f"{A4F_API_BASE_URL}/chat/completions", headers=headers, json=chat_data, timeout=1200)
                chat_response.raise_for_status()
                json_data = chat_response.json()
                choices = json_data.get('choices', [])
                if choices and len(choices) > 0:
                    ai_response_text = choices[0].get('message', {}).get('content')
                else:
                    ai_response_text = None
                if not ai_response_text:
                    raise ValueError("Chat completion returned empty response.")
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
        except (httpx.RequestError, ValueError, KeyError, IndexError, json.JSONDecodeError, error.TimedOut) as e:
            logger.error(f"Error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                await processing_message.edit_text("⏳ Hold on a sec!")
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

async def exit_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    keys_to_clear = ['voice_mode_voice', 'voice_chat_history',
                     'voice_mode_pro_active', 'voice_mode_pro_context', 'voice_mode_pro_history']
    for key in keys_to_clear:
        if key in context.user_data:
            context.user_data.pop(key)
    await update.message.reply_text("✅ Mode exited. Returning to the main menu.")
    await start_command(update, context)
    return USER_MAIN

async def redeem_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: `/redeem YOUR-CODE`")
        return

    code_to_redeem, user_id, codes = context.args, update.effective_user.id, load_redeem_codes()

    # Fix: handle list input
    if isinstance(code_to_redeem, list):
        code_to_redeem = code_to_redeem[0]  # get the first item if it's a list

    if code_to_redeem in codes and codes[code_to_redeem]["is_active"]:
        credits_to_add = codes[code_to_redeem]["credits"]
        user_data = load_user_data(user_id)
        user_data["credits"] += credits_to_add
        codes[code_to_redeem]["is_active"] = False
        codes[code_to_redeem]["redeemed_by"] = user_id
        save_user_data(user_id, user_data)
        save_redeem_codes(codes)
        await update.message.reply_markdown_v2(f"🎉 Success\\! *{credits_to_add}* credits added\\. New balance: *{user_data['credits']}*\\.")
    else:
        await update.message.reply_text("❌ This code is invalid or has already been used.")

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
        temp_path = os.path.join(TEMP_DIR, f"temp_{uuid.uuid4()}.tmp")
        await file.download_to_drive(temp_path)
        with open(temp_path, 'r', encoding='utf-8', errors='ignore') as f: content = f.read()
        os.remove(temp_path)
        if len(content) > 100000:
            await processing_message.edit_text("❌ Document content is too long to process."); return
        await processing_message.edit_text("📚 Summarizing document...")
        context.user_data.update({'category': 'summarize', 'model': 'provider-1/sonar'})
        context.user_data['last_prompt'] = content
        # Create a mock message to pass to process_task
        mock_message = update.message
        mock_message.text = content
        await process_task(type('obj', (object,), {'effective_message': mock_message, 'effective_user': update.effective_user, 'callback_query': None}), context, 'summarize')
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
    if REASONING_MESSAGES.get("image"):
        await asyncio.sleep(1)
        await processing_message.edit_text(REASONING_MESSAGES.get("image"))
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
                data = { "model": "provider-6/gpt-4.1", "messages": [{"role": "user", "content": creative_brief}] }
                response = await client.post(f"{A4F_API_BASE_URL}/chat/completions", headers=headers, json=data, timeout=1200)
                response.raise_for_status()
                json_data = response.json()
                choices = json_data.get('choices', [])
                final_prompt = None
                if choices and len(choices) > 0:
                    final_prompt = choices[0].get('message', {}).get('content')
                if not final_prompt:
                    raise ValueError("Creative Director AI failed to generate a prompt.")
                await processing_message.edit_text(f"🎨 Painting your masterpiece...\n\n_Final Prompt: {final_prompt}_", parse_mode=ParseMode.MARKDOWN_V2)
                image_data = { "model": "provider-4/imagen-3", "prompt": final_prompt, "size": "1024x1024" }
                image_response = await client.post(f"{A4F_API_BASE_URL}/images/generations", headers=headers, json=image_data, timeout=180)
                image_response.raise_for_status()
                json_data = image_response.json()
                data_list = json_data.get('data', [])
                if not data_list or len(data_list) == 0:
                    raise ValueError("Image generation failed to return data.")
                image_url = data_list[0].get('url')
                if not image_url: 
                    raise ValueError("Image generation failed to return a URL.")
                sent_message = await update.message.reply_photo(photo=image_url, caption=f"_{escape_markdown_v2(final_prompt)}_", parse_mode=ParseMode.MARKDOWN_V2)
                await forward_to_admin_if_watched(sent_message, context)
                await processing_message.delete()
                user_data = load_user_data(user_id)
                user_data['stats']['image'] = user_data['stats'].get('image', 0) + 1
                save_user_data(user_id, user_data)
                return USER_MAIN
        except (httpx.RequestError, ValueError, KeyError, IndexError, json.JSONDecodeError) as e:
            logger.error(f"Error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                await processing_message.edit_text("⏳ Hold on a sec!")
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
                choices = json_data.get('choices', [])
                result_text = None
                if choices and len(choices) > 0:
                    result_text = choices[0].get('message', {}).get('content')
                if not result_text:
                    raise ValueError("Web Pilot returned an empty response.")
                final_message = await processing_message.edit_text(result_text, parse_mode=ParseMode.MARKDOWN)
                await forward_to_admin_if_watched(final_message, context)
                return AWAITING_WEB_PROMPT
        except (httpx.RequestError, ValueError, KeyError, IndexError, json.JSONDecodeError) as e:
            logger.error(f"Error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                await processing_message.edit_text("⏳ Hold on a sec!")
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

# ========== WORKFLOW HANDLERS ==========

async def new_workflow_entry(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Start the workflow creation process."""
    query = update.callback_query
    await query.answer()
    text = ("✍️ **Describe your new agent's workflow\\.**\n\n"
            "Be specific: mention models, the order of operations, and any custom APIs\\. To use an audio file as input, just say 'when I send an audio file'\\.\n\n"
            "*Example*: `When I send an audio file, transcribe it, then use the text to create a video.`")
    await safe_edit_message(query, text, parse_mode=ParseMode.MARKDOWN_V2)
    return GET_WORKFLOW_DESCRIPTION

async def process_workflow_description(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Process the workflow description and design it using AI."""
    description = update.message.text
    user_id = update.effective_user.id
    
    # Check credits for workflow creation (costs 2 credits)
    if not use_multiple_credits(user_id, 2):
        await update.message.reply_text("🚫 You need at least 2 credits to create an agent. Please redeem a code or wait for your daily refill.")
        return await start_command(update, context)
    
    msg = await update.message.reply_text("🧠 Understood. Designing the workflow now... this might take a moment.")
    
    try:
        workflow_design = await design_workflow(description)
        if "error" in workflow_design:
            refund_multiple_credits(user_id, 2)  # Refund on error
            await msg.edit_text(f"❌ Design Failed: {workflow_design['error']}\n\nPlease try describing your workflow again or rephrase your request.")
            return GET_WORKFLOW_DESCRIPTION
        
        context.user_data['new_workflow_design'] = workflow_design
        keyboard = [
            [InlineKeyboardButton("✅ Confirm & Name Agent", callback_data="confirm_creation")], 
            [InlineKeyboardButton("✏️ Redescribe", callback_data="create_new")], 
            [InlineKeyboardButton("❌ Cancel", callback_data="main_menu")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        text = ("🤖 **Workflow Designed\\!**\n\n"
                "Here is the plan I've created\\. Does this look correct\\?\n\n"
                f"```json\n{escape_markdown_v2(json.dumps(workflow_design, indent=2))}\n```")
        await msg.edit_text(text, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN_V2)
        return CONFIRM_WORKFLOW
    except Exception as e:
        refund_multiple_credits(user_id, 2)  # Refund on error
        logger.error(f"Error in workflow design: {e}")
        await msg.edit_text("❌ An error occurred while designing the workflow. Please try again.")
        return GET_WORKFLOW_DESCRIPTION

async def confirm_and_get_name(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Confirm workflow and ask for name."""
    await update.callback_query.answer()
    await safe_edit_message(update.callback_query, "👍 Great\\! Please give your new agent a short, memorable name\\.", parse_mode=ParseMode.MARKDOWN_V2)
    return GET_WORKFLOW_NAME

async def save_workflow(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Save the workflow with the given name."""
    name = update.message.text.strip()
    user_id = update.effective_user.id
    user_workflows = load_workflows(user_id)
    
    if name in user_workflows:
        await update.message.reply_text("An agent with that name already exists\\. Please choose another name\\.", parse_mode=ParseMode.MARKDOWN_V2)
        return GET_WORKFLOW_NAME
    
    # Create workflow with metadata (default: private, free)
    workflow_metadata = create_workflow_metadata(
        name=name,
        workflow_data=context.user_data['new_workflow_design'],
        creator_id=user_id,
        is_public=False,
        price=0
    )
    
    user_workflows[name] = workflow_metadata
    save_workflows(user_id, user_workflows)
    context.user_data['current_workflow_name'] = name
    del context.user_data['new_workflow_design']
    
    # Debug logging
    logger.info(f"Saved workflow '{name}' for user {user_id}. Total workflows: {list(user_workflows.keys())}")
    
    # Ask about privacy settings
    keyboard = [
        [InlineKeyboardButton("🔒 Keep Private", callback_data="privacy_private")],
        [InlineKeyboardButton("🌐 Make Public", callback_data="privacy_public")],
        [InlineKeyboardButton("⏭️ Skip Settings", callback_data="privacy_skip")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        f"✅ Agent '{escape_markdown_v2(name)}' has been saved\\!\n\n🔧 Would you like to configure privacy settings?",
        parse_mode=ParseMode.MARKDOWN_V2,
        reply_markup=reply_markup
    )
    return WORKFLOW_PRIVACY_SETTINGS

async def cancel_workflow_creation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancel workflow creation and refund credits."""
    if update.callback_query:
        await update.callback_query.answer()
    
    # Refund credits if workflow was being created
    if 'new_workflow_design' in context.user_data:
        del context.user_data['new_workflow_design']
        refund_multiple_credits(update.effective_user.id, 2)
    
    return await start_command(update, context)

async def list_workflows(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """List user's workflows with edit and delete options."""
    query = update.callback_query
    await query.answer()
    user_id = update.effective_user.id
    user_workflows = load_workflows(user_id)
    
    # Debug logging
    logger.info(f"User {user_id} workflows: {list(user_workflows.keys()) if user_workflows else 'None'}")
    
    if not user_workflows:
        text = "📚 You don't have any agents yet\\.\n\nPress 'Create New Agent' to build your first one\\!"
        markup = InlineKeyboardMarkup([[InlineKeyboardButton("🚀 Create New Agent", callback_data="act_create_workflow")], [InlineKeyboardButton("🏠 Back to Main Menu", callback_data="main_menu")]])
        await safe_edit_message(query, text, reply_markup=markup, parse_mode=ParseMode.MARKDOWN_V2)
        return USER_MAIN
    
    keyboard = []
    for name, workflow_metadata in user_workflows.items():
        # Handle both old and new format
        if isinstance(workflow_metadata, dict) and "workflow" in workflow_metadata:
            is_public = workflow_metadata.get("is_public", False)
            price = workflow_metadata.get("price", 0)
            edit_history = workflow_metadata.get("edit_history", [])
            
            # Determine status and edit icons
            status_icon = "🌐" if is_public else "🔒"
            ai_edit_icon = ""
            
            # Check if workflow has been AI-edited
            if edit_history:
                ai_edits = [h for h in edit_history if h.get('type') == 'ai_edit']
                if ai_edits:
                    ai_edit_icon = " 🤖"  # AI-edited indicator
                elif len(edit_history) > 0:
                    ai_edit_icon = " ✏️"  # Manually edited indicator
            
            price_text = f" (💰{price}c)" if is_public and price > 0 else " (🆓)" if is_public else ""
            name_display = f"{status_icon} {name}{ai_edit_icon}{price_text}"
        else:
            name_display = f"🔒 {name}"
        
        # Main row with run button
        keyboard.append([InlineKeyboardButton(f"▶️ {name_display}", callback_data=f"select_workflow:{name}")])
        
        # Action buttons row
        action_buttons = [
            InlineKeyboardButton("✏️ Edit", callback_data=f"edit_workflow:{name}"),
            InlineKeyboardButton("📄 JSON", callback_data=f"view_json:{name}"),
            InlineKeyboardButton("🗑️ Delete", callback_data=f"delete_workflow:{name}")
        ]
        
        # Add edit history button if available
        if isinstance(workflow_metadata, dict) and workflow_metadata.get("edit_history"):
            action_buttons.insert(2, InlineKeyboardButton("📝 History", callback_data=f"edit_history:{name}"))
        
        keyboard.append(action_buttons)
    
    keyboard.append([InlineKeyboardButton("⬅️ Back to Main Menu", callback_data="main_menu")])
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    text = "📚 *Your Agents*\n\n🔒 \\= Private, 🌐 \\= Public, 🤖 \\= AI\\-edited, ✏️ \\= Manually edited\n\nSelect an agent to run it, or use the action buttons\\."
    await safe_edit_message(query, text, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN_V2)
    return USER_MAIN

async def edit_history_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Display edit history for a workflow."""
    query = update.callback_query
    await query.answer()
    workflow_name = query.data.split(":")[1]
    user_id = update.effective_user.id
    
    user_workflows = load_workflows(user_id)
    
    if workflow_name not in user_workflows:
        await safe_edit_message(query, "❌ Workflow not found\\.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]]))
        return USER_MAIN
    
    workflow_metadata = user_workflows[workflow_name]
    edit_history = workflow_metadata.get("edit_history", []) if isinstance(workflow_metadata, dict) else []
    
    if not edit_history:
        text = f"📝 **Edit History: '{escape_markdown_v2(workflow_name)}'**\n\n❌ No edit history available for this workflow\\."
    else:
        text = f"📝 **Edit History: '{escape_markdown_v2(workflow_name)}'**\n\n"
        
        for i, edit in enumerate(edit_history[-5:], 1):  # Show last 5 edits
            edit_type = "🤖 AI Edit" if edit.get('type') == 'ai_edit' else "✏️ Manual Edit"
            timestamp = edit.get('timestamp', 'Unknown time')
            instructions = edit.get('instructions', 'No details available')
            
            # Parse timestamp for better display
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                formatted_time = dt.strftime('%m/%d %H:%M')
            except:
                formatted_time = timestamp[:16] if len(timestamp) > 16 else timestamp
            
            text += f"**{i}\\. {edit_type}** \\({formatted_time}\\)\n"
            dots = "\\.\\.\\." if len(instructions) > 100 else ""
            text += f"â {escape_markdown_v2(instructions[:100])}{dots}\n\n"
    
    keyboard = [
        [InlineKeyboardButton("✏️ Edit Workflow", callback_data=f"edit_workflow:{workflow_name}")],
        [InlineKeyboardButton("🤖 AI Edit", callback_data=f"ai_edit:{workflow_name}")],
        [InlineKeyboardButton("📚 Back to Library", callback_data="act_list_workflows")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await safe_edit_message(query, text, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN_V2)
    return USER_MAIN

async def view_json_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle viewing workflow JSON."""
    query = update.callback_query
    await query.answer()
    workflow_name = query.data.split(":")[1]
    user_id = update.effective_user.id
    
    user_workflows = load_workflows(user_id)
    
    if workflow_name not in user_workflows:
        await safe_edit_message(query, "❌ Workflow not found\\.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]]))
        return USER_MAIN
    
    workflow_metadata = user_workflows[workflow_name]
    workflow = workflow_metadata.get("workflow", workflow_metadata)  # Handle both old and new formats
    
    try:
        workflow_json = json.dumps(workflow, indent=2)
        
        if len(workflow_json) > 4000:
            # Save to temporary file and send as document
            temp_file_path = os.path.join(TEMP_DIR, f"{workflow_name}_workflow.json")
            with open(temp_file_path, 'w', encoding='utf-8') as f:
                f.write(workflow_json)
            
            with open(temp_file_path, 'rb') as f:
                await query.message.reply_document(
                    document=f,
                    filename=f"{workflow_name}_workflow.json",
                    caption=f"📄 JSON for workflow '{workflow_name}'"
                )
            
            # Clean up temp file
            try:
                os.remove(temp_file_path)
            except:
                pass
        else:
            # Send as formatted text
            await query.message.reply_text(
                f"📄 **JSON for '{escape_markdown_v2(workflow_name)}':**\n\n```json\n{workflow_json}\n```",
                parse_mode=ParseMode.MARKDOWN_V2
            )
        
        await safe_edit_message(
            query,
            f"✅ JSON for '{escape_markdown_v2(workflow_name)}' sent\\!",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]]),
            parse_mode=ParseMode.MARKDOWN_V2
        )
        
    except Exception as e:
        logger.error(f"Error viewing JSON for workflow {workflow_name}: {e}")
        await safe_edit_message(
            query,
            f"❌ Error viewing JSON for '{escape_markdown_v2(workflow_name)}'\\. Please try again\\.",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]]),
            parse_mode=ParseMode.MARKDOWN_V2
        )
    
    return USER_MAIN

async def select_workflow(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Select and run a workflow."""
    query = update.callback_query
    await query.answer()
    workflow_name = query.data.split(":")[1]
    user_id = update.effective_user.id
    user_workflows = load_workflows(user_id)
    
    if workflow_name not in user_workflows:
        await safe_edit_message(query, "❌ Workflow not found\\.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]]), parse_mode=ParseMode.MARKDOWN_V2)
        return USER_MAIN
    
    workflow_metadata = user_workflows[workflow_name]
    workflow = workflow_metadata.get("workflow", workflow_metadata)  # Handle both old and new formats
    context.user_data['active_workflow'] = workflow_name
    
    # Calculate credits needed based on workflow steps
    credits_needed = calculate_workflow_credits(workflow)
    
    if workflow.get("requires_input", True):
        prompt_text = (f"🎬 **Agent Activated: '{escape_markdown_v2(workflow_name)}'**\n\n"
                      f"💰 *Running this agent will cost {credits_needed} credits*\n\n"
                      f"This agent is now active\\. Please provide the input to start the workflow \\(e\\.g\\., text or an audio file\\)\\.")
        keyboard = [[InlineKeyboardButton("⬅️ Exit to Main Menu", callback_data="main_menu")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await safe_edit_message(query, prompt_text, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN_V2)
        return USER_MAIN
    else:
        await safe_edit_message(query, f"🎬 **Agent Activated: '{escape_markdown_v2(workflow_name)}'**\n\nThis agent does not require input\\. Starting automatically\\.\\.\\.", parse_mode=ParseMode.MARKDOWN_V2)
        
        # Check credits for workflow execution (dynamic based on steps)
        if not use_multiple_credits(user_id, credits_needed):
            await query.message.reply_text(f"🚫 You need at least {credits_needed} credits to run this agent. Please redeem a code or wait for your daily refill.")
            return await start_command(update, context)
        
        try:
            await execute_workflow(workflow, "", query.message.chat_id, context)
        except Exception as e:
            refund_multiple_credits(user_id, credits_needed)  # Refund on error
            logger.error(f"Error executing workflow: {e}")
            await query.message.reply_text("❌ An error occurred while running the agent. Credits have been refunded.")
        
        # Don't call start_command with callback query, just return to main menu
        return USER_MAIN

async def handle_workflow_text_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle text input for active workflow."""
    active_workflow_name = context.user_data.get('active_workflow')
    active_marketplace_workflow = context.user_data.get('active_marketplace_workflow')
    
    if not active_workflow_name and not active_marketplace_workflow:
        return USER_MAIN
    
    user_id = update.effective_user.id
    
    # Handle marketplace workflow
    if active_marketplace_workflow:
        return await handle_marketplace_workflow_input(update, context)
    
    # Handle regular workflow
    user_workflows = load_workflows(user_id)
    workflow_metadata = user_workflows.get(active_workflow_name)
    
    if not workflow_metadata:
        await update.message.reply_text("❌ Active workflow not found\\.", parse_mode=ParseMode.MARKDOWN_V2)
        context.user_data.pop('active_workflow', None)
        return USER_MAIN
    
    workflow = workflow_metadata.get("workflow", workflow_metadata)  # Handle both old and new formats
    credits_needed = calculate_workflow_credits(workflow)
    
    # Check credits for workflow execution (dynamic based on steps)
    if not use_multiple_credits(user_id, credits_needed):
        await update.message.reply_text(f"🚫 You need at least {credits_needed} credits to run this agent. Please redeem a code or wait for your daily refill.")
        context.user_data.pop('active_workflow', None)
        return await start_command(update, context)
    
    await update.message.reply_text("🚀 Input received\\! Starting the agent workflow\\.\\.\\.", parse_mode=ParseMode.MARKDOWN_V2)
    
    try:
        await execute_workflow(workflow, update.message.text, update.message.chat_id, context)
        context.user_data.pop('active_workflow', None)
    except Exception as e:
        refund_multiple_credits(user_id, credits_needed)  # Refund on error
        logger.error(f"Error executing workflow: {e}")
        await update.message.reply_text("❌ An error occurred while running the agent. Credits have been refunded.")
        context.user_data.pop('active_workflow', None)
    
    return USER_MAIN

async def handle_workflow_file_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle file input for active workflow."""
    active_workflow_name = context.user_data.get('active_workflow')
    if not active_workflow_name:
        return USER_MAIN
    
    user_id = update.effective_user.id
    user_workflows = load_workflows(user_id)
    workflow_metadata = user_workflows.get(active_workflow_name)
    
    if not workflow_metadata:
        await update.message.reply_text("❌ Active workflow not found\\.", parse_mode=ParseMode.MARKDOWN_V2)
        context.user_data.pop('active_workflow', None)
        return USER_MAIN
    
    workflow = workflow_metadata.get("workflow", workflow_metadata)  # Handle both old and new formats
    credits_needed = calculate_workflow_credits(workflow)
    
    # Check credits for workflow execution (dynamic based on steps)
    if not use_multiple_credits(user_id, credits_needed):
        await update.message.reply_text(f"🚫 You need at least {credits_needed} credits to run this agent. Please redeem a code or wait for your daily refill.")
        context.user_data.pop('active_workflow', None)
        return await start_command(update, context)
    
    # Determine file type and handle accordingly
    file_type = None
    file_path = None
    
    if update.message.photo:
        file_type = "photo"
        await update.message.reply_text("🚀 Photo received\\! Starting the agent workflow\\.\\.\\.", parse_mode=ParseMode.MARKDOWN_V2)
        photo_file = await update.message.photo[-1].get_file()
        file_path = os.path.join(TEMP_DIR, f"workflow_{uuid.uuid4()}.jpg")
        await photo_file.download_to_drive(file_path)
        
    elif update.message.video:
        file_type = "video"
        await update.message.reply_text("🚀 Video received\\! Starting the agent workflow\\.\\.\\.", parse_mode=ParseMode.MARKDOWN_V2)
        video_file = await update.message.video.get_file()
        file_path = os.path.join(TEMP_DIR, f"workflow_{uuid.uuid4()}.mp4")
        await video_file.download_to_drive(file_path)
        
    elif update.message.audio or update.message.voice:
        file_type = "audio"
        await update.message.reply_text("🚀 Audio file received\\! Starting the agent workflow\\.\\.\\.", parse_mode=ParseMode.MARKDOWN_V2)
        audio_file = await (update.message.audio or update.message.voice).get_file()
        file_path = os.path.join(TEMP_DIR, f"workflow_{uuid.uuid4()}.ogg")
        await audio_file.download_to_drive(file_path)
        
    else:
        await update.message.reply_text("❌ Unsupported file type\\. Please send a photo, video, or audio file\\.", parse_mode=ParseMode.MARKDOWN_V2)
        return USER_MAIN
    
    try:
        await execute_workflow(workflow, "", update.message.chat_id, context, initial_file_path=file_path)
        context.user_data.pop('active_workflow', None)
    except Exception as e:
        refund_multiple_credits(user_id, credits_needed)  # Refund on error
        logger.error(f"Error executing workflow: {e}")
        await update.message.reply_text("❌ An error occurred while running the agent. Credits have been refunded.")
        context.user_data.pop('active_workflow', None)
    
    return USER_MAIN

# ========== WORKFLOW PRIVACY & MARKETPLACE HANDLERS ==========

async def workflow_privacy_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle workflow privacy settings."""
    query = update.callback_query
    await query.answer()
    action = query.data.split("_")[-1]
    user_id = update.effective_user.id
    workflow_name = context.user_data.get('current_workflow_name')
    
    if not workflow_name:
        await safe_edit_message(query, "❌ Workflow not found\\.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]]))
        return USER_MAIN
    
    user_workflows = load_workflows(user_id)
    workflow_metadata = user_workflows[workflow_name]
    
    if action == "private":
        workflow_metadata["is_public"] = False
        save_workflows(user_id, user_workflows)
        context.user_data.pop('current_workflow_name', None)
        await safe_edit_message(query, f"✅ Agent '{escape_markdown_v2(workflow_name)}' is now **private**\\.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]]), parse_mode=ParseMode.MARKDOWN_V2)
        return USER_MAIN
    
    elif action == "public":
        workflow_metadata["is_public"] = True
        save_workflows(user_id, user_workflows)
        
        # Ask about pricing
        keyboard = [
            [InlineKeyboardButton("🆓 Free", callback_data="price_free")],
            [InlineKeyboardButton("💰 Set Price", callback_data="price_paid")],
            [InlineKeyboardButton("⏭️ Skip Pricing", callback_data="price_skip")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await safe_edit_message(
            query,
            f"✅ Agent '{escape_markdown_v2(workflow_name)}' is now **public**\\!\n\n💰 Would you like to set a price for your agent?",
            reply_markup=reply_markup,
            parse_mode=ParseMode.MARKDOWN_V2
        )
        return WORKFLOW_PRICING_SETTINGS
    
    # Handle pricing callbacks
    elif action == "free":
        workflow_metadata["price"] = 0
        save_workflows(user_id, user_workflows)
        
        # Add to marketplace if public
        if workflow_metadata.get("is_public", False):
            marketplace_workflows = load_marketplace_workflows()
            workflow_id = f"{user_id}_{workflow_name}"
            marketplace_workflows[workflow_id] = {
                'name': workflow_name,
                'workflow': workflow_metadata.get('workflow', workflow_metadata),
                'creator_id': user_id,
                'price': 0,
                'downloads': 0,
                'rating': 0.0,
                'ratings_count': 0
            }
            save_marketplace_workflows(marketplace_workflows)
        
        context.user_data.pop('current_workflow_name', None)
        await safe_edit_message(query, f"✅ Agent '{escape_markdown_v2(workflow_name)}' is now **public and free**\\!", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]]), parse_mode=ParseMode.MARKDOWN_V2)
        return USER_MAIN
    
    elif action == "paid":
        # This will be handled by the price input handler
        await safe_edit_message(
            query,
            f"💰 **Set Price for '{escape_markdown_v2(workflow_name)}'**\n\nEnter the price \\(0\\-100 credits\\):",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("❌ Cancel", callback_data="privacy_skip")]]),
            parse_mode=ParseMode.MARKDOWN_V2
        )
        return WORKFLOW_PRICING_SETTINGS
    
    elif action == "skip":
        context.user_data.pop('current_workflow_name', None)
        await safe_edit_message(query, "✅ Agent saved with default settings\\!", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]]))
        return USER_MAIN

async def workflow_price_input_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle price input for workflow."""
    user_id = update.effective_user.id
    price_text = update.message.text.strip()
    workflow_name = context.user_data.get('current_workflow_name')
    
    if not workflow_name:
        await update.message.reply_text("❌ No workflow selected\\. Please try again\\.", parse_mode=ParseMode.MARKDOWN_V2)
        return USER_MAIN
    
    try:
        price = int(price_text)
        if price < 0 or price > 100:
            await update.message.reply_text("❌ Price must be between 0 and 100 credits\\. Please try again\\.")
            return WORKFLOW_PRICING_SETTINGS
    except ValueError:
        await update.message.reply_text("❌ Please enter a valid number for the price\\.")
        return WORKFLOW_PRICING_SETTINGS
    
    user_workflows = load_workflows(user_id)
    if workflow_name not in user_workflows:
        await update.message.reply_text("❌ Workflow not found\\. Please try again\\.", parse_mode=ParseMode.MARKDOWN_V2)
        return USER_MAIN
    
    workflow_metadata = user_workflows[workflow_name]
    if isinstance(workflow_metadata, dict):
        workflow_metadata['price'] = price
    else:
        # Convert old format to new format
        user_workflows[workflow_name] = {
            'workflow': workflow_metadata,
            'creator_id': user_id,
            'is_public': True,
            'price': price
        }
    
    save_workflows(user_id, user_workflows)
    
    # Update marketplace if public
    if isinstance(workflow_metadata, dict) and workflow_metadata.get('is_public', False):
        marketplace_workflows = load_marketplace_workflows()
        workflow_id = f"{user_id}_{workflow_name}"
        if workflow_id in marketplace_workflows:
            marketplace_workflows[workflow_id]['price'] = price
            save_marketplace_workflows(marketplace_workflows)
    
    context.user_data.pop('current_workflow_name', None)
    
    price_text = "🆓 Free" if price == 0 else f"💰 {price} credits"
    await update.message.reply_text(
        f"✅ Agent '{escape_markdown_v2(workflow_name)}' price set to {price_text}\\!",
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]]),
        parse_mode=ParseMode.MARKDOWN_V2
    )
    return USER_MAIN

async def toggle_privacy_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle toggling workflow privacy."""
    query = update.callback_query
    await query.answer()
    workflow_name = query.data.split(":")[1]
    user_id = update.effective_user.id
    
    user_workflows = load_workflows(user_id)
    if workflow_name not in user_workflows:
        await safe_edit_message(query, "❌ Workflow not found\\.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]]))
        return USER_MAIN
    
    workflow_metadata = user_workflows[workflow_name]
    if isinstance(workflow_metadata, dict):
        current_public = workflow_metadata.get('is_public', False)
        workflow_metadata['is_public'] = not current_public
        is_public = workflow_metadata['is_public']
    else:
        # Convert old format to new format
        user_workflows[workflow_name] = {
            'workflow': workflow_metadata,
            'creator_id': user_id,
            'is_public': True,
            'price': 0
        }
        is_public = True
    
    save_workflows(user_id, user_workflows)
    
    # Update marketplace
    marketplace_workflows = load_marketplace_workflows()
    workflow_id = f"{user_id}_{workflow_name}"
    
    if is_public:
        if workflow_id not in marketplace_workflows:
            marketplace_workflows[workflow_id] = {
                'name': workflow_name,
                'workflow': workflow_metadata.get('workflow', workflow_metadata),
                'creator_id': user_id,
                'price': workflow_metadata.get('price', 0),
                'downloads': 0,
                'rating': 0.0,
                'ratings_count': 0
            }
    else:
        if workflow_id in marketplace_workflows:
            del marketplace_workflows[workflow_id]
    
    save_marketplace_workflows(marketplace_workflows)
    
    status_text = "🌐 Public" if is_public else "🔒 Private"
    await safe_edit_message(
        query,
        f"✅ Agent '{escape_markdown_v2(workflow_name)}' is now {status_text}\\!",
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]]),
        parse_mode=ParseMode.MARKDOWN_V2
    )
    return USER_MAIN

async def change_price_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle changing workflow price."""
    query = update.callback_query
    await query.answer()
    workflow_name = query.data.split(":")[1]
    user_id = update.effective_user.id
    
    user_workflows = load_workflows(user_id)
    if workflow_name not in user_workflows:
        await safe_edit_message(query, "❌ Workflow not found\\.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]]))
        return USER_MAIN
    
    workflow_metadata = user_workflows[workflow_name]
    current_price = workflow_metadata.get('price', 0) if isinstance(workflow_metadata, dict) else 0
    
    context.user_data['current_workflow_name'] = workflow_name
    
    await safe_edit_message(
        query,
        f"💰 **Set Price for '{escape_markdown_v2(workflow_name)}'**\n\n"
        f"Current price: {current_price} credits\n\n"
        f"Enter the new price \\(0\\-100 credits\\):",
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("❌ Cancel", callback_data=f"edit_workflow:{workflow_name}")]]),
        parse_mode=ParseMode.MARKDOWN_V2
    )
    return WORKFLOW_PRICING_SETTINGS

async def edit_description_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle editing workflow description."""
    query = update.callback_query
    await query.answer()
    workflow_name = query.data.split(":")[1]
    user_id = update.effective_user.id
    
    user_workflows = load_workflows(user_id)
    if workflow_name not in user_workflows:
        await safe_edit_message(query, "❌ Workflow not found\\.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]]))
        return USER_MAIN
    
    workflow_metadata = user_workflows[workflow_name]
    current_description = workflow_metadata.get('description', '') if isinstance(workflow_metadata, dict) else ''
    
    context.user_data['current_workflow_name'] = workflow_name
    
    await safe_edit_message(
        query,
        f"✏️ **Edit Description for '{escape_markdown_v2(workflow_name)}'**\n\n"
        f"Current description: {escape_markdown_v2(current_description) if current_description else 'None'}\n\n"
        f"Enter the new description:",
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("❌ Cancel", callback_data=f"edit_workflow:{workflow_name}")]]),
        parse_mode=ParseMode.MARKDOWN_V2
    )
    return EDIT_WORKFLOW_DESCRIPTION

async def workflow_description_input_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle description input for workflow."""
    user_id = update.effective_user.id
    description = update.message.text.strip()
    workflow_name = context.user_data.get('current_workflow_name')
    
    if not workflow_name:
        await update.message.reply_text("❌ No workflow selected\\. Please try again\\.", parse_mode=ParseMode.MARKDOWN_V2)
        return USER_MAIN
    
    user_workflows = load_workflows(user_id)
    if workflow_name not in user_workflows:
        await update.message.reply_text("❌ Workflow not found\\. Please try again\\.", parse_mode=ParseMode.MARKDOWN_V2)
        return USER_MAIN
    
    workflow_metadata = user_workflows[workflow_name]
    if isinstance(workflow_metadata, dict):
        workflow_metadata['description'] = description
    else:
        # Convert old format to new format
        user_workflows[workflow_name] = {
            'workflow': workflow_metadata,
            'creator_id': user_id,
            'is_public': False,
            'price': 0,
            'description': description
        }
    
    save_workflows(user_id, user_workflows)
    
    context.user_data.pop('current_workflow_name', None)
    
    await update.message.reply_text(
        f"✅ Description for agent '{escape_markdown_v2(workflow_name)}' updated\\!",
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]]),
        parse_mode=ParseMode.MARKDOWN_V2
    )
    return USER_MAIN

async def edit_workflow_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle workflow editing."""
    query = update.callback_query
    await query.answer()
    workflow_name = query.data.split(":")[1]
    user_id = update.effective_user.id
    
    user_workflows = load_workflows(user_id)
    
    if workflow_name not in user_workflows:
        await safe_edit_message(query, "❌ Workflow not found\\.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]]))
        return USER_MAIN
    
    workflow_metadata = user_workflows[workflow_name]
    is_public = workflow_metadata.get("is_public", False) if isinstance(workflow_metadata, dict) else False
    price = workflow_metadata.get("price", 0) if isinstance(workflow_metadata, dict) else 0
    
    # Create edit options
    keyboard = [
        [InlineKeyboardButton("🔒 Toggle Privacy", callback_data=f"toggle_privacy:{workflow_name}")],
        [InlineKeyboardButton("💰 Change Price", callback_data=f"change_price:{workflow_name}")],
        [InlineKeyboardButton("✏️ Edit Description", callback_data=f"edit_desc:{workflow_name}")],
        [InlineKeyboardButton("🤖 AI Edit", callback_data=f"ai_edit:{workflow_name}")],
        [InlineKeyboardButton("⬅️ Back to Library", callback_data="act_list_workflows")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    privacy_text = "🌐 Public" if is_public else "🔒 Private"
    price_text = f"💰 {price} credits" if price > 0 else "🆓 Free"
    
    text = (f"✏️ **Edit Agent: '{escape_markdown_v2(workflow_name)}'**\n\n"
            f"**Current Settings:**\n"
            f"Privacy\\: {privacy_text}\n"
            f"Price\\: {price_text}\n\n"
            f"What would you like to edit?")
    
    await safe_edit_message(query, text, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN_V2)
    return USER_MAIN

async def ai_edit_workflow_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle AI-powered workflow editing."""
    query = update.callback_query
    await query.answer()
    workflow_name = query.data.split(":")[1]
    user_id = update.effective_user.id
    
    user_workflows = load_workflows(user_id)
    
    if workflow_name not in user_workflows:
        await safe_edit_message(query, "❌ Workflow not found\\.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]]))
        return USER_MAIN
    
    # Store the workflow name in context for later use
    context.user_data['current_workflow_name'] = workflow_name
    
    text = (f"🤖 **AI Edit: '{escape_markdown_v2(workflow_name)}'**\n\n"
            f"Describe what changes you'd like to make to your workflow\\. You can:\n\n"
            f"• Change the workflow logic or steps\n"
            f"• Modify parameters and settings\n"
            f"• Add or remove functionality\n"
            f"• Update prompts or descriptions\n"
            f"• Change output formats\n\n"
            f"**Examples:**\n"
            f"\\- \"Make the output more creative\"\n"
            f"\\- \"Add a step to generate a summary\"\n"
            f"\\- \"Change video aspect ratio to 16:9\"\n"
            f"\\- \"Make the prompts more detailed\"\n\n"
            f"💭 **What changes would you like to make?**")
    
    keyboard = [[InlineKeyboardButton("❌ Cancel", callback_data=f"edit_workflow:{workflow_name}")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await safe_edit_message(query, text, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN_V2)
    return AWAITING_AI_EDIT_INSTRUCTIONS

async def ai_edit_workflow_input_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Process AI edit instructions for workflow."""
    user_id = update.effective_user.id
    edit_instructions = update.message.text.strip()
    workflow_name = context.user_data.get('current_workflow_name')
    
    if not workflow_name:
        await update.message.reply_text("❌ No workflow selected\\. Please try again\\.", parse_mode=ParseMode.MARKDOWN_V2)
        return USER_MAIN
    
    user_workflows = load_workflows(user_id)
    if workflow_name not in user_workflows:
        await update.message.reply_text("❌ Workflow not found\\. Please try again\\.", parse_mode=ParseMode.MARKDOWN_V2)
        return USER_MAIN
    
    # Check credits
    if not check_and_use_credit(user_id):
        await update.message.reply_text("🚫 You need at least 1 credit to use AI editing\\.", parse_mode=ParseMode.MARKDOWN_V2)
        return USER_MAIN
    
    processing_message = await update.message.reply_text("🤖 AI is analyzing and editing your workflow\\.\\.\\.")
    
    try:
        workflow_metadata = user_workflows[workflow_name]
        current_workflow = workflow_metadata.get('workflow', workflow_metadata) if isinstance(workflow_metadata, dict) else workflow_metadata
        
        # Create edit prompt for AI
        workflow_json = json.dumps(current_workflow, indent=2)
        edit_prompt = f"""You are an AI workflow editor. Please modify the following workflow JSON based on the user's instructions.

Current Workflow:
{workflow_json}

User's Edit Instructions: {edit_instructions}

Please return ONLY the modified JSON workflow with the same structure but incorporating the user's requested changes. Make sure the JSON is valid and maintains the workflow format.

Important guidelines:
- Keep the same overall structure and format
- Ensure all required fields are present
- Make changes that align with the user's instructions
- If adding new steps, use appropriate step types and parameters
- Maintain consistency with existing workflow patterns

Here are the available model types and their parameters:

1.  `"type": "chat"`
    - `model`: (string) Model name.
    - `prompt_template`: (string) The prompt to send. Use `{variable_name}` for placeholders.
    - `output_variable`: (string) The name to store the text output.
    - Available Models: "provider-3/gpt-4", "provider-3/gpt-4.1-mini", "provider-6/o4-mini-high", "provider-6/o4-mini-low", "provider-6/o3-high", "provider-6/o3-medium", "provider-6/o3-low", "provider-3/gpt-4o-mini-search-preview", "provider-6/gpt-4o", "provider-6/gpt-4.1-nano", "provider-6/gpt-4.1-mini", "provider-3/gpt-4.1-nano", "provider-6/o4-mini-medium", "provider-1/deepseek-v3-0324", "provider-6/minimax-m1-40k", "provider-6/kimi-k2", "provider-3/kimi-k2", "provider-6/qwen3-coder-480b-a35b", "provider-3/llama-3.1-405b", "provider-3/qwen-3-235b-a22b-2507", "provider-6/gemini-2.5-flash-thinking", "provider-6/gemini-2.5-flash", "provider-1/llama-3.1-405b-instruct-turbo", "provider-3/llama-3.1-70b", "provider-3/qwen-2.5-coder-32b", "provider-6/kimi-k2-instruct", "provider-6/r1-1776", "provider-6/deepseek-r1-uncensored", "provider-1/deepseek-r1-0528"

2.  `"type": "image_generation"`
    - `output_variable`: (string) The name to store the output image URL.
    - `prompt_from`: (string) Variable containing the prompt text.
    - **Standard Models (Support `n` and `size`):**
        - `model`: (string) One of: "provider-4/imagen-3", "provider-3/FLUX.1-schnell", "provider-1/FLUX.1-schnell", "provider-2/FLUX.1-schnell-v2", "provider-6/sana-1.5", "provider-6/sana-1.5-flash"
        - `n`: (integer, optional) Number of images.
        - `size`: (string, optional) e.g., "1024x1024", "1792x1024", "1024x1792".
    - **Advanced FLUX Models (Do NOT use `n` parameter):**
        - `model`: (string) One of: "provider-6/FLUX.1-kontext-max", "provider-6/FLUX.1-kontext-pro", "provider-6/FLUX.1-kontext-dev", "provider-3/FLUX.1-dev", "provider-6/FLUX.1-dev", "provider-1/FLUX.1.1-pro", "provider-6/FLUX.1-pro", "provider-1/FLUX.1-kontext-pro", "provider-6/FLUX.1-1-pro"
        - `size`: (string, optional) e.g., "1024x1024", "1792x1024", "1024x1792".

3.  `"type": "image_edit"`
    - `model`: (string) Model name.
    - `prompt_from`: (string) Variable containing the edit instruction.
    - `image_from`: (string) Variable containing the URL of the image to edit.
    - `output_variable`: (string) The name to store the edited image URL.
    - Available Models: "provider-6/black-forest-labs-flux-1-kontext-pro", "provider-3/flux-kontext-dev", "provider-6/black-forest-labs-flux-1-kontext-max"

4.  `"type": "video_generation"`
    - `model`: (string) Model name.
    - `prompt_from`: (string) Variable containing the prompt text.
    - `ratio`: (string) e.g., "9:16", "16:9".
    - `duration`: (integer, optional, default: 4) Max 4.
    - `quality`: (string, optional, default: "480p") Max "480p".
    - `output_variable`: (string) The name to store the output video URL.
    - Available Models: "provider-6/wan-2.1"

5.  `"type": "tts"`
    - `model`: (string) Model name.
    - `input_from`: (string) Variable containing the text to convert to speech.
    - `voice`: (string) One of: alloy, echo, fable, onyx, nova, shimmer.
    - `output_variable`: (string) The name to store the output audio file path.
    - Available Models: "provider-3/tts-1", "provider-6/sonic-2", "provider-6/sonic"

6.  `"type": "transcription"`
    - `model`: (string) Model name.
    - `file_path_from`: (string) Variable containing the path of the audio file to transcribe (e.g., `{initial_file_path}`).
    - `output_variable`: (string) The name to store the transcribed text.
    - Available Models: "provider-2/whisper-1", "provider-6/distil-whisper-large-v3-en"

7.  `"type": "custom_api"`
    - `url`: (string) The endpoint URL.
    - `method`: (string) "POST", "GET", etc.
    - `headers`: (dict) e.g., {"Authorization": "Bearer {api_key_variable}", "Content-Type": "application/json"}.
    - `body_template`: (dict/string) The request body as a JSON template.
    - `output_variable`: (string) The name to store the result.
    - `output_parser_path`: (string, optional) e.g., "choices.0.message.content".

    must use this models only for predefined model edits no new model should be added like if the usert says whisper1 you should use provider-3/whisper-1 and not provider-2/whisper-1 or provider3/whisper1."""

        # Use the workflow designer model for AI editing
        api_key = get_random_api_key()
        if not api_key:
            await processing_message.edit_text("❌ No API key available\\. Please try again later\\.")
            refund_credit(user_id)
            return USER_MAIN

        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        json_data = {
            "model": WORKFLOW_DESIGNER_MODEL,
            "messages": [{"role": "user", "content": edit_prompt}],
            "max_tokens": 4000,
            "temperature": 0.3
        }

        async with httpx.AsyncClient() as client:
            response = await make_api_request_with_retry(
                client, f"{A4F_API_BASE_URL}/chat/completions", headers, json_data=json_data, timeout=120
            )
            result = response.json()
            
            if 'choices' in result and len(result['choices']) > 0:
                edited_content = result['choices'][0]['message']['content'].strip()
                
                # Try to extract JSON from the response
                try:
                    # Remove any markdown code block formatting
                    if edited_content.startswith('```'):
                        edited_content = edited_content.split('```')[1]
                        if edited_content.startswith('json'):
                            edited_content = edited_content[4:]
                    edited_content = edited_content.strip()
                    
                    edited_workflow = json.loads(edited_content)
                    
                    # Update the workflow
                    if isinstance(workflow_metadata, dict):
                        workflow_metadata['workflow'] = edited_workflow
                        # Add edit history if not exists
                        if 'edit_history' not in workflow_metadata:
                            workflow_metadata['edit_history'] = []
                        workflow_metadata['edit_history'].append({
                            'timestamp': datetime.now().isoformat(),
                            'instructions': edit_instructions,
                            'type': 'ai_edit'
                        })
                    else:
                        # Convert old format to new format
                        user_workflows[workflow_name] = {
                            'workflow': edited_workflow,
                            'creator_id': user_id,
                            'is_public': False,
                            'price': 0,
                            'edit_history': [{
                                'timestamp': datetime.now().isoformat(),
                                'instructions': edit_instructions,
                                'type': 'ai_edit'
                            }]
                        }
                    
                    save_workflows(user_id, user_workflows)
                    
                    await processing_message.delete()
                    
                    success_text = (f"✅ **Workflow '{escape_markdown_v2(workflow_name)}' successfully edited\\!**\n\n"
                                  f"🎯 **Applied changes:** {escape_markdown_v2(edit_instructions)}\n\n"
                                  f"The workflow has been updated and is ready to use\\. You can find it in your agents library\\.")
                    
                    keyboard = [
                        [InlineKeyboardButton("▶️ Run Workflow", callback_data=f"select_workflow:{workflow_name}")],
                        [InlineKeyboardButton("✏️ Edit Again", callback_data=f"ai_edit:{workflow_name}")],
                        [InlineKeyboardButton("📚 View Library", callback_data="act_list_workflows")],
                        [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]
                    ]
                    reply_markup = InlineKeyboardMarkup(keyboard)
                    
                    await update.message.reply_text(success_text, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN_V2)
                    return USER_MAIN
                    
                except json.JSONDecodeError:
                    await processing_message.edit_text("❌ Failed to parse the edited workflow\\. Please try again with clearer instructions\\.")
                    refund_credit(user_id)
                    return AWAITING_AI_EDIT_INSTRUCTIONS
            else:
                await processing_message.edit_text("❌ Failed to edit workflow\\. Please try again\\.")
                refund_credit(user_id)
                return AWAITING_AI_EDIT_INSTRUCTIONS
                
    except Exception as e:
        logger.error(f"Error in AI workflow editing: {e}")
        await processing_message.edit_text("❌ An error occurred while editing the workflow\\. Please try again\\.")
        refund_credit(user_id)
        return AWAITING_AI_EDIT_INSTRUCTIONS

async def delete_workflow_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle workflow deletion."""
    query = update.callback_query
    await query.answer()
    workflow_name = query.data.split(":")[1]
    user_id = update.effective_user.id
    
    user_workflows = load_workflows(user_id)
    
    if workflow_name not in user_workflows:
        await safe_edit_message(query, "❌ Workflow not found\\.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]]))
        return USER_MAIN
    
    # Create confirmation buttons
    keyboard = [
        [InlineKeyboardButton("✅ Yes, Delete", callback_data=f"confirm_delete:{workflow_name}")],
        [InlineKeyboardButton("❌ Cancel", callback_data="act_list_workflows")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    text = (f"🗑️ **Delete Agent: '{escape_markdown_v2(workflow_name)}'**\n\n"
            f"⚠️ This action cannot be undone\\!\n\n"
            f"Are you sure you want to delete this agent?")
    
    await safe_edit_message(query, text, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN_V2)
    return USER_MAIN

async def confirm_delete_workflow(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Confirm and execute workflow deletion."""
    query = update.callback_query
    await query.answer()
    workflow_name = query.data.split(":")[1]
    user_id = update.effective_user.id
    
    user_workflows = load_workflows(user_id)
    
    if workflow_name in user_workflows:
        # Check if it's public and remove from marketplace
        workflow_metadata = user_workflows[workflow_name]
        if isinstance(workflow_metadata, dict) and workflow_metadata.get("is_public", False):
            # Find and remove from marketplace
            marketplace_workflows = load_marketplace_workflows()
            to_remove = []
            for workflow_id, market_metadata in marketplace_workflows.items():
                if market_metadata.get("creator_id") == user_id and market_metadata.get("name") == workflow_name:
                    to_remove.append(workflow_id)
            
            for workflow_id in to_remove:
                remove_workflow_from_marketplace(workflow_id)
        
        # Remove from user's workflows
        del user_workflows[workflow_name]
        save_workflows(user_id, user_workflows)
        
        await safe_edit_message(
            query,
            f"✅ Agent '{escape_markdown_v2(workflow_name)}' has been deleted\\.",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("📚 Back to Library", callback_data="act_list_workflows")]]),
            parse_mode=ParseMode.MARKDOWN_V2
        )
    else:
        await safe_edit_message(query, "❌ Workflow not found\\.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]]))
    
    return USER_MAIN

async def marketplace_browse(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Browse the marketplace of public workflows."""
    query = update.callback_query
    await query.answer()
    user_id = update.effective_user.id
    
    marketplace_workflows = load_marketplace_workflows()
    
    if not marketplace_workflows:
        text = "🏪 **Agent Hub**\n\nNo public agents available yet\\. Be the first to publish one\\!"
        markup = InlineKeyboardMarkup([[InlineKeyboardButton("🚀 Create Agent", callback_data="act_create_workflow")], [InlineKeyboardButton("🏠 Back to Main Menu", callback_data="main_menu")]])
        await safe_edit_message(query, text, reply_markup=markup, parse_mode=ParseMode.MARKDOWN_V2)
        return USER_MAIN
    
    keyboard = []
    for workflow_id, workflow_metadata in list(marketplace_workflows.items())[:10]:  # Show first 10
        name = workflow_metadata.get("name", "Unnamed")
        price = workflow_metadata.get("price", 0)
        downloads = workflow_metadata.get("downloads", 0)
        price_text = "🆓 Free" if price == 0 else f"💰 {price}c"
        button_text = f"{name} ({price_text}) - {downloads} downloads"
        keyboard.append([InlineKeyboardButton(button_text, callback_data=f"market_view:{workflow_id}")])
        
        # Add install/buy buttons
        if price == 0:
            keyboard.append([InlineKeyboardButton(f"📥 Install '{name}'", callback_data=f"install:{workflow_id}")])
        else:
            keyboard.append([InlineKeyboardButton(f"💰 Buy '{name}' ({price}c)", callback_data=f"buy:{workflow_id}")])
    
    keyboard.append([InlineKeyboardButton("⬅️ Back to Main Menu", callback_data="main_menu")])
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    text = "🏪 **Agent Hub**\n\nDiscover and install public agents created by the community\\!"
    await safe_edit_message(query, text, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN_V2)
    return MARKETPLACE_BROWSE

async def list_owned_workflows(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """List workflows owned/downloaded by the user."""
    query = update.callback_query
    await query.answer()
    user_id = update.effective_user.id
    
    owned_workflows = load_user_owned_workflows(user_id)
    
    if not owned_workflows:
        text = "💼 You haven't downloaded any agents from the Agent Hub yet\\.\n\nVisit the Agent Hub to discover amazing agents\\!"
        markup = InlineKeyboardMarkup([[InlineKeyboardButton("🏪 Agent Hub", callback_data="act_marketplace")], [InlineKeyboardButton("🏠 Back to Main Menu", callback_data="main_menu")]])
        await safe_edit_message(query, text, reply_markup=markup, parse_mode=ParseMode.MARKDOWN_V2)
        return USER_MAIN
    
    keyboard = []
    for workflow_id in owned_workflows:
        # Try to load workflow from marketplace
        marketplace_workflows = load_marketplace_workflows()
        if workflow_id in marketplace_workflows:
            workflow_metadata = marketplace_workflows[workflow_id]
            name = workflow_metadata.get("name", "Unnamed")
            keyboard.append([InlineKeyboardButton(f"▶️ Run '{name}'", callback_data=f"run_owned:{workflow_id}")])
    
    keyboard.append([InlineKeyboardButton("⬅️ Back to Main Menu", callback_data="main_menu")])
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    text = "💼 **My Owned Agents**\n\nThese are agents you've downloaded from the Agent Hub\\."
    await safe_edit_message(query, text, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN_V2)
    return USER_MAIN

async def handle_marketplace_workflow_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle input for marketplace workflows."""
    active_workflow_id = context.user_data.get('active_marketplace_workflow')
    if not active_workflow_id:
        return USER_MAIN
    
    user_id = update.effective_user.id
    
    # Check ownership
    owned_workflows = load_user_owned_workflows(user_id)
    if active_workflow_id not in owned_workflows and not is_admin(user_id):
        await update.message.reply_text("❌ You don't own this agent\\.", parse_mode=ParseMode.MARKDOWN_V2)
        context.user_data.pop('active_marketplace_workflow', None)
        return USER_MAIN
    
    # Load from marketplace
    marketplace_workflows = load_marketplace_workflows()
    if active_workflow_id not in marketplace_workflows:
        await update.message.reply_text("❌ Workflow not found\\.", parse_mode=ParseMode.MARKDOWN_V2)
        context.user_data.pop('active_marketplace_workflow', None)
        return USER_MAIN
    
    workflow_metadata = marketplace_workflows[active_workflow_id]
    workflow = workflow_metadata.get("workflow", {})
    credits_needed = calculate_workflow_credits(workflow)
    
    # Check credits
    if not use_multiple_credits(user_id, credits_needed):
        await update.message.reply_text(f"🚫 You need at least {credits_needed} credits to run this agent. Please redeem a code or wait for your daily refill.")
        context.user_data.pop('active_marketplace_workflow', None)
        return await start_command(update, context)
    
    await update.message.reply_text("🚀 Input received\\! Starting the agent workflow\\.\\.\\.", parse_mode=ParseMode.MARKDOWN_V2)
    
    try:
        # Handle both text and file input
        if update.message.text:
            await execute_workflow(workflow, update.message.text, update.message.chat_id, context)
        elif update.message.audio or update.message.voice:
            audio_file = await (update.message.audio or update.message.voice).get_file()
            file_path = os.path.join(TEMP_DIR, f"marketplace_workflow_{uuid.uuid4()}")
            await audio_file.download_to_drive(file_path)
            await execute_workflow(workflow, "", update.message.chat_id, context, initial_file_path=file_path)
        
        context.user_data.pop('active_marketplace_workflow', None)
    except Exception as e:
        refund_multiple_credits(user_id, credits_needed)
        logger.error(f"Error executing marketplace workflow: {e}")
        await update.message.reply_text("❌ An error occurred while running the agent. Credits have been refunded.")
        context.user_data.pop('active_marketplace_workflow', None)
    
    return USER_MAIN

async def marketplace_view_workflow_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle viewing marketplace workflow details."""
    query = update.callback_query
    await query.answer()
    workflow_id = query.data.split(":")[1]
    user_id = update.effective_user.id
    
    marketplace_workflows = load_marketplace_workflows()
    if workflow_id not in marketplace_workflows:
        await safe_edit_message(query, "❌ Workflow not found\\.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]]))
        return USER_MAIN
    
    workflow_metadata = marketplace_workflows[workflow_id]
    name = workflow_metadata.get("name", "Unnamed")
    description = workflow_metadata.get("description", "No description available")
    price = workflow_metadata.get("price", 0)
    downloads = workflow_metadata.get("downloads", 0)
    creator_id = workflow_metadata.get("creator_id", "Unknown")
    
    # Check if user already owns this workflow
    owned_workflows = load_user_owned_workflows(user_id)
    already_owned = workflow_id in owned_workflows
    
    price_text = "🆓 Free" if price == 0 else f"💰 {price} credits"
    owned_text = "✅ You own this workflow" if already_owned else ""
    
    text = (f"🏪 **Agent Details**\n\n"
            f"**Name:** {escape_markdown_v2(name)}\n"
            f"**Price:** {escape_markdown_v2(price_text)}\n"
            f"**Downloads:** {downloads}\n"
            f"**Description:** {escape_markdown_v2(description)}\n\n"
            f"{escape_markdown_v2(owned_text)}")
    
    keyboard = []
    if not already_owned:
        if price == 0:
            keyboard.append([InlineKeyboardButton(f"📥 Install '{name}'", callback_data=f"install:{workflow_id}")])
        else:
            keyboard.append([InlineKeyboardButton(f"💰 Buy '{name}' ({price}c)", callback_data=f"buy:{workflow_id}")])
    else:
        keyboard.append([InlineKeyboardButton(f"▶️ Run '{name}'", callback_data=f"run_owned:{workflow_id}")])
    
    keyboard.append([InlineKeyboardButton("⬅️ Back to Agent Hub", callback_data="act_marketplace")])
    keyboard.append([InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await safe_edit_message(query, text, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN_V2)
    return USER_MAIN

async def install_workflow_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle installing free workflows from marketplace."""
    query = update.callback_query
    await query.answer()
    workflow_id = query.data.split(":")[1]
    user_id = update.effective_user.id
    
    marketplace_workflows = load_marketplace_workflows()
    if workflow_id not in marketplace_workflows:
        await safe_edit_message(query, "❌ Workflow not found\\.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]]))
        return USER_MAIN
    
    workflow_metadata = marketplace_workflows[workflow_id]
    price = workflow_metadata.get("price", 0)
    
    if price > 0:
        await safe_edit_message(query, "❌ This workflow is not free\\. Use the Buy button instead\\.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]]))
        return USER_MAIN
    
    # Check if user already owns this workflow
    owned_workflows = load_user_owned_workflows(user_id)
    if workflow_id in owned_workflows:
        await safe_edit_message(query, "✅ You already own this workflow\\.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]]))
        return USER_MAIN
    
    # Add to owned workflows
    owned_workflows.append(workflow_id)
    save_user_owned_workflows(user_id, owned_workflows)
    
    # Update download count
    workflow_metadata["downloads"] = workflow_metadata.get("downloads", 0) + 1
    marketplace_workflows[workflow_id] = workflow_metadata
    save_marketplace_workflows(marketplace_workflows)
    
    name = workflow_metadata.get("name", "Unnamed")
    await safe_edit_message(
        query, 
        f"✅ Successfully installed '{escape_markdown_v2(name)}'\\!\n\nYou can now find it in 'My Owned Agents'\\.", 
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]]),
        parse_mode=ParseMode.MARKDOWN_V2
    )
    return USER_MAIN

async def buy_workflow_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle buying paid workflows from marketplace."""
    query = update.callback_query
    await query.answer()
    workflow_id = query.data.split(":")[1]
    user_id = update.effective_user.id
    
    marketplace_workflows = load_marketplace_workflows()
    if workflow_id not in marketplace_workflows:
        await safe_edit_message(query, "❌ Workflow not found\\.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]]))
        return USER_MAIN
    
    workflow_metadata = marketplace_workflows[workflow_id]
    price = workflow_metadata.get("price", 0)
    name = workflow_metadata.get("name", "Unnamed")
    
    if price == 0:
        await safe_edit_message(query, "❌ This workflow is free\\. Use the Install button instead\\.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]]))
        return USER_MAIN
    
    # Check if user already owns this workflow
    owned_workflows = load_user_owned_workflows(user_id)
    if workflow_id in owned_workflows:
        await safe_edit_message(query, "✅ You already own this workflow\\.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]]))
        return USER_MAIN
    
    # Check if user has enough credits
    user_data = load_user_data(user_id)
    user_credits = user_data.get("credits", 0)
    
    if user_credits < price:
        await safe_edit_message(
            query, 
            f"❌ You don't have enough credits\\. You need {price} credits but have {user_credits}\\.", 
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]]),
            parse_mode=ParseMode.MARKDOWN_V2
        )
        return USER_MAIN
    
    # Deduct credits and add to owned workflows
    user_data["credits"] = user_credits - price
    save_user_data(user_id, user_data)
    
    owned_workflows.append(workflow_id)
    save_user_owned_workflows(user_id, owned_workflows)
    
    # Update download count
    workflow_metadata["downloads"] = workflow_metadata.get("downloads", 0) + 1
    marketplace_workflows[workflow_id] = workflow_metadata
    save_marketplace_workflows(marketplace_workflows)
    
    await safe_edit_message(
        query, 
        f"✅ Successfully purchased '{escape_markdown_v2(name)}' for {price} credits\\!\n\nYou can now find it in 'My Owned Agents'\\.", 
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]]),
        parse_mode=ParseMode.MARKDOWN_V2
    )
    return USER_MAIN

async def run_owned_workflow_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle running owned marketplace workflows."""
    query = update.callback_query
    await query.answer()
    workflow_id = query.data.split(":")[1]
    user_id = update.effective_user.id
    
    # Check ownership
    owned_workflows = load_user_owned_workflows(user_id)
    if workflow_id not in owned_workflows:
        await safe_edit_message(query, "❌ You don't own this workflow\\.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]]))
        return USER_MAIN
    
    # Load from marketplace
    marketplace_workflows = load_marketplace_workflows()
    if workflow_id not in marketplace_workflows:
        await safe_edit_message(query, "❌ Workflow not found\\.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]]))
        return USER_MAIN
    
    workflow_metadata = marketplace_workflows[workflow_id]
    workflow = workflow_metadata.get("workflow", {})
    name = workflow_metadata.get("name", "Unnamed")
    
    context.user_data['active_marketplace_workflow'] = workflow_id
    
    # Calculate credits needed based on workflow steps
    credits_needed = calculate_workflow_credits(workflow)
    
    if workflow.get("requires_input", True):
        prompt_text = (f"🎬 **Agent Activated: '{escape_markdown_v2(name)}'**\n\n"
                      f"💰 *Running this agent will cost {credits_needed} credits*\n\n"
                      f"This agent is now active\\. Please provide the input to start the workflow \\(e\\.g\\., text or an audio file\\)\\.")
        keyboard = [[InlineKeyboardButton("⬅️ Exit to Main Menu", callback_data="main_menu")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await safe_edit_message(query, prompt_text, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN_V2)
        return USER_MAIN
    else:
        await safe_edit_message(query, f"🎬 **Agent Activated: '{escape_markdown_v2(name)}'**\n\nThis agent does not require input\\. Starting automatically\\.\\.\\.", parse_mode=ParseMode.MARKDOWN_V2)
        
        # Check credits for workflow execution (dynamic based on steps)
        if not use_multiple_credits(user_id, credits_needed):
            await query.message.reply_text(f"🚫 You need at least {credits_needed} credits to run this agent. Please redeem a code or wait for your daily refill.")
            return USER_MAIN
        
        try:
            await execute_workflow(workflow, "", query.message.chat_id, context)
        except Exception as e:
            refund_multiple_credits(user_id, credits_needed)  # Refund on error
            logger.error(f"Error executing marketplace workflow: {e}")
            await query.message.reply_text("❌ An error occurred while running the agent. Credits have been refunded.")
        
        return USER_MAIN

async def admin_panel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not is_admin(update.effective_user.id):
        if update.callback_query: await update.callback_query.answer("⛔️ This is for administrators only.", show_alert=True)
        else: await update.message.reply_text("⛔️ This command is for administrators only.")
        return USER_MAIN
    text = f"⚙️ *Admin Dashboard*\n\nSelect an action to perform\\."
    maintenance_text = "✅ ON" if _settings.get('maintenance') else "❌ OFF"
    keyboard = [
        [InlineKeyboardButton(f"🛠️ Maintenance: {maintenance_text}", callback_data="admin_maintenance")],
        [InlineKeyboardButton("📊 Bot Stats", callback_data="admin_stats"), InlineKeyboardButton("⚙️ Bot Settings", callback_data="admin_settings")],
        [InlineKeyboardButton("🔑 API Keys", callback_data="admin_keys"), InlineKeyboardButton("👥 Users", callback_data="admin_users")],
        [InlineKeyboardButton("📢 Broadcast", callback_data="admin_broadcast"), InlineKeyboardButton("🎁 Generate Codes", callback_data="admin_gen_codes")],
        [InlineKeyboardButton("📜 Error Log", callback_data="admin_errors")],
        [InlineKeyboardButton("🏠 Back to Main Menu", callback_data="main_menu")]
    ]
    if update.callback_query:
        await update.callback_query.answer()
        await safe_edit_message(update.callback_query, text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.MARKDOWN_V2)
    else:
        await update.message.reply_markdown_v2(text, reply_markup=InlineKeyboardMarkup(keyboard))
    return ADMIN_MAIN

async def admin_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    action = query.data.split('_', 1)[-1]
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
            ram_text = f"{ram.used / 1e9:.2f}/{ram.total / 1e9:.2f} GB ({ram.percent}%)"
            disk_text = f"{disk.used / 1e9:.2f}/{disk.total / 1e9:.2f} GB ({disk.percent}%)"
        except (PermissionError, FileNotFoundError):
            cpu_text = ram_text = disk_text = "N/A (Permission Denied)"
        text = (f"📊 *Bot & System Statistics*\n\n"
                f"*Users:* `{total_users}`\n"
                f"*Active Redeem Codes:* `{active_codes}`\n\n"
                f"💻 *CPU:* {escape_markdown_v2(cpu_text)}\n"
                f"🧠 *RAM:* {escape_markdown_v2(ram_text)}\n"
                f"💽 *Disk:* {escape_markdown_v2(disk_text)}")
        await safe_edit_message(query, text, parse_mode=ParseMode.MARKDOWN_V2, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back", callback_data="admin_back")]]))
    elif action == "settings":
        text = "*⚙️ Bot Settings*\n\n"
        for key, value in _settings.items():
            text += f"`{key}`: `{value}`\n"
        text += "\nSend setting to change in `key value` format \\(e\\.g\\., `daily_credits 15`\\)\\."
        context.user_data['admin_action'] = 'change_setting'
        await safe_edit_message(query, text, parse_mode=ParseMode.MARKDOWN_V2, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back", callback_data="admin_back")]]))
        return ADMIN_AWAITING_INPUT
    elif action == 'users':
        await safe_edit_message(query, "Enter User ID to manage, or send `/all` to list users\\.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back", callback_data="admin_back")]]))
        context.user_data['admin_action'] = 'manage_user'
        return ADMIN_AWAITING_INPUT
    elif action == 'keys':
        keys_status = load_api_keys()
        text = "🔑 *API Key Status*\n\n"
        for i, key_info in enumerate(keys_status):
            status = "✅ Active" if key_info.get('active', True) else "❌ Disabled"
            key_display = escape_markdown_v2(f"{key_info['key'][:12]}...{key_info['key'][-4:]}")
            text += f"`{i}:` {key_display} \\- {status}\n"
        text += "\nSend key index to toggle its status\\."
        context.user_data['admin_action'] = 'toggle_key'
        await safe_edit_message(query, text, parse_mode=ParseMode.MARKDOWN_V2, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back", callback_data="admin_back")]]))
        return ADMIN_AWAITING_INPUT
    elif action == "broadcast":
        await safe_edit_message(query, "Send the message to broadcast to all users\\.")
        context.user_data['admin_action'] = 'broadcast'
        return ADMIN_AWAITING_INPUT
    elif action == "gen_codes":
        await safe_edit_message(query, "Send details in `credits amount` format \\(e\\.g\\., `10 5` to generate 5 codes worth 10 credits each\\)\\.")
        context.user_data['admin_action'] = 'gen_codes'
        return ADMIN_AWAITING_INPUT
    elif action == "errors":
        if not os.path.exists(ERROR_LOG_FILE) or os.path.getsize(ERROR_LOG_FILE) == 0:
            await safe_edit_message(query, "No errors logged yet\\.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back", callback_data="admin_back")]]))
        else:
            with open(ERROR_LOG_FILE, 'rb') as f:
                await query.message.reply_document(document=f, filename="error_log.txt")
            await query.message.reply_text("Error log sent\\. Do you want to clear it?", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("✅ Yes, clear", callback_data="admin_clear_errors"), InlineKeyboardButton("⬅️ Back", callback_data="admin_back")]]))
    elif action == "clear_errors":
        if os.path.exists(ERROR_LOG_FILE):
             open(ERROR_LOG_FILE, 'w').close()
             await safe_edit_message(query, "✅ Error log cleared\\.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back to Admin Panel", callback_data="admin_back")]]))
    elif action == "back":
        return await admin_panel(update, context)
    return ADMIN_MAIN

async def admin_input_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    action = context.user_data.pop('admin_action', None)
    text = update.message.text
    if action == "change_setting":
        try:
            key, value = text.split(maxsplit=1)
            if key in _settings:
                _settings[key] = type(DEFAULT_SETTINGS.get(key, str))(value)
                save_settings()
                await update.message.reply_text(f"✅ Setting `{key}` updated to `{_settings[key]}`.")
            else:
                await update.message.reply_text("Invalid setting key.")
        except Exception as e:
            await update.message.reply_text(f"Invalid format. Use `key value`. Error: {e}")
    elif action == "manage_user":
        if text.lower() == '/all':
             all_users = [f.split('.')[0] for f in os.listdir(USERS_DIR) if f.endswith('.json')]
             user_list = "\n".join([f"`{uid}`" for uid in all_users])
             await update.message.reply_markdown_v2(f"*All User IDs:*\n{user_list or 'No users found.'}")
        else:
            try:
                target_user_id = int(text)
                user_file = os.path.join(USERS_DIR, f"{target_user_id}.json")
                if not os.path.exists(user_file):
                    await update.message.reply_text(f"No data for user ID `{target_user_id}`.")
                else:
                    user_data = load_user_data(target_user_id)
                    info_text = (f"ℹ️ *User Info: `{target_user_id}`*\n\n"
                                 f"*Credits:* `{user_data.get('credits', 'N/A')}`\n*Last Login:* `{user_data.get('last_login_date', 'N/A')}`\n"
                                 f"*Personality:* _{escape_markdown_v2(user_data.get('personality') or 'Not set')}_\n*Banned:* {'Yes' if user_data.get('banned') else 'No'}")
                    keyboard = [
                        [InlineKeyboardButton("💰 Add/Deduct Credits", callback_data=f"admin_user_cred_{target_user_id}"),
                         InlineKeyboardButton("🚫 Ban" if not user_data.get('banned') else "✅ Unban", callback_data=f"admin_user_ban_{target_user_id}")],
                        [InlineKeyboardButton("⬅️ Back to Admin", callback_data="admin_back")]
                    ]
                    await update.message.reply_markdown_v2(info_text, reply_markup=InlineKeyboardMarkup(keyboard))
                    return ADMIN_MAIN
            except ValueError: await update.message.reply_text("Invalid User ID.")
    elif action == "toggle_key":
        try:
            key_index = int(text)
            keys_status = load_api_keys()
            if 0 <= key_index < len(keys_status):
                keys_status[key_index]['active'] = not keys_status[key_index].get('active', True)
                save_api_keys(keys_status)
                await update.message.reply_text(f"✅ Key `{key_index}` has been {'enabled' if keys_status[key_index]['active'] else 'disabled'}.")
            else:
                await update.message.reply_text("Invalid key index.")
        except ValueError: await update.message.reply_text("Invalid index.")
    elif action == "broadcast":
        user_count = len(os.listdir(USERS_DIR))
        context.user_data['broadcast_message_text'] = text
        keyboard = [[InlineKeyboardButton("✅ Yes, send it", callback_data="brod_confirm_yes"), InlineKeyboardButton("❌ No, cancel", callback_data="brod_confirm_no")]]
        await update.message.reply_markdown_v2(f"You are about to send this message to *{user_count}* users\\. Are you sure?", reply_markup=InlineKeyboardMarkup(keyboard))
        return AWAITING_BROADCAST_CONFIRMATION
    elif action == "gen_codes":
        try:
            credits_str, amount_str = text.split()
            credits, amount = int(credits_str), int(amount_str)
            codes, new_codes_text = load_redeem_codes(), []
            for _ in range(amount):
                full_code = f"SYPNS-{uuid.uuid4().hex[:12].upper()}-BOT"
                codes[full_code] = {"credits": credits, "is_active": True}
                new_codes_text.append(f"`{escape_markdown_v2(full_code)}`")
            save_redeem_codes(codes)
            await update.message.reply_markdown_v2(f"✅ Generated *{amount}* codes worth *{credits}* credits:\n\n" + "\n".join(new_codes_text))
        except ValueError: await update.message.reply_text("Invalid format. Use `credits amount`.")
    return await admin_panel(update, context)

async def admin_user_actions_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    parts = query.data.split('_')
    action = parts[1]  # admin_user_cred_123 -> cred
    user_id_str = parts[3]  # admin_user_cred_123 -> 123
    target_user_id = int(user_id_str)
    if action == "cred":
        context.user_data['admin_action'] = f'add_credits_{target_user_id}'
        await safe_edit_message(query, f"How many credits to add/deduct from user `{target_user_id}`? (Use a negative number to deduct)", parse_mode=ParseMode.MARKDOWN_V2)
        return ADMIN_AWAITING_INPUT
    elif action == "ban":
        user_data = load_user_data(target_user_id)
        user_data['banned'] = not user_data.get('banned', False)
        save_user_data(target_user_id, user_data)
        status = 'banned' if user_data['banned'] else 'unbanned'
        await safe_edit_message(query, f"✅ User `{target_user_id}` has been {status}.", parse_mode=ParseMode.MARKDOWN_V2)
        try:
            await context.bot.send_message(chat_id=target_user_id, text=f"You have been {status} by an administrator.")
        except Exception as e:
            logger.warning(f"Could not notify user {target_user_id} of status change: {e}")
    return await admin_panel(update, context)

async def broadcast_confirm_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    if query.data == "brod_confirm_no":
        await safe_edit_message(query, "Broadcast cancelled.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back to Admin Panel", callback_data="admin_back")]]))
        return ADMIN_MAIN
    await safe_edit_message(query, "📢 Sending broadcast...")
    user_files = os.listdir(USERS_DIR)
    success_count, fail_count = 0, 0
    text = context.user_data.pop('broadcast_message_text', None)
    if not text:
        await safe_edit_message(query, "Error: Broadcast message not found.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back to Admin Panel", callback_data="admin_back")]])); return ADMIN_MAIN
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
    await safe_edit_message(query, final_text, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back to Admin Panel", callback_data="admin_back")]]), parse_mode=ParseMode.MARKDOWN_V2)
    return ADMIN_MAIN

async def admin_command_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update.effective_user.id): 
        return

    def escape_markdown(text):
        # Escape all special MarkdownV2 characters
        escape_chars = r'_*[]()~`>#+-=|{}.!'
        return re.sub(f'([{re.escape(escape_chars)}])', r'\\\1', text)

    parts = update.message.text.split()
    if not parts:
        return
    command = parts[0].lstrip('/').lower()

    if command == "msg":
        if len(context.args) < 2:
            await update.message.reply_text("Usage: `/msg <user_id> <message>`")
            return
        try:
            target_user_id = int(context.args[0])
            message_text = " ".join(context.args[1:])
            await context.bot.send_message(
                chat_id=target_user_id,
                text=f"✉️ A message from the admin:\n\n{message_text}"
            )
            await update.message.reply_text(
                f"✅ Message sent to user {target_user_id}.",
                parse_mode=None
            )
        except ValueError:
            await update.message.reply_text("Invalid user ID format. Usage: `/msg <user_id> <message>`")

    elif command == "cred":
        if len(context.args) < 2:
            await update.message.reply_text("Usage: `/cred <user_id> <amount>`")
            return
        try:
            target_user_id = int(context.args[0])
            amount = int(context.args[1])
            user_data = load_user_data(target_user_id)
            user_data['credits'] += amount
            save_user_data(target_user_id, user_data)
            operation = "Gave" if amount >= 0 else "Deducted"
            await update.message.reply_text(
                f"{operation} {abs(amount)} credits to/from user {target_user_id}. New balance: {user_data['credits']}.",
                parse_mode=None
            )
        except ValueError:
            await update.message.reply_text("Invalid format. Usage: `/cred <user_id> <amount>` (use negative for deduction)")

    elif command in ["watch", "unwatch"]:
        if not context.args:
            await update.message.reply_text(f"Usage: `/{command} <user_id>`")
            return
        try:
            target_user_id = int(context.args[0])
            if command == "watch":
                _watched_users.add(target_user_id)
                await update.message.reply_text(
                    escape_markdown(f"👀 Now watching user `{target_user_id}`. All their interactions will be forwarded here."),
                    parse_mode='MarkdownV2'
                )
            else:
                _watched_users.discard(target_user_id)
                await update.message.reply_text(
                    escape_markdown(f"✅ Stopped watching user `{target_user_id}`."),
                    parse_mode='MarkdownV2'
                )
            save_watched_users()
        except ValueError:
            await update.message.reply_text(f"Invalid user ID. Usage: `/{command} <user_id>`")

    elif command == "listwatched":
        if not _watched_users:
            await update.message.reply_text("No users are currently being watched.")
        else:
            text = "*👀 Watched Users:*\n" + "\n".join([f"`{escape_markdown(str(uid))}`" for uid in _watched_users])
            await update.message.reply_text(
                escape_markdown(text),
                parse_mode='MarkdownV2'
            )

    elif command == "getdata":
        if not context.args:
            await update.message.reply_text("Usage: `/getdata <user_id>`")
            return
        try:
            target_user_id = int(context.args[0])
            user_file = os.path.join(USERS_DIR, f"{target_user_id}.json")
            if os.path.exists(user_file):
                with open(user_file, 'rb') as f:
                    await update.message.reply_document(document=f)
            else:
                await update.message.reply_text(
                    escape_markdown(f"No data file found for user `{target_user_id}`."),
                    parse_mode='MarkdownV2'
                )
        except ValueError:
            await update.message.reply_text("Invalid user ID. Usage: `/getdata <user_id>`")

    elif command == "gethistory":
        if not context.args:
            await update.message.reply_text("Usage: `/gethistory <user_id>`")
            return
        try:
            target_user_id = int(context.args[0])
            user_data = load_user_data(target_user_id)
            chat_history = user_data.get('chat_history', [])
            voice_history = user_data.get('voice_chat_history', [])
            if not chat_history and not voice_history:
                await update.message.reply_text(
                    escape_markdown(f"No saved history found for user `{target_user_id}`."),
                    parse_mode='MarkdownV2'
                )
                return
            
            history_text = f"--- Chat History for {target_user_id} ---\n\n"
            if chat_history:
                history_text += "--- Text Chat ---\n"
                for msg in chat_history:
                    history_text += f"[{msg['role'].upper()}]: {escape_markdown(msg['content'])}\n"
            if voice_history:
                history_text += "\n--- Voice Chat ---\n"
                for msg in voice_history:
                    history_text += f"[{msg['role'].upper()}]: {escape_markdown(msg['content'])}\n"
            
            history_file_path = os.path.join(TEMP_DIR, f"history_{target_user_id}.txt")
            with open(history_file_path, "w", encoding="utf-8") as f:
                f.write(history_text)
            with open(history_file_path, "rb") as f:
                await update.message.reply_document(document=f)
            os.remove(history_file_path)
        except ValueError:
            await update.message.reply_text("Invalid user ID. Usage: `/gethistory <user_id>`")

    elif command == "globalstats":
        total_credits = 0
        total_stats = {k: 0 for k in LOADING_MESSAGES.keys()}
        user_files = [f for f in os.listdir(USERS_DIR) if f.endswith('.json')]
        user_activity = {}
        for user_file in user_files:
            user_id_str = user_file.split('.')[0]
            try:
                with open(os.path.join(USERS_DIR, user_file), 'r') as f:
                    data = json.load(f)
                total_credits += data.get('credits', 0)
                user_stats = data.get('stats', {})
                activity_count = sum(user_stats.values())
                for key, usage in user_stats.items():
                    if key in total_stats:
                        total_stats[key] += usage
                if activity_count > 0:
                    user_activity[user_id_str] = activity_count
            except (json.JSONDecodeError, ValueError):
                continue
        
        sorted_users = sorted(user_activity.items(), key=lambda item: item[1], reverse=True)
        top_5_users = sorted_users[:5]
        
        text = (f"🌐 *Global Bot Statistics*\n\n"
                f"💰 *Total Credits in Circulation:* `{total_credits}`\n\n"
                f"📊 *Aggregate Tool Usage:*\n")
        
        sorted_stats = sorted(total_stats.items(), key=lambda item: item[1], reverse=True)
        for key, value in sorted_stats:
            if value > 0:
                text += f"  •  `{key.title()}`: {value}\n"
        
        text += "\n🏆 *Top 5 Most Active Users:*\n"
        if top_5_users:
            for i, (user_id, count) in enumerate(top_5_users, 1):
                text += f"{i}. `{escape_markdown(user_id)}` \\({count} actions\\)\n"
        else:
            text += "_No user activity recorded yet._"
        
        await update.message.reply_text(
            escape_markdown(text),
            parse_mode='MarkdownV2'
        )

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Exception while handling an update:", exc_info=context.error)
    try:
        tb_list = traceback.format_exception(None, context.error, context.error.__traceback__)
        tb_string = "".join(tb_list)
        with open(ERROR_LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(f"--- Time: {datetime.now()} ---\n")
            f.write(f"Update: {update}\n")
            if context.user_data:
                f.write(f"Context User Data: {context.user_data}\n")
            f.write(f"Traceback:\n{tb_string}\n\n")
    except Exception as e:
        logger.error(f"Could not write to error log file: {e}")
    if isinstance(update, Update) and update.effective_message:
        try: await update.effective_message.reply_text("An unexpected error occurred. The admin has been notified and the issue has been logged.")
        except Exception as e: logger.error(f"Failed to send error message to user: {e}")

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    if 'mongo_client' in globals() and mongo_client:
        try:
            mongo_client.close()
            logger.info("MongoDB connection closed")
        except Exception as e:
            logger.error(f"Error closing MongoDB connection: {e}")
    sys.exit(0)

def main() -> None:
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Add timeout protection for setup
    try:
        setup_data_directory()
    except Exception as e:
        logger.error(f"Error during setup: {e}")
        # Continue anyway, the bot can work with file system fallback
    
    if not TELEGRAM_BOT_TOKEN: 
        logger.critical("TOKEN not set!")
        return
    if not INITIAL_A4F_KEYS: 
        logger.critical("API KEYS not set!")
        return
    if not ADMIN_CHAT_ID: 
        logger.warning("ADMIN_CHAT_ID not set!")
    request = HTTPXRequest(read_timeout=60.0, write_timeout=60.0, connect_timeout=30.0, pool_timeout=60.0)
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).concurrent_updates(True).request(request).build()
    conv_handler = ConversationHandler(
        entry_points=[
            CommandHandler("start", start_command),
            CommandHandler("admin", admin_panel),
            CommandHandler("personality", personality_command_entry),
            CallbackQueryHandler(start_command, pattern="^main_menu$")
        ],
        states={
            USER_MAIN: [
                CallbackQueryHandler(action_handler, pattern="^act_"),
                CallbackQueryHandler(start_command, pattern="^main_menu$"),
                CallbackQueryHandler(select_workflow, pattern="^select_workflow:"),
                CallbackQueryHandler(marketplace_browse, pattern="^market_view:"),
                CallbackQueryHandler(run_owned_workflow_handler, pattern="^run_owned:"),
                CallbackQueryHandler(install_workflow_handler, pattern="^install:"),
                CallbackQueryHandler(buy_workflow_handler, pattern="^buy:"),
                CallbackQueryHandler(edit_workflow_handler, pattern="^edit_workflow:"),
                CallbackQueryHandler(ai_edit_workflow_handler, pattern="^ai_edit:"),
                CallbackQueryHandler(edit_history_handler, pattern="^edit_history:"),
                CallbackQueryHandler(delete_workflow_handler, pattern="^delete_workflow:"),
                CallbackQueryHandler(confirm_delete_workflow, pattern="^confirm_delete:"),
                CallbackQueryHandler(toggle_privacy_handler, pattern="^toggle_privacy:"),
                CallbackQueryHandler(change_price_handler, pattern="^change_price:"),
                CallbackQueryHandler(edit_description_handler, pattern="^edit_desc:"),
                CallbackQueryHandler(view_json_handler, pattern="^view_json:"),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_workflow_text_input),
                MessageHandler(filters.AUDIO | filters.VOICE | filters.PHOTO | filters.VIDEO, handle_workflow_file_input)
            ],
            ADMIN_MAIN: [
                CallbackQueryHandler(admin_callback_handler, pattern="^admin_"),
                CallbackQueryHandler(admin_user_actions_handler, pattern="^admin_user_"),
                CallbackQueryHandler(start_command, pattern="^main_menu$"),
            ],
            ADMIN_AWAITING_INPUT: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, admin_input_handler),
                CallbackQueryHandler(admin_panel, pattern="^admin_back"),
            ],
            AWAITING_BROADCAST_CONFIRMATION: [CallbackQueryHandler(broadcast_confirm_handler, pattern="^brod_confirm_")],
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
            SELECTING_VOICE_MODEL_CHOICE: [CallbackQueryHandler(voice_model_choice_handler, pattern="^vm_")],
            AWAITING_VOICE_MODE_INPUT: [MessageHandler(filters.VOICE, voice_mode_input_handler)],
            AWAITING_VOICE_MODE_PRO_INPUT: [
                CallbackQueryHandler(voice_mode_pro_voice_select_handler, pattern='^vm_pro_voice_'),
                MessageHandler(filters.VOICE, voice_mode_pro_input_handler),
                MessageHandler(filters.PHOTO, voice_mode_pro_photo_handler)
            ],
            AWAITING_MIXER_CONCEPT_1: [MessageHandler(filters.TEXT & ~filters.COMMAND, mixer_concept_1_handler)],
            AWAITING_MIXER_CONCEPT_2: [MessageHandler(filters.TEXT & ~filters.COMMAND, mixer_concept_2_handler)],
            AWAITING_WEB_PROMPT: [MessageHandler(filters.TEXT & ~filters.COMMAND, web_pilot_handler)],
            GET_WORKFLOW_DESCRIPTION: [MessageHandler(filters.TEXT & ~filters.COMMAND, process_workflow_description)],
            GET_WORKFLOW_NAME: [MessageHandler(filters.TEXT & ~filters.COMMAND, save_workflow)],
            CONFIRM_WORKFLOW: [
                CallbackQueryHandler(confirm_and_get_name, pattern="^confirm_creation$"), 
                CallbackQueryHandler(new_workflow_entry, pattern="^create_new$"),
                CallbackQueryHandler(cancel_workflow_creation, pattern="^main_menu$")
            ],
            WORKFLOW_PRIVACY_SETTINGS: [
                CallbackQueryHandler(workflow_privacy_handler, pattern="^privacy_")
            ],
            WORKFLOW_PRICING_SETTINGS: [
                CallbackQueryHandler(workflow_privacy_handler, pattern="^price_"),
                MessageHandler(filters.TEXT & ~filters.COMMAND, workflow_price_input_handler)
            ],
            MARKETPLACE_BROWSE: [
                CallbackQueryHandler(marketplace_browse, pattern="^act_marketplace$"),
                CallbackQueryHandler(marketplace_view_workflow_handler, pattern="^market_view:"),
                CallbackQueryHandler(install_workflow_handler, pattern="^install:"),
                CallbackQueryHandler(buy_workflow_handler, pattern="^buy:"),
                CallbackQueryHandler(run_owned_workflow_handler, pattern="^run_owned:")
            ],
            AWAITING_AI_EDIT_INSTRUCTIONS: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, ai_edit_workflow_input_handler),
                CallbackQueryHandler(edit_workflow_handler, pattern="^edit_workflow:")
            ],
            EDIT_WORKFLOW_DESCRIPTION: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, workflow_description_input_handler),
                CallbackQueryHandler(edit_workflow_handler, pattern="^edit_workflow:")
            ],

        },
        fallbacks=[CommandHandler("start", start_command), CommandHandler("cancel", cancel_handler), CommandHandler("exit", exit_command)],
        conversation_timeout=1800, name="main_conversation", persistent=False, allow_reentry=True
    )
    application.add_handler(conv_handler)
    application.add_handler(CommandHandler("help", help_handler))
    application.add_handler(CommandHandler("newchat", new_chat_command))
    application.add_handler(CommandHandler(["me", "mycredits", "profile"], profile_handler))
    application.add_handler(CommandHandler("redeem", redeem_command))
    admin_cmds = ["msg", "cred", "watch", "unwatch", "listwatched", "getdata", "gethistory", "globalstats"]
    application.add_handler(CommandHandler(admin_cmds, admin_command_handler, filters=filters.User(ADMIN_CHAT_ID)))
    application.add_handler(MessageHandler(filters.Document.ALL, document_handler))
    application.add_error_handler(error_handler)
    logger.info("Bot is running...")
    application.run_polling()

if __name__ == "__main__":
    main()