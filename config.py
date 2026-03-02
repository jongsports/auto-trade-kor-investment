import os
import yaml
from pathlib import Path
import logging
from dotenv import load_dotenv
from datetime import datetime

# 1. 민감한 정보(.env 파일) 로드
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(env_path)

# 2. 일반 설정값(config.yaml) 로드
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config_dir", "config.yaml")

def load_yaml_config(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

config = load_yaml_config(CONFIG_PATH)

# ===============================================
# 민감한 정보 (.env 에서 로드됨)
# ===============================================
APP_KEY = os.getenv("APP_KEY", "")
APP_SECRET = os.getenv("APP_SECRET", "")
ACCOUNT_NUMBER = os.getenv("ACCOUNT_NUMBER", "")
CANO = ACCOUNT_NUMBER # Backwards compatibility
DART_API_KEY = os.getenv("DART_API_KEY", "")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

EMAIL_SMTP_SERVER = os.getenv("EMAIL_SMTP_SERVER", "")
EMAIL_SENDER = os.getenv("EMAIL_SENDER", "")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")
EMAIL_RECIPIENTS = os.getenv("EMAIL_RECIPIENTS", "")

# ===============================================
# 일반 설정 정보 (config.yaml 에서 로드됨)
# ===============================================

# API 카테고리
api_config = config.get("api", {})
DEMO_MODE = api_config.get("demo_mode", True)

# Trading 카테고리
trading_config = config.get("trading", {})
MAX_POSITION_SIZE = float(trading_config.get("max_position_size", 0.1))
MAX_DAILY_LOSS = float(trading_config.get("max_daily_loss", 0.02))
STOP_LOSS_PCT = float(trading_config.get("stop_loss_pct", 0.05))
VOLATILITY_LOOKBACK = int(trading_config.get("volatility_lookback", 20))
MAX_STOCK_RATIO = float(trading_config.get("max_stock_ratio", 0.1))
MAX_INVESTMENT_RATIO = float(trading_config.get("max_investment_ratio", 0.2))
LOSS_CUT_RATIO = float(trading_config.get("loss_cut_ratio", 0.02))
PROFIT_CUT_RATIO = float(trading_config.get("profit_cut_ratio", 0.05))
MAX_STOCK_COUNT = int(trading_config.get("max_stock_count", 3))
MAX_HOLD_DAYS = int(trading_config.get("max_hold_days", 3))

# Screening 카테고리
screen_config = config.get("screening", {})
MIN_MARKET_CAP = int(screen_config.get("min_market_cap", 100000000000))
MIN_VOLUME = int(screen_config.get("min_volume", 100000))
MIN_VOLATILITY = float(screen_config.get("min_volatility", 0.01))
MAX_VOLATILITY = float(screen_config.get("max_volatility", 0.5))
MIN_PRICE = int(screen_config.get("min_price", 1000))
MAX_PRICE = int(screen_config.get("max_price", 1000000))
MOMENTUM_DAYS = int(screen_config.get("momentum_days", 3))
MIN_GAP_UP = float(screen_config.get("min_gap_up", 0.02))
MIN_VOLUME_RATIO = float(screen_config.get("min_volume_ratio", 2.5))
MIN_AMOUNT_RATIO = float(screen_config.get("min_amount_ratio", 3.0))
MIN_MA5_RATIO = float(screen_config.get("min_ma5_ratio", 0.03))
# 시초가 갭 필터 (다단계 동적 스크리닝 #8)
OPENING_GAP_DOWN_THRESHOLD = float(screen_config.get("opening_gap_down_threshold", 0.03))
OPENING_GAP_UP_THRESHOLD   = float(screen_config.get("opening_gap_up_threshold",   0.05))

# Notifications 카테고리 (yaml에서 활성화 여부 등만 가져오고, 토큰은 env우선순위 사용)
NOTI_CONFIG = config.get("notifications", {})
TELEGRAM_ENABLED = NOTI_CONFIG.get("telegram", {}).get("enabled", True)
# 만약 yaml 파일에 토큰이 적혀있고, env가 비어있다면 그걸 쓴다.
if not TELEGRAM_TOKEN:
    TELEGRAM_TOKEN = NOTI_CONFIG.get("telegram", {}).get("token", "")
if not TELEGRAM_CHAT_ID:
    TELEGRAM_CHAT_ID = NOTI_CONFIG.get("telegram", {}).get("chat_id", "")

EMAIL_ENABLED = NOTI_CONFIG.get("email", {}).get("enabled", False)
if not EMAIL_SMTP_SERVER:
    EMAIL_SMTP_SERVER = NOTI_CONFIG.get("email", {}).get("smtp_server", "")
if not EMAIL_SENDER:
    EMAIL_SENDER = NOTI_CONFIG.get("email", {}).get("sender", "")
if not EMAIL_PASSWORD:
    EMAIL_PASSWORD = NOTI_CONFIG.get("email", {}).get("password", "")
if not EMAIL_RECIPIENTS:
    EMAIL_RECIPIENTS = NOTI_CONFIG.get("email", {}).get("recipient", "")

EMAIL_SMTP_PORT = int(NOTI_CONFIG.get("email", {}).get("smtp_port", 587))

# 백테스팅 카테고리
bt_config = config.get("backtesting", {})
BACKTEST_START_DATE = bt_config.get("start_date", "2023-01-01")
BACKTEST_END_DATE = bt_config.get("end_date", "2023-12-31")
BACKTEST_INITIAL_CAPITAL = int(bt_config.get("initial_capital", 100000000))
BACKTEST_COMMISSION = float(bt_config.get("commission", 0.00015))
BACKTEST_SLIPPAGE = float(bt_config.get("slippage", 0.0001))

# 로깅 설정
log_config = config.get("logging", {})
LOG_LEVEL = log_config.get("level", "INFO")
LOG_FILE = log_config.get("file", "logs/auto_trade.log")
LOG_MAX_SIZE = int(log_config.get("max_size", 10485760))
LOG_BACKUP_COUNT = int(log_config.get("backup_count", 5))

# 전략 관련 추가 상수 및 기타(config.py 내 자체 정의)
TAKE_PROFIT_RATIO = PROFIT_CUT_RATIO
STOP_LOSS_RATIO = LOSS_CUT_RATIO
TRAILING_STOP = 0.03  
MAX_STOCKS = MAX_STOCK_COUNT
PORTFOLIO_RATIO = MAX_INVESTMENT_RATIO
STOCK_RATIO = MAX_STOCK_RATIO

# Confluence Matrix 가중치
WEIGHT_TECHNICAL = 40
WEIGHT_ORDER_FLOW = 40
WEIGHT_NEWS = 20

# Overnight Betting 설정
OVERNIGHT_BUY_START = "15:10"
OVERNIGHT_BUY_END = "15:20"
OVERNIGHT_SELL_TIME = "09:05"

MORNING_ENTRY_START = "09:00"
MORNING_ENTRY_END = "09:05"
ADDITIONAL_ENTRY_END = "09:10"
MARKET_CLOSE = "15:30"
LUNCH_START_TIME = "11:20"
LUNCH_END_TIME = "13:00"

# Dynamic Screening Settings
DYNAMIC_SCREENING_TIMES = ["09:05", "10:30", "13:30", "14:40"]
MIN_INTRADAY_VOLUME_RATIO = 0.5 # 50% of avg daily volume reached
INTRADAY_MOMENTUM_WEIGHT = 1.5  # Boost score if price > open and volume surging

BACKTEST_MODE = os.getenv("BACKTEST_MODE", "False").lower() == "true"

# 공통 데이터 디렉토리 설정 
DATA_DIR = Path("data")
LOG_DIR = Path("logs")
BACKTEST_DIR = Path("backtest_results")
SCREENING_RESULTS_FILE = DATA_DIR / "screening_results.json"

DATA_DIR.mkdir(exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True) if os.path.dirname(LOG_FILE) else LOG_DIR.mkdir(exist_ok=True)
BACKTEST_DIR.mkdir(exist_ok=True)

# 로깅 매니저
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

def setup_logging():
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, LOG_LEVEL))
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root_logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root_logger.addHandler(console_handler)
    return root_logger

CACHE_CONFIG = {
    "CACHE_TYPE": "filesystem",
    "CACHE_DIR": "cache",
    "CACHE_DEFAULT_TIMEOUT": 1800,
    "CACHE_THRESHOLD": 1000,
}
