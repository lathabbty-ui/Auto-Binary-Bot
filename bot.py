import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd
import pandas_ta as ta
from tvDatafeed import TvDatafeed, Interval
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# =========================
# ENV CONFIG
# =========================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
CHAT_ID = os.getenv("CHAT_ID", "").strip()

SCAN_INTERVAL_SECONDS = int(os.getenv("SCAN_INTERVAL_SECONDS", "5"))
BARS = int(os.getenv("BARS", "200"))

# Comma-separated pairs, format: SYMBOL:EXCHANGE
# Example: EURUSD:OANDA,GBPUSD:OANDA,AUDUSD:OANDA
PAIRS_ENV = os.getenv(
    "PAIRS",
    "EURUSD:OANDA,GBPUSD:OANDA,AUDUSD:OANDA"
).strip()

TV_USERNAME = os.getenv("TV_USERNAME", "").strip()
TV_PASSWORD = os.getenv("TV_PASSWORD", "").strip()

# =========================
# LOGGING
# =========================
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("render_signal_bot")


@dataclass
class AnalysisResult:
    pair_name: str
    price: float
    signal: str
    entry_timing: str
    confidence: int
    reason: str
    rsi: float
    stoch: float
    macd: float
    macd_signal: float


def parse_pairs(raw: str):
    pairs = []
    for item in raw.split(","):
        item = item.strip()
        if not item or ":" not in item:
            continue
        symbol, exchange = item.split(":", 1)
        pairs.append({"symbol": symbol.strip(), "exchange": exchange.strip()})
    return pairs


PAIRS = parse_pairs(PAIRS_ENV)

if TV_USERNAME and TV_PASSWORD:
    tv = TvDatafeed(username=TV_USERNAME, password=TV_PASSWORD)
else:
    tv = TvDatafeed()

last_sent_signals: Dict[str, str] = {}


def safe_round(value: Optional[float], digits: int = 5) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


def fetch_data(symbol: str, exchange: str) -> Optional[pd.DataFrame]:
    try:
        df = tv.get_hist(
            symbol=symbol,
            exchange=exchange,
            interval=Interval.in_1_minute,
            n_bars=BARS,
        )
        if df is None or df.empty or len(df) < 50:
            return None
        return df.copy()
    except Exception as e:
        logger.exception("Failed fetching data for %s:%s: %s", exchange, symbol, e)
        return None


def compute_indicators(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    try:
        df["ema9"] = ta.ema(df["close"], length=9)
        df["ema21"] = ta.ema(df["close"], length=21)
        df["rsi"] = ta.rsi(df["close"], length=14)

        macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
        df["macd"] = macd["MACD_12_26_9"]
        df["macd_signal"] = macd["MACDs_12_26_9"]
        df["macd_hist"] = macd["MACDh_12_26_9"]

        bb = ta.bbands(df["close"], length=20, std=2.0)
        df["bb_lower"] = bb["BBL_20_2.0"]
        df["bb_mid"] = bb["BBM_20_2.0"]
        df["bb_upper"] = bb["BBU_20_2.0"]

        stoch = ta.stoch(df["high"], df["low"], df["close"], k=14, d=3, smooth_k=3)
        df["stoch_k"] = stoch["STOCHk_14_3_3"]
        df["stoch_d"] = stoch["STOCHd_14_3_3"]

        return df.dropna().copy()
    except Exception as e:
        logger.exception("Indicator calculation failed: %s", e)
        return None


def decide_signal(symbol: str, exchange: str, df: pd.DataFrame) -> Optional[AnalysisResult]:
    if df is None or df.empty or len(df) < 2:
        return None

    last = df.iloc[-1]
    prev = df.iloc[-2]

    price = float(last["close"])
    pair_name = f"{symbol} ({exchange})"

    signal = "NO_TRADE"
    entry_timing = "WAIT_CONFIRMATION"
    confidence = 50
    reason_parts = []

    bullish_trend = last["ema9"] > last["ema21"] and price > last["ema9"]
    bearish_trend = last["ema9"] < last["ema21"] and price < last["ema9"]

    bullish_momentum = last["rsi"] > 50 and last["macd"] > last["macd_signal"]
    bearish_momentum = last["rsi"] < 50 and last["macd"] < last["macd_signal"]

    near_lower_bb = price <= last["bb_lower"] * 1.001
    near_upper_bb = price >= last["bb_upper"] * 0.999

    stoch_oversold = last["stoch_k"] < 20
    stoch_overbought = last["stoch_k"] > 80

    stoch_rising = last["stoch_k"] > prev["stoch_k"]
    stoch_falling = last["stoch_k"] < prev["stoch_k"]

    macd_hist_rising = last["macd_hist"] > prev["macd_hist"]
    macd_hist_falling = last["macd_hist"] < prev["macd_hist"]

    trap_zone = (
        (stoch_oversold and bearish_momentum and near_lower_bb) or
        (stoch_overbought and bullish_momentum and near_upper_bb)
    )

    if trap_zone:
        signal = "NO_TRADE"
        confidence = 78
        entry_timing = "WAIT_CONFIRMATION"
        reason_parts.append("trap zone: exhaustion conflicts with momentum")
    else:
        if bullish_trend and bullish_momentum and near_lower_bb and stoch_oversold and stoch_rising:
            signal = "BUY"
            confidence = 84
            reason_parts.extend([
                "bullish trend",
                "RSI/MACD bullish",
                "lower-band bounce",
                "stochastic rising from oversold",
            ])
            if macd_hist_rising and price > prev["close"]:
                entry_timing = "ENTER_NOW"
                confidence += 6
            else:
                entry_timing = "WAIT_PULLBACK"

        elif bearish_trend and bearish_momentum and near_upper_bb and stoch_overbought and stoch_falling:
            signal = "SELL"
            confidence = 84
            reason_parts.extend([
                "bearish trend",
                "RSI/MACD bearish",
                "upper-band rejection",
                "stochastic falling from overbought",
            ])
            if macd_hist_falling and price < prev["close"]:
                entry_timing = "ENTER_NOW"
                confidence += 6
            else:
                entry_timing = "WAIT_PULLBACK"

        elif bullish_trend and bullish_momentum and last["rsi"] >= 55 and macd_hist_rising:
            signal = "BUY"
            confidence = 72
            entry_timing = "WAIT_PULLBACK"
            reason_parts.append("bullish continuation")

        elif bearish_trend and bearish_momentum and last["rsi"] <= 45 and macd_hist_falling:
            signal = "SELL"
            confidence = 72
            entry_timing = "WAIT_PULLBACK"
            reason_parts.append("bearish continuation")

        else:
            signal = "NO_TRADE"
            confidence = 62
            entry_timing = "WAIT_CONFIRMATION"
            reason_parts.append("conditions not aligned")

    return AnalysisResult(
        pair_name=pair_name,
        price=price,
        signal=signal,
        entry_timing=entry_timing,
        confidence=min(confidence, 99),
        reason=", ".join(reason_parts),
        rsi=float(last["rsi"]),
        stoch=float(last["stoch_k"]),
        macd=float(last["macd"]),
        macd_signal=float(last["macd_signal"]),
    )


def build_message(result: AnalysisResult) -> str:
    emoji = {"BUY": "🟢", "SELL": "🔴", "NO_TRADE": "⚪"}.get(result.signal, "⚪")
    return (
        f"{emoji} SIGNAL ALERT\n\n"
        f"PAIR: {result.pair_name}\n"
        f"PRICE: {safe_round(result.price, 5)}\n"
        f"SIGNAL: {result.signal}\n"
        f"ENTRY: {result.entry_timing}\n"
        f"CONFIDENCE: {result.confidence}%\n\n"
        f"RSI: {safe_round(result.rsi, 2)}\n"
        f"STOCH: {safe_round(result.stoch, 2)}\n"
        f"MACD: {safe_round(result.macd, 5)}\n"
        f"MACD SIGNAL: {safe_round(result.macd_signal, 5)}\n\n"
        f"REASON: {result.reason}"
    )


async def send_text(app: Application, text: str) -> None:
    if not CHAT_ID:
        logger.warning("CHAT_ID missing; skipping Telegram message")
        return
    try:
        await app.bot.send_message(chat_id=CHAT_ID, text=text)
    except Exception as e:
        logger.exception("Telegram send failed: %s", e)


async def scan_once(app: Application) -> None:
    global last_sent_signals

    for item in PAIRS:
        symbol = item["symbol"]
        exchange = item["exchange"]
        key = f"{symbol}:{exchange}"

        df = fetch_data(symbol, exchange)
        if df is None:
            logger.warning("No data for %s", key)
            continue

        df = compute_indicators(df)
        if df is None or df.empty:
            logger.warning("Indicators unavailable for %s", key)
            continue

        result = decide_signal(symbol, exchange, df)
        if result is None:
            continue

        print(
            f"{result.pair_name} | "
            f"price={safe_round(result.price, 5)} | "
            f"signal={result.signal} | "
            f"entry={result.entry_timing} | "
            f"conf={result.confidence}%"
        )

        current_signature = f"{result.signal}:{result.entry_timing}"
        previous_signature = last_sent_signals.get(key)

        if result.signal != "NO_TRADE" and current_signature != previous_signature:
            await send_text(app, build_message(result))
            last_sent_signals[key] = current_signature


async def scanner_loop(app: Application) -> None:
    await send_text(app, "✅ Render multi-pair signal bot started.")
    while True:
        try:
            await scan_once(app)
        except Exception as e:
            logger.exception("Scanner loop error: %s", e)
        await asyncio.sleep(SCAN_INTERVAL_SECONDS)


async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Bot is running.\n\n"
        "/pairs - show configured pairs\n"
        "/scan - run one manual scan\n"
        "/ping - test reply"
    )


async def ping_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("pong ✅")


async def pairs_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    lines = [f"- {p['symbol']} ({p['exchange']})" for p in PAIRS]
    await update.message.reply_text("Configured pairs:\n" + "\n".join(lines))


async def scan_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Running manual scan...")
    await scan_once(context.application)
    await update.message.reply_text("Manual scan finished.")


async def post_init(app: Application) -> None:
    app.create_task(scanner_loop(app))


def validate_env() -> None:
    if not TELEGRAM_TOKEN:
        raise RuntimeError("Missing TELEGRAM_TOKEN environment variable.")
    if not PAIRS:
        raise RuntimeError("No valid PAIRS configured.")


def main() -> None:
    validate_env()

    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("ping", ping_cmd))
    app.add_handler(CommandHandler("pairs", pairs_cmd))
    app.add_handler(CommandHandler("scan", scan_cmd))
    app.post_init = post_init

    logger.info("Starting bot...")
    app.run_polling()


if __name__ == "__main__":
    main()
