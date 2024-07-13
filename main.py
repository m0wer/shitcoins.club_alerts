"""
Script that parses the prices of https://shitcoins.club/, saves them in a csv
file and sends notifications to a telegram channel when the commission is below a
certain threshold.

Uses tenacity, aiogram, aiohttp, BeautifulSoup, pydantic, pandas, plotly, and kaleido.

Install dependencies with:

pip install tenacity aiogram aiohttp beautifulsoup4 pydantic pandas plotly kaleido

Run the script once every hour with a cron:

0 * * * * python3 /path/to/main.py

Or a cron to run the docker container:

0 * * * * docker run --rm -v /path/to/data:/data shitcoins-club-parser
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import asyncio
import csv
import logging
import os
import json
import sys
from datetime import datetime, timedelta
import aiohttp
from aiogram import Bot
from aiogram.types import FSInputFile
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_fixed
from pydantic import BaseModel
import cloudscraper

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
logger = logging.getLogger(__name__)

# Telegram bot token
TG_TOKEN: str = os.environ.get("TG_TOKEN")

# Telegram channel id
TG_CHANNEL_ID: str = os.environ.get("TG_CHANNEL_ID")

# Threshold for the commission
THRESHOLD: float = float(os.environ.get("THRESHOLD", 2.5))

# Path to the csv file
CSV_PATH: str = os.environ.get("CSV_PATH", "/data/price.csv")

# Path to the state file
STATE_PATH: str = os.environ.get("STATE_PATH", "/data/state.json")

# URL to parse
URL: str = "https://shitcoins.club/getRates"

# Timeout for the requests
TIMEOUT: int = 10

# Timeout for the retries
RETRY_TIMEOUT: int = 10

# Number of retries
RETRY_COUNT: int = 3

# Time to wait between retries
RETRY_WAIT: int = 10

# Disable sending the messages to the telegram channel (just record the prices)
DISABLE_SEND: bool = os.environ.get("DISABLE_SEND", False)


class Currency(BaseModel):
    name: str


class CryptoCurrency(Currency):
    pass


class FiatCurrency(Currency):
    pass


FIAT_CURRENCIES: list[FiatCurrency] = [
    FiatCurrency(name="EUR"),
]

CRYPTO_CURRENCIES: list[CryptoCurrency] = [
    CryptoCurrency(name="BTC"),
    CryptoCurrency(name="ETH"),
    CryptoCurrency(name="USDT"),
    # CryptoCurrency(name="USDC"),
    # CryptoCurrency(name="LTC"),
    # CryptoCurrency(name="TRX"),
    # CryptoCurrency(name="DASH"),
]

# mapping from currency to the api url for the eur spot price and a bool indicating
# if the conversion is direct or inverse
SPOT_URL: dict[str, tuple[str, bool]] = {
    "BTC": (
        "https://api.binance.com/api/v1/ticker/price?symbol=BTCEUR",
        True,
    ),
    "ETH": (
        "https://api.binance.com/api/v1/ticker/price?symbol=ETHEUR",
        True,
    ),
    "USDT": (
        "https://api.binance.com/api/v1/ticker/price?symbol=EURUSDT",
        False,
    ),
    # "USDC": (
    #    "https://api.binance.com/api/v1/ticker/price?symbol=EURUSDT",
    #    False,
    # ),
    # "LTC": (
    #    "https://api.binance.com/api/v1/ticker/price?symbol=LTCEUR",
    #    True,
    # ),
    # "TRX": (
    #    "https://api.binance.com/api/v1/ticker/price?symbol=TRXEUR",
    #    True,
    # ),
}


class Commission(BaseModel):
    buy: float
    sell: float


class Price(BaseModel):
    crypto_currency: CryptoCurrency
    fiat_currency: FiatCurrency
    buy: float
    sell: float
    time: datetime

    async def get_commission(self) -> Commission | None:
        """Get the commission for the price.

        Returns:
            Commission for the price or None if the price is not available
        """
        if not self.crypto_currency.name.split("_")[0] in SPOT_URL:
            return None
        spot_url, inverse = SPOT_URL[self.crypto_currency.name.split("_")[0]]
        async with aiohttp.ClientSession() as session:
            async with session.get(spot_url) as response:
                if response.status != 200:
                    return None
                data = await response.json()
                if not "price" in data:
                    return None
                spot_price = float(data["price"])
                if inverse:
                    buy = self.buy / spot_price
                    sell = self.sell / spot_price
                else:
                    buy = self.buy * spot_price
                    sell = self.sell * spot_price
                return Commission(
                    buy=round((1 - buy) * 100, 2), sell=round(-(1 - sell) * 100)
                )


@retry(stop=stop_after_attempt(RETRY_COUNT), wait=wait_fixed(RETRY_WAIT))
async def get_prices() -> list:
    """Get the prices from the website.

    The response is a JSON object

    Returns:
        list: list of dicts with the prices
    """
    scraper = cloudscraper.create_scraper()

    response = scraper.get(URL)
    if response.status_code != 200:
        logger.error("Error getting prices")
        raise Exception("Failed to get prices")

    prices = []
    data = response.json()
    for e in data:
        if e.get("toCurrency") == "EUR":
            prices.append(
                Price(
                    crypto_currency=CryptoCurrency(name=e.get("fromCurrency")["name"]),
                    fiat_currency=FiatCurrency(name=e.get("toCurrency")),
                    buy=e.get("rateBid"),
                    sell=e.get("rateAsk"),
                    time=datetime.now(),
                )
            )

    return prices


def get_plot(crypto_currency: str | None = None, n_days: int = 30):
    # Read the CSV
    df = pd.read_csv(CSV_PATH)

    # Convert the 'time' column to datetime
    df["time"] = pd.to_datetime(df["time"])

    # Filter the DataFrame for the specified cryptocurrency
    if crypto_currency:
        df = df[df["crypto_currency"] == crypto_currency]

    # Filter the last n_days
    cutoff_time = datetime.now() - timedelta(days=n_days)
    df = df[df["time"] >= cutoff_time]

    # Ensure 'commission_buy' and 'commission_sell' columns are not empty
    if (
        df.empty
        or "commission_buy" not in df.columns
        or "commission_sell" not in df.columns
    ):
        logger.error(
            f"No data available for {crypto_currency} in the last {n_days} days."
        )
        return None

    # Calculate median values
    median_buy = df["commission_buy"].median()
    median_sell = df["commission_sell"].median()

    # Create the figure
    fig = go.Figure()

    # Add commission buy line
    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["commission_buy"],
            mode="lines",
            name=f'Buy (Last: {df["commission_buy"].iloc[-1]:.2f}%)',
            line=dict(color="red"),
        )
    )

    # Add commission sell line
    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["commission_sell"],
            mode="lines",
            name=f'Sell (Last: {df["commission_sell"].iloc[-1]:.2f}%)',
            line=dict(color="blue"),
        )
    )

    # Add horizontal lines for median values with median values in the legend
    fig.add_trace(
        go.Scatter(
            x=[df["time"].min(), df["time"].max()],
            y=[median_buy, median_buy],
            mode="lines",
            name=f"Median Buy: {median_buy:.2f}%",
            line=dict(color="rgba(255,0,0,0.5)", dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[df["time"].min(), df["time"].max()],
            y=[median_sell, median_sell],
            mode="lines",
            name=f"Median Sell: {median_sell:.2f}%",
            line=dict(color="rgba(0,0,255,0.5)", dash="dash"),
        )
    )

    # Update layout
    fig.update_layout(
        title=f"Crypto ATM Commissions {crypto_currency if crypto_currency else ''}",
        xaxis_title="Time",
        yaxis_title="Commission (%)",
        legend_title="Commission Types",
    )

    return fig


def load_state():
    """Load the state from the state file."""
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, "r") as f:
            return json.load(f)
    return {}


def save_state(state):
    """Save the state to the state file."""
    with open(STATE_PATH, "w") as f:
        json.dump(state, f)


async def main():
    state = load_state()
    logger.debug(state)
    prices = await get_prices()
    logger.debug(prices)
    # mapping of crypto_currency to previous commission (24h ago)
    previous_commissions: dict[str, Commission | None] = {}
    with open(CSV_PATH, "r") as f:
        for line in f.readlines():
            # skip header line
            if line.startswith("crypto_currency"):
                continue
            (
                crypto_currency,
                fiat_currency,
                buy,
                sell,
                time,
                commission_buy,
                commission_sell,
            ) = line.strip().split(",")
            if datetime.fromisoformat(time) < (datetime.now() - timedelta(days=2)):
                continue
            if datetime.fromisoformat(time) > (datetime.now() - timedelta(days=1)):
                logger.debug("too new")
                break
            logger.debug(line)
            try:
                previous_commissions[crypto_currency] = Commission(
                    buy=float(commission_buy), sell=float(commission_sell)
                )
            except ValueError:
                previous_commissions[crypto_currency] = None

    for price in prices:
        logger.info(f"Price: {price}")
        commission = await price.get_commission()
        logger.info(f"Commission: {commission}")
        # save results to csv
        with open(CSV_PATH, "a") as f:
            # columns: crypto_currency,fiat_currency,buy,sell,time,commission_buy,commission_sell
            writer = csv.writer(f)
            writer.writerow(
                [
                    price.crypto_currency.name,
                    price.fiat_currency.name,
                    price.buy,
                    price.sell,
                    price.time.isoformat(),
                    commission.buy if commission else None,
                    commission.sell if commission else None,
                ]
            )
        message = f"{price.crypto_currency.name} en {price.fiat_currency.name} a {price.buy:.2f}€ ({commission.buy if commission else 'NA'}%)/{price.sell:.2f}€ ({commission.sell if commission else 'NA'} %)"
        logger.info(message)

        if DISABLE_SEND:
            continue

        last_notification_time = state.get(price.crypto_currency.name)
        if (
            commission
            and previous_commissions.get(price.crypto_currency.name)
            and (
                abs(
                    commission.buy
                    - previous_commissions[price.crypto_currency.name].buy
                )
                >= THRESHOLD
                or abs(
                    commission.sell
                    - previous_commissions[price.crypto_currency.name].sell
                )
                >= THRESHOLD
            )
            and (
                not last_notification_time
                or datetime.fromisoformat(last_notification_time)
                < datetime.now() - timedelta(hours=24)
            )
        ):
            logger.debug(f"Getting plot for {price.crypto_currency.name}")
            fig = get_plot(price.crypto_currency.name, 90)
            bot = Bot(token=TG_TOKEN)
            # write the image to a random temporary file
            fig.write_image("/tmp/plot.png")
            logger.info(f"Sending message to telegram channel: {message}")
            await bot.send_photo(
                chat_id=TG_CHANNEL_ID,
                photo=FSInputFile("/tmp/plot.png"),
                caption=message,
            )
            logger.info(
                f"Message sent to telegram channel: {price.crypto_currency.name}"
            )
            state[price.crypto_currency.name] = datetime.now().isoformat()
            save_state(state)


if __name__ == "__main__":
    asyncio.run(main())
