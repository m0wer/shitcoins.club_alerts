"""
Script that parses the prices of https://shitcoins.club/, saves them in a csv
file and sends notifications to a telegram channel when the comission is below a
certain threshold.

Uses tenacity, aiogram, aiohttp and BeautifulSoup.

Install dependencies with:

pip install tenacity aiogram aiohttp beautifulsoup4 pydantic pandas plotly kaleido

Run the script once every hour with a cron:

0 * * * * python3 /path/to/main.py

Or a cron to run the docker container:

0 * * * * docker run --rm -v /path/to/data:/data shitcoins-club-parser
"""

import plotly.express as px
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Telegram bot token
TG_TOKEN: str = os.environ.get("TG_TOKEN")

# Telegram channel id
TG_CHANNEL_ID: str = os.environ.get("TG_CHANNEL_ID")

# Threshold for the comission
THRESHOLD: float = float(os.environ.get("THRESHOLD", 2.5))

# Path to the csv file
CSV_PATH: str = os.environ.get("CSV_PATH", "/data/price.csv")

# URL to parse
URL: str = "https://shitcoins.club/getRatesUpdates"

# Timeout for the requests
TIMEOUT: int = 10

# Timeout for the retries
RETRY_TIMEOUT: int = 10

# Number of retries
RETRY_COUNT: int = 3

# Time to wait between retries
RETRY_WAIT: int = 10

# Time to wait between requests
REQUEST_WAIT: int = 10


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
    CryptoCurrency(name="USDC"),
    CryptoCurrency(name="LTC"),
    CryptoCurrency(name="TRX"),
    CryptoCurrency(name="DASH"),
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
    "USDC": (
        "https://api.binance.com/api/v1/ticker/price?symbol=EURUSDT",
        False,
    ),
    "LTC": (
        "https://api.binance.com/api/v1/ticker/price?symbol=LTCEUR",
        True,
    ),
    "TRX": (
        "https://api.binance.com/api/v1/ticker/price?symbol=TRXEUR",
        True,
    ),
}


class Comission(BaseModel):
    buy: float
    sell: float


class Price(BaseModel):
    crypto_currency: CryptoCurrency
    fiat_currency: FiatCurrency
    buy: float
    sell: float
    time: datetime

    async def get_comission(self) -> Comission | None:
        """Get the comission for the price.

        Returns:
            Comission for the price or None if the price is not available
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
                return Comission(
                    buy=round((1 - buy) * 100, 2), sell=round(-(1 - sell) * 100)
                )


async def get_prices() -> list:
    """Get the prices from the website.

    The response is a text/event-stream

    Returns:
        list: list of dicts with the prices
    """
    async with aiohttp.ClientSession() as session:
        async with session.get(URL) as response:
            if response.status != 200:
                logger.error("Error getting prices")
                return []
            prices = []
            async for line in response.content:
                line = line.decode("utf-8").strip()
                if line.startswith("data:"):
                    data = line.replace("data:", "")
                    for e in json.loads(data):
                        if e.get("toCurrency") == "EUR":
                            prices.append(
                                Price(
                                    crypto_currency=CryptoCurrency(
                                        name=e.get("fromCurrency")
                                    ),
                                    fiat_currency=FiatCurrency(
                                        name=e.get("toCurrency")
                                    ),
                                    buy=e.get("rateBid"),
                                    sell=e.get("rateAsk"),
                                    time=datetime.now(),
                                )
                            )
                    break
    return prices


def get_plot(crypto_currency: str | None = None, n_days: int = 30):
    # Read the CSV
    df = pd.read_csv(CSV_PATH)
    if crypto_currency:
        df = df[df["crypto_currency"] == crypto_currency]

    # Filter the last n_days
    df = df[df["time"] >= (datetime.now() - timedelta(days=n_days)).isoformat()]

    # Plot the buy and sell comission graphs
    fig = px.line(
        df,
        x="time",
        y=["comission_buy", "comission_sell"],
        title=f"Comissions {crypto_currency if crypto_currency else ''}",
    )
    return fig


async def main():
    prices = await get_prices()
    # mapping of crypto_currency to previous comission
    previous_comissions: dict[str, Comission | None] = {}
    with open(CSV_PATH, "r") as f:
        # read the last len(prices) lines
        for line in f.readlines()[-len(prices) :]:
            (
                crypto_currency,
                fiat_currency,
                buy,
                sell,
                time,
                comission_buy,
                comission_sell,
            ) = line.strip().split(",")
            try:
                previous_comissions[crypto_currency] = Comission(
                    buy=float(comission_buy), sell=float(comission_sell)
                )
            except ValueError:
                previous_comissions[crypto_currency] = None

    for price in prices:
        comission = await price.get_comission()
        # save results to csv
        with open(CSV_PATH, "a") as f:
            # columns: crypto_currency,fiat_currency,buy,sell,time,comission_buy,comission_sell
            writer = csv.writer(f)
            writer.writerow(
                [
                    price.crypto_currency.name,
                    price.fiat_currency.name,
                    price.buy,
                    price.sell,
                    price.time.isoformat(),
                    comission.buy if comission else None,
                    comission.sell if comission else None,
                ]
            )
        message = f"{price.crypto_currency.name} en {price.fiat_currency.name} a {price.buy:.2f}€ ({comission.buy if comission else 'NA'}%)/{price.sell:.2f}€ ({comission.sell if comission else 'NA'} %)"
        logger.info(message)

        # notify the channel if the comission change since last record in the csv is
        # above the threshold
        if (
            comission
            and previous_comissions.get(price.crypto_currency.name)
            and (
                abs(comission.buy - previous_comissions[price.crypto_currency.name].buy)
                >= THRESHOLD
                or abs(
                    comission.sell
                    - previous_comissions[price.crypto_currency.name].sell
                )
            )
        ):
            fig = get_plot(price.crypto_currency.name)
            bot = Bot(token=TG_TOKEN)
            # write the image to a random temporary file
            fig.write_image("/tmp/plot.png")
            await bot.send_photo(
                chat_id=TG_CHANNEL_ID,
                photo=FSInputFile("/tmp/plot.png"),
                caption=message,
            )
            logger.info(
                f"Message sent to telegram channel: {price.crypto_currency.name}"
            )


if __name__ == "__main__":
    asyncio.run(main())
