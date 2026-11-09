"""Shared fixtures."""
import numpy as np
import pytest

from ivmon import config
from ivmon.clean import build_clean_quotes
from ivmon.db import get_connection, init_schema
from ivmon.iv import build_iv_quotes
from ivmon.pricing import black_price
from ivmon.svi import build_svi

MODEL_SNAP = "2026-01-02 21:00:00"
MODEL_EXPIRY = "2026-04-03"  # 91 days later -> T = 91/365
MODEL_T = 91 / 365.0
MODEL_FORWARD = 101.0
MODEL_STRIKES = list(range(80, 122, 2))  # 21 strikes, call and put at each


def model_vol(strike: float) -> float:
    """A gentle skewed smile used to generate clean quotes."""
    k = float(np.log(strike / MODEL_FORWARD))
    return 0.25 - 0.25 * k + 0.5 * k * k


@pytest.fixture
def model_con():
    """In-memory DB: clean model chain ingested, cleaned, IV'd and SVI-fitted."""
    con = get_connection(":memory:")
    init_schema(con)
    for k in MODEL_STRIKES:
        for kind in ("C", "P"):
            mid = black_price(MODEL_FORWARD, k, MODEL_T, model_vol(k),
                              config.RISK_FREE_RATE, kind)
            con.execute(
                "INSERT INTO raw_quotes (snapshot_ts, underlying, spot, expiry, "
                "option_type, strike, bid, ask, last, volume, open_interest, "
                "vendor_iv, last_trade_ts) VALUES "
                f"('{MODEL_SNAP}', 'SPY', 100, '{MODEL_EXPIRY}', '{kind}', {k}, "
                f"{mid * 0.98}, {mid * 1.02}, {mid}, 10, 100, {model_vol(k)}, "
                "'2026-01-02 20:00:00')").fetchall()
    build_clean_quotes(con)
    build_iv_quotes(con)
    build_svi(con)
    ts = con.execute("SELECT max(snapshot_ts) FROM svi_params").fetchall()[0][0]
    return con, ts
