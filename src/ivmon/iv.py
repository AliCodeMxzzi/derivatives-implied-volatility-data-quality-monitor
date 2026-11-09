"""Build iv_quotes: our own implied vol for OTM options, next to vendor IV.

Works on one snapshot at a time (default: the latest in clean_quotes).
"""
import logging
import math
from datetime import datetime

import duckdb
import pandas as pd

from ivmon import config
from ivmon.db import get_connection, init_schema
from ivmon.forward import estimate_forward
from ivmon.pricing import implied_vol

log = logging.getLogger(__name__)

FLAGS = ["crossed_market", "zero_bid", "wide_spread", "stale_trade", "low_liquidity"]
OUT_COLUMNS = [
    "snapshot_ts", "underlying", "expiry", "option_type", "strike", "spot",
    "forward", "forward_method", "tte_years", "mid", "rel_spread",
    "vendor_iv", "own_iv", "iv_reason", "iv_diff", *FLAGS,
]


def is_otm(option_type: str, strike: float, forward: float) -> bool:
    """OTM = puts below the forward, calls at or above it."""
    return strike < forward if option_type == "P" else strike >= forward


def quote_iv(row: pd.Series, forward: float, t: float) -> tuple[float, str]:
    """IV for one OTM quote, or (NaN, reason) if the mid can't be trusted."""
    for flag in config.IV_SKIP_FLAGS:
        if bool(row[flag]):
            return float("nan"), f"skipped_{flag}"
    if pd.isna(row["mid"]):
        return float("nan"), "no_mid"
    result = implied_vol(float(row["mid"]), forward, float(row["strike"]), t,
                         config.RISK_FREE_RATE, row["option_type"])
    return result.iv, result.reason


def build_iv_quotes(
    con: duckdb.DuckDBPyConnection, snapshot_ts: datetime | None = None,
) -> int:
    """(Re)build iv_quotes for one snapshot. Returns rows written."""
    if snapshot_ts is None:
        snapshot_ts = con.execute(
            "SELECT max(snapshot_ts) FROM clean_quotes").fetchall()[0][0]
    if snapshot_ts is None:
        log.warning("clean_quotes is empty; nothing to do")
        return 0

    df = con.execute(
        "SELECT * FROM clean_quotes WHERE snapshot_ts = ?", [snapshot_ts]).df()
    out_rows: list[dict] = []
    for (underlying, expiry), group in df.groupby(["underlying", "expiry"]):
        spot = group["spot"].iloc[0]
        t = float(group["tte_years"].iloc[0])
        spot = float(spot) if pd.notna(spot) else float("nan")
        fwd = estimate_forward(group, spot, t)
        for _, row in group.iterrows():
            if not math.isfinite(fwd.forward):
                iv, reason = float("nan"), "no_forward"
            elif not is_otm(row["option_type"], float(row["strike"]), fwd.forward):
                continue  # ITM rows are not used: OTM only
            else:
                iv, reason = quote_iv(row, fwd.forward, t)
            vendor = row["vendor_iv"]
            diff = iv - vendor if (math.isfinite(iv) and pd.notna(vendor)) else float("nan")
            out_rows.append({
                "snapshot_ts": snapshot_ts, "underlying": underlying,
                "expiry": expiry, "option_type": row["option_type"],
                "strike": row["strike"], "spot": spot, "forward": fwd.forward,
                "forward_method": fwd.method, "tte_years": t, "mid": row["mid"],
                "rel_spread": row["rel_spread"], "vendor_iv": vendor,
                "own_iv": iv, "iv_reason": reason, "iv_diff": diff,
                **{f: bool(row[f]) for f in FLAGS},
            })

    out = pd.DataFrame(out_rows, columns=OUT_COLUMNS)
    for col in ["spot", "forward", "tte_years", "mid", "rel_spread",
                "vendor_iv", "own_iv", "iv_diff"]:
        out[col] = out[col].astype("Float64")  # NaN -> NULL in DuckDB

    con.execute("DELETE FROM iv_quotes WHERE snapshot_ts = ?", [snapshot_ts]).fetchall()
    if not out.empty:
        con.register("iv_incoming", out)
        try:
            con.execute(
                f"INSERT INTO iv_quotes SELECT {', '.join(OUT_COLUMNS)} FROM iv_incoming"
            ).fetchall()
        finally:
            con.unregister("iv_incoming")
    log.info("iv_quotes rebuilt for %s: %d rows", snapshot_ts, len(out))
    return len(out)


def reason_counts(con: duckdb.DuckDBPyConnection, snapshot_ts: datetime) -> pd.DataFrame:
    """How many OTM quotes ended in each iv_reason."""
    return con.execute(
        "SELECT iv_reason, count(*) AS n FROM iv_quotes WHERE snapshot_ts = ? "
        "GROUP BY 1 ORDER BY n DESC", [snapshot_ts]).df()


def iv_comparison(con: duckdb.DuckDBPyConnection, snapshot_ts: datetime) -> pd.DataFrame:
    """Own IV vs vendor IV per underlying, in vol points (1.0 = 1% vol).

    Compared only where we solved and the vendor IV looks real (> 1%).
    iv_diff = own - vendor, so a positive number means we are higher.
    """
    sql = """
    WITH otm AS (SELECT * FROM iv_quotes WHERE snapshot_ts = ?),
    cmp AS (SELECT * FROM otm WHERE own_iv IS NOT NULL AND vendor_iv > 0.01)
    SELECT
        o.underlying,
        count(*) AS n_otm,
        count(o.own_iv) AS n_solved,
        (SELECT count(*) FROM cmp c WHERE c.underlying = o.underlying) AS n_compared,
        (SELECT round(100 * median(c.iv_diff), 2) FROM cmp c
          WHERE c.underlying = o.underlying) AS median_diff,
        (SELECT round(100 * median(abs(c.iv_diff)), 2) FROM cmp c
          WHERE c.underlying = o.underlying) AS median_abs_diff,
        (SELECT round(100 * quantile_cont(abs(c.iv_diff), 0.9), 2) FROM cmp c
          WHERE c.underlying = o.underlying) AS p90_abs_diff
    FROM otm o
    GROUP BY o.underlying
    ORDER BY o.underlying
    """
    return con.execute(sql, [snapshot_ts]).df()


def main() -> None:
    from ivmon.clean import build_clean_quotes

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    con = get_connection()
    init_schema(con)
    build_clean_quotes(con)
    ts = con.execute("SELECT max(snapshot_ts) FROM clean_quotes").fetchall()[0][0]
    build_iv_quotes(con, ts)
    pd.set_option("display.width", 200)
    print(f"\nSnapshot: {ts}\n\nIV outcome per reason code:")
    print(reason_counts(con, ts).to_string(index=False))
    print("\nForward method per expiry:")
    print(con.execute(
        "SELECT forward_method, count(DISTINCT (underlying, expiry)) AS n_expiries "
        "FROM iv_quotes WHERE snapshot_ts = ? GROUP BY 1", [ts]).df().to_string(index=False))
    print("\nOwn IV vs vendor IV (vol points):")
    print(iv_comparison(con, ts).to_string(index=False))


if __name__ == "__main__":
    main()
