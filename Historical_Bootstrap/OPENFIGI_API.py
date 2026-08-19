#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
获取S&P 500成分股的ISIN / CUSIP —— 基于iShares IVV官方持仓CSV

关键点(踩坑记录)
----------------
1) OpenFIGI公开API不会在结果里返回ISIN/CUSIP(授权限制,只能作为输入)。

2) iShares的CSV下载接口本身是对的:
   https://www.ishares.com/us/products/239726/ishares-core-sp-500-etf/
   1467271812596.ajax?fileType=csv&fileName=IVV_holdings&dataType=fund
   多个第三方脚本(R的tidyquant、GitHub上的ishares爬虫项目等)长期使用
   这个链接且证实近期仍然有效。但如果"裸调"这个链接(不带Cookie、不带
   Referer),会被当成爬虫拦截,返回一个普通网页而不是真实CSV数据
   ——之前反复失败就是这个原因。

3) 解决办法: 用requests.Session()先访问一次产品主页,拿到该请求过程中
   种下的Cookie,再带着这个Cookie、以及Referer指向产品主页,去请求CSV
   下载接口,模拟真实浏览器"先看页面、再点下载按钮"的行为。

如果这一版仍然被拦截,大概率是触发了更高级的Bot防护(如Akamai/
Cloudflare的JS挑战),那就只能退回"浏览器手动复制表格"的笨办法了。
"""

import argparse
import io
import sys
from datetime import datetime

import pandas as pd
import requests

PRODUCT_PAGE_URL = "https://www.ishares.com/us/products/239726/ishares-core-sp-500-etf"
IVV_HOLDINGS_URL = (
    "https://www.ishares.com/us/products/239726/"
    "ishares-core-sp-500-etf/1467271812596.ajax"
    "?fileType=csv&fileName=IVV_holdings&dataType=fund"
)

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def cusip_to_isin(cusip: str, country_code: str = "US") -> str:
    """由CUSIP计算ISIN(美股)。已用IBM/Apple真实数据验证正确。"""
    cusip = str(cusip).strip().upper()
    if len(cusip) != 9 or not cusip.isalnum():
        return ""
    body = country_code.upper() + cusip
    digits = ""
    for ch in body:
        digits += ch if ch.isdigit() else str(ord(ch) - ord("A") + 10)
    total = 0
    for i, d in enumerate(reversed(digits)):
        n = int(d)
        if i % 2 == 0:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return f"{country_code.upper()}{cusip}{(10 - (total % 10)) % 10}"


def download_csv_text(session: requests.Session) -> str:
    # 第一步: 像正常用户一样先访问产品主页,拿到Cookie
    session.get(PRODUCT_PAGE_URL, headers=BROWSER_HEADERS, timeout=30)

    # 第二步: 带着Cookie + Referer去请求CSV下载接口
    csv_headers = dict(BROWSER_HEADERS)
    csv_headers["Referer"] = PRODUCT_PAGE_URL
    csv_headers["Accept"] = "text/csv,application/csv,*/*"

    resp = session.get(IVV_HOLDINGS_URL, headers=csv_headers, timeout=30)
    resp.raise_for_status()

    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return resp.content.decode(enc)
        except UnicodeDecodeError:
            continue
    return resp.content.decode("utf-8", errors="replace")


def parse_ishares_holdings(csv_text: str) -> pd.DataFrame:
    lines = csv_text.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Ticker,"):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("NOT_REAL_CSV")  # 用特殊标记,上层判断是否命中反爬
    data_str = "\n".join(lines[header_idx:])
    return pd.read_csv(io.StringIO(data_str), thousands=",")


def clean_and_filter(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    print("\n文件实际列名:", list(df.columns))

    keep_cols = [c for c in ["Ticker", "Name", "Sector", "Asset Class",
                              "Weight (%)", "CUSIP", "ISIN", "SEDOL",
                              "Exchange"]
                 if c in df.columns]
    df = df[keep_cols].copy()

    if "Asset Class" in df.columns:
        df = df[df["Asset Class"].astype(str).str.strip().eq("Equity")]

    for id_col in ("CUSIP", "ISIN"):
        if id_col in df.columns:
            df[id_col] = df[id_col].astype(str).str.strip()

    df = df.dropna(subset=["Ticker"]).reset_index(drop=True)

    if "ISIN" not in df.columns and "CUSIP" in df.columns:
        df["ISIN"] = df["CUSIP"].apply(
            lambda c: cusip_to_isin(c, "US") if c and c != "nan" else ""
        )
        print("(原始CSV没有ISIN列,已根据CUSIP自动推算)")

    return df


def main():
    parser = argparse.ArgumentParser(description="获取S&P 500成分股ISIN/CUSIP")
    parser.add_argument("--output", default="sp500_isin.csv")
    args = parser.parse_args()

    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] 模拟浏览会话下载iShares IVV持仓数据...")

    session = requests.Session()
    try:
        csv_text = download_csv_text(session)
    except requests.RequestException as e:
        print(f"网络请求失败: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        df_raw = parse_ishares_holdings(csv_text)
    except ValueError:
        print(
            "\n仍然被拦截: 拿到的内容不是真实CSV数据(是网页/错误页)。\n"
            "说明iShares对这个下载接口用了更强的反爬机制(如JS挑战),"
            "单纯用requests模拟请求头/Cookie已经绕不过去了。\n"
            "建议改用浏览器里手动复制表格的方式(把Holdings表格全部展开后"
            "选中、复制、粘贴到Excel/文本文件,再发给我处理)。",
            file=sys.stderr,
        )
        sys.exit(1)

    df = clean_and_filter(df_raw)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"\n共获取 {len(df)} 只成分股")
    print(f"已保存至: {args.output}")
    print("\n预览前10行:")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()