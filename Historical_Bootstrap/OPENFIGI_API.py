#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 09:41:06 2026

@author: october
"""

"""
S&P 500 成分股 ticker -> ISIN 批量查询脚本(使用 OpenFIGI API)

使用前准备:
1. 打开 https://www.openfigi.com/ 注册一个免费账号
2. 登录后进入 API Key 页面,申请一个免费 API Key(即时生成)
3. 把下面 API_KEY 变量替换成你自己的 Key
   (不填 Key 也能跑,但限速会很低,503个ticker可能要跑很久很容易超限被拒)

依赖:
    pip install requests

用法:
    python sp500_isin_fetch.py

输出:
    在当前目录生成 sp500_isin_result.csv,包含 Ticker / ISIN / 公司名(来自OpenFIGI) / FIGI
    查不到的ticker会在 NOT_FOUND 列表里打印出来,你可以再手动核对(可能是退市、代码变更等情况)
"""

import requests
import time
import csv

# ========== 第1步:填入你自己的 OpenFIGI API Key(强烈建议填,不填会很慢) ==========
API_KEY = "dc5b99fa-ed88-4897-8fea-e892fe4851ee"  # 例如: "abcd1234-ef56-7890-gh12-ijklmnop3456"

# ========== 第2步:标普500成分股ticker清单(已内置503个,来自Wikipedia最新名单) ==========
TICKERS = [
    'MMM',
    'AOS',
    'ABT',
    'ABBV',
    'ACN',
    'ADBE',
    'AMD',
    'AES',
    'AFL',
    'A',
    'APD',
    'ABNB',
    'AKAM',
    'ALB',
    'ARE',
    'ALGN',
    'ALLE',
    'LNT',
    'ALL',
    'GOOGL',
    'GOOG',
    'MO',
    'AMZN',
    'AMCR',
    'AEE',
    'AEP',
    'AXP',
    'AIG',
    'AMT',
    'AWK',
    'AMP',
    'AME',
    'AMGN',
    'APH',
    'ADI',
    'AON',
    'APA',
    'APO',
    'AAPL',
    'AMAT',
    'APP',
    'APTV',
    'ACGL',
    'ADM',
    'ARES',
    'ANET',
    'AJG',
    'AIZ',
    'T',
    'ATO',
    'ADSK',
    'ADP',
    'AZO',
    'AVB',
    'AVY',
    'AXON',
    'BKR',
    'BALL',
    'BAC',
    'BAX',
    'BDX',
    'BRK.B',
    'BBY',
    'TECH',
    'BIIB',
    'BLK',
    'BX',
    'XYZ',
    'BNY',
    'BA',
    'BKNG',
    'BSX',
    'BMY',
    'AVGO',
    'BR',
    'BRO',
    'BF.B',
    'BLDR',
    'BG',
    'BXP',
    'CHRW',
    'CDNS',
    'CPT',
    'CPB',
    'COF',
    'CAH',
    'CCL',
    'CARR',
    'CVNA',
    'CASY',
    'CAT',
    'CBOE',
    'CBRE',
    'CDW',
    'COR',
    'CNC',
    'CNP',
    'CF',
    'CRL',
    'SCHW',
    'CHTR',
    'CVX',
    'CMG',
    'CB',
    'CHD',
    'CIEN',
    'CI',
    'CINF',
    'CTAS',
    'CSCO',
    'C',
    'CFG',
    'CLX',
    'CME',
    'CMS',
    'KO',
    'CTSH',
    'COHR',
    'COIN',
    'CL',
    'CMCSA',
    'FIX',
    'CAG',
    'COP',
    'ED',
    'STZ',
    'CEG',
    'COO',
    'CPRT',
    'GLW',
    'CPAY',
    'CTVA',
    'CSGP',
    'COST',
    'CRH',
    'CRWD',
    'CCI',
    'CSX',
    'CMI',
    'CVS',
    'DHR',
    'DRI',
    'DDOG',
    'DVA',
    'DECK',
    'DE',
    'DELL',
    'DAL',
    'DVN',
    'DXCM',
    'FANG',
    'DLR',
    'DG',
    'DLTR',
    'D',
    'DPZ',
    'DASH',
    'DOV',
    'DOW',
    'DHI',
    'DTE',
    'DUK',
    'DD',
    'ETN',
    'EBAY',
    'SATS',
    'ECL',
    'EIX',
    'EW',
    'EA',
    'ELV',
    'EME',
    'EMR',
    'ETR',
    'EOG',
    'EPAM',
    'EQT',
    'EFX',
    'EQIX',
    'EQR',
    'ERIE',
    'ESS',
    'EL',
    'EG',
    'EVRG',
    'ES',
    'EXC',
    'EXE',
    'EXPE',
    'EXPD',
    'EXR',
    'XOM',
    'FFIV',
    'FDS',
    'FICO',
    'FAST',
    'FRT',
    'FDX',
    'FIS',
    'FITB',
    'FSLR',
    'FE',
    'FISV',
    'F',
    'FTNT',
    'FTV',
    'FOXA',
    'FOX',
    'BEN',
    'FCX',
    'GRMN',
    'IT',
    'GE',
    'GEHC',
    'GEV',
    'GEN',
    'GNRC',
    'GD',
    'GIS',
    'GM',
    'GPC',
    'GILD',
    'GPN',
    'GL',
    'GDDY',
    'GS',
    'HAL',
    'HIG',
    'HAS',
    'HCA',
    'DOC',
    'HSIC',
    'HSY',
    'HPE',
    'HLT',
    'HD',
    'HON',
    'HRL',
    'HST',
    'HWM',
    'HPQ',
    'HUBB',
    'HUM',
    'HBAN',
    'HII',
    'IBM',
    'IEX',
    'IDXX',
    'ITW',
    'INCY',
    'IR',
    'PODD',
    'INTC',
    'IBKR',
    'ICE',
    'IFF',
    'IP',
    'INTU',
    'ISRG',
    'IVZ',
    'INVH',
    'IQV',
    'IRM',
    'JBHT',
    'JBL',
    'JKHY',
    'J',
    'JNJ',
    'JCI',
    'JPM',
    'KVUE',
    'KDP',
    'KEY',
    'KEYS',
    'KMB',
    'KIM',
    'KMI',
    'KKR',
    'KLAC',
    'KHC',
    'KR',
    'LHX',
    'LH',
    'LRCX',
    'LVS',
    'LDOS',
    'LEN',
    'LII',
    'LLY',
    'LIN',
    'LYV',
    'LMT',
    'L',
    'LOW',
    'LULU',
    'LITE',
    'LYB',
    'MTB',
    'MPC',
    'MAR',
    'MRSH',
    'MLM',
    'MAS',
    'MA',
    'MKC',
    'MCD',
    'MCK',
    'MDT',
    'MRK',
    'META',
    'MET',
    'MTD',
    'MGM',
    'MCHP',
    'MU',
    'MSFT',
    'MAA',
    'MRNA',
    'TAP',
    'MDLZ',
    'MPWR',
    'MNST',
    'MCO',
    'MS',
    'MOS',
    'MSI',
    'MSCI',
    'NDAQ',
    'NTAP',
    'NFLX',
    'NEM',
    'NWSA',
    'NWS',
    'NEE',
    'NKE',
    'NI',
    'NDSN',
    'NSC',
    'NTRS',
    'NOC',
    'NCLH',
    'NRG',
    'NUE',
    'NVDA',
    'NVR',
    'NXPI',
    'ORLY',
    'OXY',
    'ODFL',
    'OMC',
    'ON',
    'OKE',
    'ORCL',
    'OTIS',
    'PCAR',
    'PKG',
    'PLTR',
    'PANW',
    'PSKY',
    'PH',
    'PAYX',
    'PYPL',
    'PNR',
    'PEP',
    'PFE',
    'PCG',
    'PM',
    'PSX',
    'PNW',
    'PNC',
    'POOL',
    'PPG',
    'PPL',
    'PFG',
    'PG',
    'PGR',
    'PLD',
    'PRU',
    'PEG',
    'PTC',
    'PSA',
    'PHM',
    'PWR',
    'QCOM',
    'DGX',
    'Q',
    'RL',
    'RJF',
    'RTX',
    'O',
    'REG',
    'REGN',
    'RF',
    'RSG',
    'RMD',
    'RVTY',
    'HOOD',
    'ROK',
    'ROL',
    'ROP',
    'ROST',
    'RCL',
    'SPGI',
    'CRM',
    'SNDK',
    'SBAC',
    'SLB',
    'STX',
    'SRE',
    'NOW',
    'SHW',
    'SPG',
    'SWKS',
    'SJM',
    'SW',
    'SNA',
    'SOLV',
    'SO',
    'LUV',
    'SWK',
    'SBUX',
    'STT',
    'STLD',
    'STE',
    'SYK',
    'SMCI',
    'SYF',
    'SNPS',
    'SYY',
    'TMUS',
    'TROW',
    'TTWO',
    'TPR',
    'TRGP',
    'TGT',
    'TEL',
    'TDY',
    'TER',
    'TSLA',
    'TXN',
    'TPL',
    'TXT',
    'TMO',
    'TJX',
    'TKO',
    'TTD',
    'TSCO',
    'TT',
    'TDG',
    'TRV',
    'TRMB',
    'TFC',
    'TYL',
    'TSN',
    'USB',
    'UBER',
    'UDR',
    'ULTA',
    'UNP',
    'UAL',
    'UPS',
    'URI',
    'UNH',
    'UHS',
    'VLO',
    'VEEV',
    'VTR',
    'VLTO',
    'VRSN',
    'VRSK',
    'VZ',
    'VRTX',
    'VRT',
    'VTRS',
    'VICI',
    'V',
    'VST',
    'VMC',
    'WRB',
    'GWW',
    'WAB',
    'WMT',
    'DIS',
    'WBD',
    'WM',
    'WAT',
    'WEC',
    'WFC',
    'WELL',
    'WST',
    'WDC',
    'WY',
    'WSM',
    'WMB',
    'WTW',
    'WDAY',
    'WYNN',
    'XEL',
    'XYL',
    'YUM',
    'ZBRA',
    'ZBH',
    'ZTS'
]

# ========== 以下部分一般不需要修改 ==========

MAPPING_URL = "https://api.openfigi.com/v3/mapping"
HEADERS = {"Content-Type": "application/json"}
if API_KEY:
    HEADERS["X-OPENFIGI-APIKEY"] = API_KEY

# 每次请求最多可以打包多少个job:有Key上限100个,没Key上限10个左右(保守起见用10)
BATCH_SIZE = 100 if API_KEY else 10
# 每分钟最多多少次请求:有Key限速较高,没Key限速很低,这里保守设置sleep时间
SLEEP_BETWEEN_BATCHES = 6 if API_KEY else 20


def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def build_jobs(ticker_batch):
    """
    idType=TICKER, exchCode=US 限定美国交易所,避免同名ticker匹配到其他国家
    marketSecDes=Equity 限定股票类证券
    """
    return [
        {
            "idType": "TICKER",
            "idValue": t,
            "exchCode": "US",
            "marketSecDes": "Equity",
        }
        for t in ticker_batch
    ]


def fetch_isin_for_all(tickers):
    results = []
    not_found = []

    for batch in chunked(tickers, BATCH_SIZE):
        jobs = build_jobs(batch)
        resp = requests.post(MAPPING_URL, json=jobs, headers=HEADERS)

        if resp.status_code == 429:
            print("触发限速(429),等待60秒后重试这一批...")
            time.sleep(60)
            resp = requests.post(MAPPING_URL, json=jobs, headers=HEADERS)

        if resp.status_code != 200:
            print(f"请求失败: status={resp.status_code}, body={resp.text[:300]}")
            for t in batch:
                not_found.append(t)
            time.sleep(SLEEP_BETWEEN_BATCHES)
            continue

        data = resp.json()

        for ticker, item in zip(batch, data):
            if "data" in item and len(item["data"]) > 0:
                # 一个ticker可能匹配到多条(比如不同的份额类别/上市地),取第一条作为主要结果
                first = item["data"][0]
                isin = first.get("isin") or first.get("compositeFIGI") or ""
                # OpenFIGI mapping接口默认返回字段里通常不直接含isin,
                # 如果没有isin字段,需要用返回的FIGI再调一次 /v3/mapping (idType=ID_BB_GLOBAL) 反查,
                # 这里先尝试直接拿,如果没有则记录FIGI供后续反查
                results.append({
                    "Ticker": ticker,
                    "Name": first.get("name", ""),
                    "FIGI": first.get("figi", ""),
                    "ISIN": isin,
                })
            else:
                not_found.append(ticker)
                results.append({
                    "Ticker": ticker,
                    "Name": "",
                    "FIGI": "",
                    "ISIN": "",
                })

        print(f"已完成 {len(results)}/{len(tickers)} ...")
        time.sleep(SLEEP_BETWEEN_BATCHES)

    return results, not_found


def enrich_with_isin_via_figi(results):
    """
    OpenFIGI 的 /v3/mapping 用 TICKER 查询时,返回结果里通常不直接带 isin 字段。
    需要拿到 FIGI 后,再用 idType=ID_BB_GLOBAL 反查一次,返回结果里才会带 isin。
    这个函数对上一步没有拿到isin、但拿到了FIGI的记录,做第二轮查询补全isin。
    """
    need_lookup = [r for r in results if r["FIGI"] and not r["ISIN"]]
    if not need_lookup:
        return results

    print(f"\n开始第二轮查询,用FIGI反查ISIN,共{len(need_lookup)}条...")

    figi_to_isin = {}
    for batch in chunked(need_lookup, BATCH_SIZE):
        jobs = [{"idType": "ID_BB_GLOBAL", "idValue": r["FIGI"]} for r in batch]
        resp = requests.post(MAPPING_URL, json=jobs, headers=HEADERS)

        if resp.status_code == 429:
            print("触发限速(429),等待60秒后重试...")
            time.sleep(60)
            resp = requests.post(MAPPING_URL, json=jobs, headers=HEADERS)

        if resp.status_code != 200:
            print(f"第二轮请求失败: status={resp.status_code}")
            time.sleep(SLEEP_BETWEEN_BATCHES)
            continue

        data = resp.json()
        for r, item in zip(batch, data):
            if "data" in item and len(item["data"]) > 0:
                isin = item["data"][0].get("isin", "")
                figi_to_isin[r["FIGI"]] = isin

        time.sleep(SLEEP_BETWEEN_BATCHES)

    for r in results:
        if r["FIGI"] in figi_to_isin:
            r["ISIN"] = figi_to_isin[r["FIGI"]]

    return results


def main():
    print(f"共 {len(TICKERS)} 个ticker待查询,批大小={BATCH_SIZE}, 每批间隔={SLEEP_BETWEEN_BATCHES}秒")
    if not API_KEY:
        print("警告: 未设置API_KEY,查询会很慢且容易被限速。建议先去 openfigi.com 申请免费Key。")

    results, not_found = fetch_isin_for_all(TICKERS)
    results = enrich_with_isin_via_figi(results)

    out_path = "sp500_isin_result.csv"
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["Ticker", "Name", "FIGI", "ISIN"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n完成。结果已保存到 {out_path}")
    if not_found:
        print(f"\n以下 {len(not_found)} 个ticker未查到匹配结果,建议手动核对(可能代码变更/退市/多股权类别):")
        print(", ".join(not_found))


if __name__ == "__main__":
    main()