#!/usr/bin/env python3
"""
ALGO_STANDALONE.py — Complete stock recommendation engine in a single file.
Fully self-contained: config embedded, no external files needed.

Usage (terminal):
    python ALGO_STANDALONE.py --debug          # quick test (2 tickers)
    python ALGO_STANDALONE.py                  # full 50-ticker universe
    python ALGO_STANDALONE.py --held AAPL NVDA # with sell alerts

Usage (Google Colab):
    1. Paste this entire file into a cell
    2. Run it — pip installs happen automatically
    3. Call: run(debug=True)
"""
from __future__ import annotations

# =====================================================================
# AUTO-INSTALL DEPENDENCIES (safe for Colab + terminal)
# =====================================================================
import subprocess, sys

def _ensure_packages():
    required = ["yfinance", "vaderSentiment", "pyyaml", "pandas", "numpy"]
    for pkg in required:
        try:
            __import__(pkg.replace("-", "_").split("==")[0])
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

_ensure_packages()

# =====================================================================
# IMPORTS
# =====================================================================
import argparse
import copy
import logging
import random
import time
import io
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Parallelism setting — adjust based on yfinance rate limits
_MAX_WORKERS = 12

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =====================================================================
# EMBEDDED CONFIG (no external YAML file needed)
# =====================================================================

_EMBEDDED_CONFIG_YAML = r"""
# -----------------------------------------------------------------------
# Universe — 1000+ stocks: S&P 500, mid-caps, growth, ADRs
# -----------------------------------------------------------------------
universe:
  # ── Technology — Mega/Large Cap ────────────────────────────────────
  - "AAPL"
  - "MSFT"
  - "NVDA"
  - "AVGO"
  - "AMD"
  - "ORCL"
  - "CRM"
  - "ADBE"
  - "INTC"
  - "QCOM"
  - "TXN"
  - "MU"
  - "NOW"
  - "INTU"
  - "SNPS"
  - "CDNS"
  - "KLAC"
  - "LRCX"
  - "AMAT"
  - "MRVL"
  - "ADI"
  - "ON"
  - "FTNT"
  - "PANW"
  - "CRWD"
  - "ZS"
  - "DDOG"
  - "SNOW"
  - "PLTR"
  - "ANET"
  - "NET"
  - "TEAM"
  - "HUBS"
  - "DOCU"
  - "MDB"
  - "TTD"
  - "SMCI"
  - "ARM"
  - "DELL"
  - "HPE"
  # ── Technology — Mid/Small Cap ─────────────────────────────────────
  - "NXPI"
  - "MPWR"
  - "SWKS"
  - "MCHP"
  - "GEN"
  - "AKAM"
  - "EPAM"
  - "CGNX"
  - "SAPMY"
  - "PCTY"
  - "WK"
  - "TENB"
  - "QLYS"
  - "RPD"
  - "VRNS"
  - "DOMO"
  - "OKTA"
  - "KRNT"
  - "ESTC"
  - "GTLB"
  - "PATH"
  - "APPF"
  - "MANH"
  - "PLTK"
  - "BSY"
  - "NICE"
  - "WEX"
  - "GWRE"
  - "TOST"
  - "FOUR"
  - "PEGA"
  - "DT"
  - "CPAY"
  - "CIEN"
  - "LITE"
  - "COHR"
  - "MKSI"
  - "ENTG"
  - "AMKR"
  - "RMBS"
  - "CRUS"
  - "SYNA"
  - "AMBA"
  - "LSCC"
  - "WOLF"
  - "ACLS"
  - "FORM"
  - "POWI"
  - "DIOD"
  # ── Communication Services ─────────────────────────────────────────
  - "GOOGL"
  - "META"
  - "NFLX"
  - "DIS"
  - "CMCSA"
  - "T"
  - "VZ"
  - "TMUS"
  - "SPOT"
  - "RBLX"
  - "PINS"
  - "SNAP"
  - "EA"
  - "TTWO"
  - "ROKU"
  - "WBD"
  - "SLAB"
  - "LBRT"
  - "FOXA"
  - "LYV"
  - "MTCH"
  - "IMAX"
  - "ZM"
  - "LUMN"
  - "NWSA"
  - "NYT"
  - "OMC"
  - "OPCH"
  # ── Consumer Discretionary — Large Cap ─────────────────────────────
  - "AMZN"
  - "TSLA"
  - "HD"
  - "MCD"
  - "NKE"
  - "COST"
  - "LOW"
  - "TJX"
  - "SBUX"
  - "BKNG"
  - "ORLY"
  - "AZO"
  - "ROST"
  - "YUM"
  - "CMG"
  - "DHI"
  - "LEN"
  - "PHM"
  - "GM"
  - "F"
  - "ABNB"
  - "DASH"
  - "UBER"
  - "LYFT"
  - "ETSY"
  - "W"
  - "RCL"
  - "WYNN"
  - "MGM"
  # ── Consumer Discretionary — Mid/Small Cap ─────────────────────────
  - "POOL"
  - "DPZ"
  - "DKNG"
  - "EXPE"
  - "MAR"
  - "HLT"
  - "LVS"
  - "CZR"
  - "NCLH"
  - "CCL"
  - "DRI"
  - "TXRH"
  - "WING"
  - "SHAK"
  - "EAT"
  - "CAKE"
  - "PLAY"
  - "DIN"
  - "BBY"
  - "FIVE"
  - "ULTA"
  - "LULU"
  - "DECK"
  - "CROX"
  - "ONON"
  - "TPR"
  - "RL"
  - "CPRI"
  - "ENOV"
  - "PVH"
  - "CFR"
  - "ANF"
  - "AEO"
  - "GAP"
  - "KMX"
  - "AN"
  - "LAD"
  - "GPC"
  - "AAP"
  - "TOL"
  - "MTH"
  - "KBH"
  - "GRMN"
  - "FOXF"
  - "BC"
  - "LCII"
  - "CVNA"
  - "CART"
  - "CHWY"
  - "BROS"
  - "CAVA"
  # ── Consumer Staples ───────────────────────────────────────────────
  - "PG"
  - "KO"
  - "PEP"
  - "PM"
  - "MO"
  - "CL"
  - "MDLZ"
  - "GIS"
  - "SJM"
  - "KR"
  - "WMT"
  - "TGT"
  - "STZ"
  - "EL"
  - "KHC"
  - "HSY"
  - "MKC"
  - "CPB"
  - "HRL"
  - "TSN"
  - "CAG"
  - "MNST"
  - "KDP"
  - "SBRA"
  - "TAP"
  - "SAM"
  - "CELH"
  - "ADM"
  - "BG"
  - "INGR"
  - "CHD"
  - "CLX"
  - "KVUE"
  - "SPB"
  - "FLS"
  - "DG"
  - "DLTR"
  - "CASY"
  - "USFD"
  - "SYY"
  - "PFGC"
  # ── Healthcare — Large Cap ─────────────────────────────────────────
  - "LLY"
  - "UNH"
  - "JNJ"
  - "ABBV"
  - "MRK"
  - "PFE"
  - "TMO"
  - "ABT"
  - "AMGN"
  - "GILD"
  - "VRTX"
  - "REGN"
  - "ISRG"
  - "DXCM"
  - "BSX"
  - "SYK"
  - "MDT"
  - "ZTS"
  - "HCA"
  - "CI"
  - "ELV"
  - "MRNA"
  - "BIIB"
  - "ILMN"
  - "ALGN"
  # ── Healthcare — Mid/Small Cap ─────────────────────────────────────
  - "IQV"
  - "CRL"
  - "MEDP"
  - "VEEV"
  - "HOLX"
  - "IDXX"
  - "PODD"
  - "TFX"
  - "INSP"
  - "NVST"
  - "RVTY"
  - "BIO"
  - "A"
  - "MTD"
  - "WAT"
  - "MATV"
  - "EW"
  - "RMD"
  - "BAX"
  - "BDX"
  - "COO"
  - "GEHC"
  - "HSIC"
  - "XRAY"
  - "CNC"
  - "MOH"
  - "DVA"
  - "THC"
  - "ENSG"
  - "PAYX"
  - "HIMS"
  - "DOCS"
  - "GDRX"
  - "SDGR"
  - "RXRX"
  - "DNA"
  - "CRSP"
  - "NTLA"
  - "BEAM"
  - "EDIT"
  - "EXAS"
  - "GH"
  - "NTRA"
  - "RARE"
  - "PCVX"
  - "SMMT"
  - "SRRK"
  - "IONS"
  - "ALNY"
  - "BMRN"
  - "INCY"
  - "NBIX"
  - "PTCT"
  - "TDS"
  - "UTHR"
  - "EXEL"
  - "SRPT"
  - "JAZZ"
  - "ARVN"
  - "IOVA"
  # ── Financials — Large Cap ─────────────────────────────────────────
  - "JPM"
  - "V"
  - "MA"
  - "GS"
  - "BLK"
  - "BAC"
  - "WFC"
  - "C"
  - "MS"
  - "SCHW"
  - "AXP"
  - "SPGI"
  - "MCO"
  - "ICE"
  - "CME"
  - "CB"
  - "AON"
  - "BRO"
  - "PGR"
  - "TRV"
  - "MET"
  - "AIG"
  - "COIN"
  - "HOOD"
  - "SOFI"
  # ── Financials — Mid/Small Cap ─────────────────────────────────────
  - "USB"
  - "PNC"
  - "TFC"
  - "FITB"
  - "MTB"
  - "HBAN"
  - "CFG"
  - "RF"
  - "KEY"
  - "ZION"
  - "COTY"
  - "ALLY"
  - "BANF"
  - "SYF"
  - "NDAQ"
  - "CBOE"
  - "MSCI"
  - "FDS"
  - "MKTX"
  - "VIRT"
  - "LPLA"
  - "RJF"
  - "SF"
  - "EVR"
  - "HLI"
  - "IBKR"
  - "MARA"
  - "RIOT"
  - "GIII"
  - "GPN"
  - "FIS"
  - "FISV"
  - "WU"
  - "PYPL"
  - "AFRM"
  - "UPST"
  - "LC"
  - "TREE"
  - "GL"
  - "AFL"
  - "PRU"
  - "LNC"
  - "ALL"
  - "CINF"
  - "HIG"
  - "WRB"
  - "RNR"
  - "ERIE"
  - "KNSL"
  - "RYAN"
  - "AJG"
  - "TROW"
  - "IVZ"
  - "BEN"
  - "AMG"
  - "NTRS"
  - "STT"
  - "BK"
  # ── Industrials — Large Cap ────────────────────────────────────────
  - "CAT"
  - "DE"
  - "HON"
  - "RTX"
  - "GE"
  - "UNP"
  - "UPS"
  - "BA"
  - "LMT"
  - "NOC"
  - "GD"
  - "MMM"
  - "EMR"
  - "ETN"
  - "ITW"
  - "WM"
  - "RSG"
  - "FAST"
  - "CTAS"
  - "URI"
  - "PWR"
  - "AXON"
  # ── Industrials — Mid/Small Cap ────────────────────────────────────
  - "CARR"
  - "OTIS"
  - "AME"
  - "ROK"
  - "DOV"
  - "NDSN"
  - "RRX"
  - "PH"
  - "SWK"
  - "TT"
  - "IR"
  - "XYL"
  - "AOS"
  - "MAS"
  - "GNRC"
  - "FTV"
  - "HUBB"
  - "LECO"
  - "WCC"
  - "CSL"
  - "FLR"
  - "J"
  - "KBR"
  - "ACM"
  - "PRIM"
  - "TTEK"
  - "MTZ"
  - "FIX"
  - "STRL"
  - "GWW"
  - "VSH"
  - "CNH"
  - "AGCO"
  - "TTC"
  - "SNA"
  - "RBC"
  - "GGG"
  - "IEX"
  - "MIDD"
  - "B"
  - "CW"
  - "HEI"
  - "TDG"
  - "HWM"
  - "CHDN"
  - "TXT"
  - "HII"
  - "LHX"
  - "BWXT"
  - "KTOS"
  - "RKLB"
  - "IRDM"
  - "CSX"
  - "NSC"
  - "CP"
  - "CNI"
  - "JBHT"
  - "ODFL"
  - "XPO"
  - "SAIA"
  - "CHRW"
  - "EXPD"
  - "KEX"
  - "MATX"
  - "DAL"
  - "UAL"
  - "AAL"
  - "LUV"
  - "ALK"
  - "SMWB"
  - "FDX"
  - "R"
  - "PCAR"
  - "CMI"
  - "CARG"
  - "WAB"
  - "ALLE"
  - "AIT"
  - "MSM"
  - "GATX"
  - "RHI"
  - "MAN"
  - "SIEGY"
  - "CPRT"
  - "SPXC"
  - "WSO"
  # ── Energy ─────────────────────────────────────────────────────────
  - "XOM"
  - "CVX"
  - "COP"
  - "EOG"
  - "SLB"
  - "MPC"
  - "VLO"
  - "PSX"
  - "OXY"
  - "TRGP"
  - "DVN"
  - "HAL"
  - "FANG"
  - "WMB"
  - "KMI"
  - "OKE"
  - "ET"
  - "EPD"
  - "MPLX"
  - "LNG"
  - "AR"
  - "EQT"
  - "RRC"
  - "AEIS"
  - "CTRA"
  - "MTDR"
  - "CHRD"
  - "PR"
  - "VNOM"
  - "BKR"
  - "FTI"
  - "NOV"
  - "ESAB"
  - "PTEN"
  - "HP"
  - "RIG"
  - "VAL"
  - "VRNT"
  - "APA"
  - "DINO"
  - "PBF"
  - "KMPR"
  - "CLNE"
  - "PLUG"
  - "FSLR"
  - "COKE"
  # ── Utilities ──────────────────────────────────────────────────────
  - "NEE"
  - "SO"
  - "DUK"
  - "D"
  - "AEP"
  - "SRE"
  - "XEL"
  - "EXC"
  - "WEC"
  - "ED"
  - "VST"
  - "PCG"
  - "EIX"
  - "ES"
  - "FE"
  - "DTE"
  - "CMS"
  - "AES"
  - "ATO"
  - "NI"
  - "PNW"
  - "OGE"
  - "NRG"
  - "CEG"
  - "PPL"
  - "ETR"
  - "CNP"
  - "EVRG"
  - "AWK"
  - "WTRG"
  # ── Real Estate ────────────────────────────────────────────────────
  - "PLD"
  - "AMT"
  - "CCI"
  - "EQIX"
  - "SPG"
  - "O"
  - "DLR"
  - "PSA"
  - "WELL"
  - "VICI"
  - "AVB"
  - "EQR"
  - "MAA"
  - "UDR"
  - "ESS"
  - "CPT"
  - "INVH"
  - "SUI"
  - "ELS"
  - "ARE"
  - "BXP"
  - "VNO"
  - "SLG"
  - "KIM"
  - "REG"
  - "FRT"
  - "NNN"
  - "EPRT"
  - "ADC"
  - "DORM"
  - "WPC"
  - "IRM"
  - "SBAC"
  - "CBRE"
  - "JLL"
  - "CWK"
  - "RKT"
  # ── Materials ──────────────────────────────────────────────────────
  - "LIN"
  - "SHW"
  - "APD"
  - "ECL"
  - "NEM"
  - "FCX"
  - "NUE"
  - "STLD"
  - "DOW"
  - "DD"
  - "PPG"
  - "VMC"
  - "MLM"
  - "EMN"
  - "CE"
  - "ALB"
  - "FMC"
  - "MOS"
  - "CF"
  - "IFF"
  - "AVNT"
  - "RPM"
  - "AXTA"
  - "CBT"
  - "TROX"
  - "CC"
  - "HUN"
  - "OLN"
  - "WLK"
  - "RS"
  - "CMC"
  - "ATI"
  - "CRS"
  - "CLF"
  - "ONTO"
  - "AA"
  - "CENX"
  - "MP"
  - "LAC"
  - "GOLD"
  - "AEM"
  - "KGC"
  - "FNV"
  - "WPM"
  - "RGLD"
  - "AG"
  - "PAAS"
  - "HL"
  - "CDE"
  - "SSRM"
  - "BTG"
  - "IAG"
  # ── International ADRs ─────────────────────────────────────────────
  - "TSM"
  - "BABA"
  - "NVO"
  - "ASML"
  - "SAP"
  - "TM"
  - "SONY"
  - "LI"
  - "NIO"
  - "XPEV"
  - "BIDU"
  - "JD"
  - "PDD"
  - "NTES"
  - "TCEHY"
  - "MUFG"
  - "SMFG"
  - "MFG"
  - "KB"
  - "SHG"
  - "WIT"
  - "INFY"
  - "HDB"
  - "IBN"
  - "UL"
  - "DEO"
  - "BTI"
  - "GSK"
  - "AZN"
  - "NVS"
  - "SNY"
  - "RDY"
  - "TAK"
  - "VALE"
  - "SCCO"
  - "SQM"
  - "GLOB"
  - "STNE"
  - "PAGS"
  - "NU"
  - "GRAB"
  - "CPNG"
  - "YUMC"
  - "QSR"
  - "COF"
  - "BFAM"
  - "RELX"
  - "WCN"
  - "BAM"
  - "BN"
  # ── High-Growth / Emerging ─────────────────────────────────────────
  - "MELI"
  - "SE"
  - "ENPH"
  - "SEDG"
  - "RIVN"
  - "LCID"
  - "DUOL"
  - "BIRK"
  - "APP"
  - "IOT"
  - "BILL"
  - "CFLT"
  - "S"
  - "MNDY"
  - "GLBE"
  - "SHOP"
  - "XYZ"
  - "TWLO"
  - "PAYO"
  - "FSLY"
  - "CLSK"
  - "VERX"
  - "SOUN"
  - "ASAN"
  - "DOCN"
  - "BRZE"
  - "CWAN"
  - "AUR"
  - "IONQ"
  - "RGTI"
  - "QUBT"
  - "LUNR"
  - "MSTR"
  - "JOBY"
  - "ACHR"
  - "PI"
  - "VRT"
  - "FICO"
  - "POWL"
  - "TW"
  - "RELY"
  - "TMDX"
  - "NUVB"
  - "OWL"
  - "ARES"
  - "APO"
  - "KKR"
  - "CG"
  - "BX"
  - "TPG"
  # ── Additional S&P 500 / Large Cap ─────────────────────────────────
  - "NXST"
  - "WTW"
  - "BR"
  - "LDOS"
  - "SAIC"
  - "BAH"
  - "IT"
  - "CTSH"
  - "ACN"
  - "IBM"
  - "HPQ"
  - "OMCL"
  - "FFIV"
  - "CDW"
  - "NTAP"
  - "STX"
  - "WDC"
  - "ZBRA"
  - "TRMB"
  - "TYL"
  - "PTC"
  - "VTR"
  - "DSGX"
  - "BNTX"
  - "VRSK"
  - "MCK"
  - "CAH"
  - "COR"
  - "LFST"
  - "WST"
  - "STE"
  - "TER"
  - "KEYS"
  - "LFUS"
  - "TEL"
  - "APH"
  - "GLW"
  - "JCI"
  - "LII"
  - "WSC"
  - "ROP"
  - "L"
  - "AIZ"
  - "INMD"
  - "ACGL"
  - "EG"
  - "FNF"
  - "FAF"
  - "MGNI"
  # ── Expanded Tech / Software ───────────────────────────────────────
  - "WDAY"
  - "BURL"
  - "CRDO"
  - "SMTC"
  - "CALX"
  - "VIAV"
  - "SANM"
  - "FLEX"
  - "CLB"
  - "PLXS"
  - "TTMI"
  - "CXT"
  - "ASGN"
  - "EXLS"
  - "TASK"
  - "KD"
  - "MTRN"
  - "PD"
  - "BL"
  - "MSGE"
  - "CNXC"
  - "G"
  - "CEVA"
  - "NTNX"
  - "APPN"
  - "FRSH"
  - "CNXN"
  - "ALRM"
  - "CERT"
  - "INTA"
  - "OSIS"
  - "PCOR"
  - "QTWO"
  - "PRGS"
  - "SPSC"
  - "NCNO"
  - "ALKT"
  - "YOU"
  - "AI"
  - "BBAI"
  - "RGEN"
  - "VTEX"
  - "LOGI"
  - "STM"
  - "QRVO"
  - "UMC"
  - "ASX"
  # ── Expanded Healthcare / Biotech ──────────────────────────────────
  - "DRVN"
  - "TNDM"
  - "LIVN"
  - "ATEC"
  - "HQY"
  - "AVAV"
  - "IRTC"
  - "OFIX"
  - "GMED"
  - "LNTH"
  - "MASI"
  - "NVCR"
  - "PRCT"
  - "RVMD"
  - "RYTM"
  - "AAOI"
  - "TVTX"
  - "XENE"
  - "ZLAB"
  - "APLS"
  - "ACLX"
  - "IMVT"
  - "MNKD"
  - "TRUP"
  - "CHGG"
  - "COUR"
  - "UDMY"
  # ── Expanded Consumer / Leisure ────────────────────────────────────
  - "PENN"
  - "BYD"
  - "SITM"
  - "SEM"
  - "FUN"
  - "SWBI"
  - "GOLF"
  - "ACVA"
  - "PTGX"
  - "BBWI"
  - "WRBY"
  - "FIGS"
  - "OLLI"
  - "BOOT"
  - "RH"
  - "HELE"
  - "LEG"
  - "ARHS"
  - "WSM"
  - "WGO"
  - "THO"
  - "CWH"
  - "MTN"
  - "PLNT"
  - "XPOF"
  - "PTON"
  - "LW"
  # ── Expanded Industrials / Construction ────────────────────────────
  - "BLBD"
  - "SITE"
  - "TREX"
  - "RUN"
  - "AAON"
  - "WFRD"
  - "CLH"
  - "IPAR"
  - "TNET"
  - "TRN"
  - "ASTS"
  - "VSAT"
  - "GSAT"
  - "BLDR"
  - "IBP"
  - "TILE"
  - "ATKR"
  - "AWI"
  - "HXL"
  - "WTS"
  - "EXP"
  # ── Expanded Financials / Banks ────────────────────────────────────
  - "WAL"
  - "FHN"
  - "WTFC"
  - "PNFP"
  - "OZK"
  - "COLB"
  - "FFIN"
  - "SBCF"
  - "TBBK"
  - "VERA"
  - "HWC"
  - "EWBC"
  - "BANR"
  - "GBCI"
  - "ABCB"
  - "FCNCA"
  - "BOKF"
  - "ROG"
  - "SSB"
  - "UMBF"
  - "ONB"
  - "FBP"
  # ── Expanded Energy / Coal / Gas ───────────────────────────────────
  - "KLIC"
  - "SM"
  - "NOG"
  - "SKYW"
  - "TALO"
  - "ARIS"
  - "CNX"
  - "AM"
  - "ASPN"
  - "BTU"
  - "SLDP"
  - "HCC"
  - "MGY"
  - "NFE"
  # ── Expanded International ADRs ────────────────────────────────────
  - "GFL"
  - "LSPD"
  - "OTLY"
  - "VNET"
  - "ZTO"
  - "BEKE"
  - "MNSO"
  - "FUTU"
  - "TIGR"
  # ── Expanded REITs ─────────────────────────────────────────────────
  - "COLD"
  - "REXR"
  - "STAG"
  - "TRNO"
  - "LTC"
  - "OHI"
  - "FIZZ"
  - "IIPR"
  - "APLE"
  - "HST"
  - "RLJ"
  - "PK"
  - "XHR"
  - "LAMR"
  - "OUT"
  - "CCO"
  - "LBRDA"
  # ── Russell 2000 / Mid-Small Cap Expansion ─────────────────────────
  - "WIX"
  - "UI"
  - "NTGR"
  - "NTCT"
  - "MITK"
  - "DGII"
  - "BAND"
  - "RAMP"
  - "FIVN"
  - "PLUS"
  - "ONTF"
  - "BLZE"
  - "GDS"
  - "KC"
  - "MOMO"
  - "WB"
  - "TUYA"
  - "QFIN"
  - "JFIN"
  - "HUYA"
  - "BZ"
  - "DAO"
  - "GOTU"
  - "TAL"
  - "EDU"
  - "NOAH"
  - "NIU"
  - "BANC"
  - "PEBO"
  - "TCBI"
  - "WSFS"
  - "CBSH"
  - "FULT"
  - "WBS"
  - "INDB"
  - "OFG"
  - "FFBC"
  - "NWBI"
  - "ASB"
  - "BPOP"
  - "FCF"
  - "OCFC"
  - "RNST"
  - "FIBK"
  - "HOMB"
  - "STBA"
  - "CBU"
  - "BUSE"
  - "UMH"
  - "PFG"
  - "CNO"
  - "RGA"
  - "UNM"
  - "GNW"
  - "MBI"
  - "AGIO"
  - "ALKS"
  - "RLAY"
  - "KYMR"
  - "EOLS"
  - "INSM"
  - "CLDX"
  - "NVAX"
  - "OPK"
  - "FATE"
  - "ARCT"
  - "RCUS"
  - "RIGL"
  - "VKTX"
  - "STOK"
  - "GERN"
  - "NRIX"
  - "SPRY"
  - "ZNTL"
  - "ALXO"
  - "CRNX"
  - "ANAB"
  - "ASND"
  - "MLAB"
  - "SINT"
  - "NUVL"
  - "FROG"
  - "PRDO"
  - "STRA"
  - "LOPE"
  - "LRN"
  - "VVV"
  - "BJ"
  - "WW"
  - "MED"
  - "NOMD"
  - "ENR"
  - "FCAP"
  - "POST"
  - "UTZ"
  - "SMPL"
  - "LANC"
  - "JJSF"
  - "FLO"
  - "KOF"
  - "CCEP"
  - "MLI"
  - "ROCK"
  - "KMT"
  - "ATR"
  - "EME"
  - "KAI"
  - "VMI"
  - "MTW"
  - "TKR"
  - "HRI"
  - "KNF"
  - "GRBK"
  - "GVA"
  - "MMS"
  - "FFAI"
  - "BLNK"
  - "CHPT"
  - "EVGO"
  - "WBX"
  - "ARLP"
  - "REI"
  - "CRC"
  - "TPL"
  - "PARR"
  - "VTNR"
  - "GPRE"
  - "RNW"
  - "OPAL"
  - "NESR"
  - "WTTR"
  - "KOS"
  - "BORR"
  - "RES"
  - "HMY"
  - "NGD"
  - "EQX"
  - "OR"
  - "FSM"
  - "SVM"
  - "EXK"
  - "BRX"
  - "SHO"
  - "DRH"
  - "INN"
  - "PEB"
  - "MCY"
  - "EPR"
  - "FR"
  - "CUZ"
  - "AKR"
  - "FCPT"
  - "NSA"
  - "CUBE"
  - "RHP"
  - "HASI"
  - "SAFE"
  - "RYAAY"
  - "BRKR"
  - "ITUB"
  - "BBD"
  - "BSAC"
  - "ICL"
  - "TEVA"
  - "CHKP"
  - "GRRR"
  - "TEM"
  - "PSNL"
  - "NN"
  - "ZURA"
  - "KLAR"
  - "PRSU"
  - "RBRK"
  - "KVYO"
  - "NMR"
  - "ASCB"
  - "WALD"
  - "DJT"
  - "SCNI"
  - "BUD"
  - "SPCE"
  - "MRCY"
  - "WWD"
  - "ESLT"
  - "CACI"

start_date: "2018-01-01"
top_n: 15
rebalance_days: 63
initial_capital: 100000
transaction_cost: 0.0005
slippage: 0.0010
max_position_pct: 0.20
max_adv_participation: 0.05
target_portfolio_vol: 0.12
risk_budget_per_asset: 0.20
stop_loss: 0.12
time_stop_days: 90
impact_k: 0.02
impact_alpha: 0.8
turnover_penalty: 0.05
random_seed: 42
min_confidence_score: 0.80

weights:
  technical: 0.30        # was 0.38 — value plays prioritize fundamentals
  fundamental: 0.70      # was 0.62 — heavier fundamentals tilt

technical_weights:
  momentum: 0.04         # was 0.10 — REDUCED, don't chase rallies
  trend: 0.05            # was 0.06
  rsi: 0.06              # was 0.02 — INCREASED & inverted (oversold reward)
  macd: 0.02
  bb_pct: 0.05           # was 0.01 — INCREASED & inverted
  volume: 0.01
  relative_strength: 0.03  # was 0.08 — REDUCED (RS leaders often at highs)
  price_structure: 0.03    # was 0.05
  volume_profile: 0.03
  discount: 0.15         # NEW — major reward for being below 52wk high

fundamental_weights:
  value: 0.25            # was 0.14 — DOUBLED (P/E + PEG + P/B + P/S)
  quality: 0.18
  growth: 0.08           # was 0.15 — REDUCED (value > growth)
  fcf_yield: 0.18        # was 0.12 — heavily boosted
  leverage_penalty: -0.08

rsi_period: 14
macd_fast: 12
macd_slow: 26
macd_signal_period: 9
bb_period: 20
bb_std: 2.0
volume_avg_period: 20
golden_cross_lookback: 5

explanation_thresholds:
  momentum_strong: 0.12
  rsi_healthy_low: 35
  rsi_healthy_high: 60
  rsi_overbought: 72
  rsi_oversold: 28
  volume_trend_strong: 1.25
  bb_pct_oversold: 0.30
  bb_pct_extended: 0.88
  roe_strong: 0.15
  peg_cheap: 1.5
  peg_expensive: 2.5
  pe_high: 35
  debt_equity_high: 150
  revenue_growth_strong: 0.08
  fcf_yield_positive: 0.025
  fcf_yield_negative: -0.02
  beta_high: 1.6
  profit_margin_strong: 0.12

news:
  max_age_hours: 120
  recency_decay: 0.10
  sell_alert_score: -0.30
  caution_score: -0.15
  bullish_boost_score: 0.25

regime:
  lookback_days: 252
  vol_window: 20
  sma_window: 200
  vol_bull_quiet: 0.15
  vol_bull_volatile_max: 0.30
  vol_bear_quiet_max: 0.20
  vol_crisis: 0.35

macro:
  vix_calm: 18.0
  vix_elevated: 25.0
  vix_fear: 35.0
  vix_crisis: 45.0
  yield_curve_inversion_threshold: -0.25
  rate_rising_threshold: 0.50
  dollar_strong_threshold: 0.04
  commodity_rising_threshold: 0.06
  commodity_falling_threshold: -0.06

sell_signals:
  news_score_threshold: -0.30
  regime_sell_on:
    - "CRISIS"
    - "BEAR_VOLATILE"
  technical_deterioration:
    rsi_overbought_sell: 78
    macd_cross_negative: true
    death_cross: true

relative_strength:
  rs_windows: [21, 63, 126]
  sector_etf_map:
    Technology: "XLK"
    Healthcare: "XLV"
    "Financial Services": "XLF"
    "Consumer Cyclical": "XLY"
    "Consumer Defensive": "XLP"
    Industrials: "XLI"
    Energy: "XLE"
    Utilities: "XLU"
    "Real Estate": "XLRE"
    "Basic Materials": "XLB"
    "Communication Services": "XLC"

price_structure:
  atr_period: 14
  trend_slope_window: 63
  pullback_window: 20
  efficiency_window: 20
  hh_hl_window: 20
  ma_dist_windows: [50, 100, 200]
  base_max_range_pct: 0.15
  base_min_duration: 15
  volatility_contraction_threshold: 0.80
  breakout_lookback: 20
  breakout_strength_min: 1.5
  breakout_volume_min: 1.5

volume_profile:
  cmf_window: 20
  ud_volume_window: 20
  obv_window: 20
  spike_threshold: 2.0
  float_turnover_window: 20
  volume_slope_window: 20

trade_plan:
  atr_stop_multiplier: 2.0
  target_1_r: 1.5
  target_2_r: 2.5
  time_stop_days: 30
  scaling_rs_threshold: 0.05

company_analysis:
  rd_high_tech: 0.10
  rd_high_other: 0.05
  rd_moderate: 0.03
  capex_expanding: 1.5
  capex_maintaining: 1.0
  buyback_min_reduction: 0.02
  dilution_min_increase: 0.03
  moat_premium_threshold: 0.10
  block_low_growth_industry: true
  block_disadvantaged_moat: true

debug_universe:
  - "AAPL"
  - "MSFT"
debug_start_date: "2023-01-01"
"""


# =====================================================================
# SHARED HELPERS
# =====================================================================


def _safe(val: Any, default: float = np.nan) -> float:
    if val is None:
        return default
    try:
        f = float(val)
        return default if (np.isnan(f) or np.isinf(f)) else f
    except (TypeError, ValueError):
        return default


def _safe_float(val: Any, default: float = np.nan) -> float:
    return _safe(val, default)


# =====================================================================
# CONFIG
# =====================================================================


def _dict_to_namespace(data: Any) -> Any:
    if isinstance(data, dict):
        return SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in data.items()})
    return data


def load_config(path: Path | None = None) -> SimpleNamespace:
    """Load config from YAML file or embedded default."""
    if path is not None and path.exists():
        with open(path) as f:
            data = yaml.safe_load(f)
    else:
        data = yaml.safe_load(_EMBEDDED_CONFIG_YAML)
    return _dict_to_namespace(data)


def config_to_weights_dict(cfg: SimpleNamespace) -> dict:
    return {
        "weights": {
            "technical": cfg.weights.technical,
            "fundamental": cfg.weights.fundamental,
        },
        "technical_weights": {
            "momentum": cfg.technical_weights.momentum,
            "trend": cfg.technical_weights.trend,
            "rsi": cfg.technical_weights.rsi,
            "macd": cfg.technical_weights.macd,
            "bb_pct": cfg.technical_weights.bb_pct,
            "volume": cfg.technical_weights.volume,
        },
        "fundamental_weights": {
            "value": cfg.fundamental_weights.value,
            "quality": cfg.fundamental_weights.quality,
            "growth": cfg.fundamental_weights.growth,
            "fcf_yield": cfg.fundamental_weights.fcf_yield,
            "leverage_penalty": cfg.fundamental_weights.leverage_penalty,
        },
    }


# =====================================================================
# EVALUATE — performance metrics
# =====================================================================


def bootstrap_sharpe(returns: np.ndarray, n_samples: int = 1000) -> float:
    if len(returns) == 0:
        return np.nan
    sharpe_vals: list[float] = []
    rng = np.random.default_rng()
    for _ in range(n_samples):
        sample = rng.choice(returns, size=len(returns), replace=True)
        std = sample.std()
        sharpe_vals.append((sample.mean() / (std + 1e-9)) * np.sqrt(252))
    return float(np.mean(sharpe_vals))


def max_drawdown(equity_curve: pd.Series) -> float:
    roll_max = equity_curve.cummax()
    drawdown = (equity_curve - roll_max) / roll_max
    return float(drawdown.min())


def annualised_return(returns: pd.Series) -> float:
    if len(returns) == 0:
        return np.nan
    total = (1 + returns).prod()
    n_years = len(returns) / 252
    return float(total ** (1 / n_years) - 1)


def compute_metrics(portfolio_values: list[float]) -> dict[str, float]:
    equity = pd.Series(portfolio_values, dtype=float)
    returns = equity.pct_change().dropna()
    if returns.empty:
        return {
            "total_return": np.nan, "cagr": np.nan, "sharpe": np.nan,
            "bootstrap_sharpe": np.nan, "max_drawdown": np.nan,
        }
    sharpe = float(returns.mean() / (returns.std() + 1e-9) * np.sqrt(252))
    total_ret = float((equity.iloc[-1] / equity.iloc[0]) - 1)
    metrics = {
        "total_return": total_ret,
        "cagr": annualised_return(returns),
        "sharpe": sharpe,
        "bootstrap_sharpe": bootstrap_sharpe(returns.values),
        "max_drawdown": max_drawdown(equity),
    }
    logger.info(
        "Backtest | Return: %.1f%% | CAGR: %.1f%% | Sharpe: %.2f | Max DD: %.1f%%",
        metrics["total_return"] * 100, metrics["cagr"] * 100,
        metrics["sharpe"], metrics["max_drawdown"] * 100,
    )
    return metrics


# =====================================================================
# DOWNLOAD — price data + fundamentals from yfinance
# =====================================================================

_FUNDAMENTAL_FIELDS = [
    "trailingPE", "forwardPE", "pegRatio", "priceToBook",
    "priceToSalesTrailing12Months", "profitMargins", "grossMargins",
    "operatingMargins", "returnOnEquity", "returnOnAssets",
    "revenueGrowth", "earningsGrowth", "debtToEquity", "currentRatio",
    "freeCashFlow", "totalRevenue", "beta", "marketCap", "floatShares",
    "sector", "industry", "averageVolume",
]
_REFERENCE_TICKERS = [
    "SPY", "XLK", "XLV", "XLF", "XLY", "XLP",
    "XLI", "XLE", "XLU", "XLRE", "XLB", "XLC",
]
_FINANCIAL_SECTORS = {"Financial Services", "Financial", "Banks", "Insurance"}


def download_prices(tickers: list[str], start_date: str) -> dict[str, pd.DataFrame]:
    logger.info("Downloading price data for %d tickers from %s", len(tickers), start_date)
    raw = yf.download(
        tickers, start=start_date, auto_adjust=True,
        group_by="ticker", progress=False,
    )
    result: dict[str, pd.DataFrame] = {}
    for tkr in tickers:
        try:
            df = raw[tkr].dropna(how="all")
            if df.empty:
                continue
            result[tkr] = df
        except KeyError:
            logger.warning("Ticker %s not found in downloaded data", tkr)
    logger.info("Loaded data for %d / %d tickers", len(result), len(tickers))
    return result


def get_adv(df: pd.DataFrame, window: int = 63) -> float:
    if len(df) < window:
        return np.nan
    return float(df["Volume"].rolling(window).mean().iloc[-1])


def fetch_fundamentals(ticker_obj: yf.Ticker) -> dict[str, Any]:
    info: dict[str, Any] = {}
    for attempt in range(2):
        try:
            info = ticker_obj.info or {}
            break
        except Exception:
            if attempt == 0:
                time.sleep(1)
    fund: dict[str, Any] = {field: info.get(field) for field in _FUNDAMENTAL_FIELDS}
    if isinstance(fund.get("trailingPE"), (int, float)) and fund["trailingPE"] < 0:
        fund["trailingPE"] = None
    eg = fund.get("earningsGrowth")
    if isinstance(eg, (int, float)) and eg <= 0:
        fund["pegRatio"] = None
    sector = fund.get("sector", "")
    if sector in _FINANCIAL_SECTORS:
        fund["debtToEquity"] = None
        fund["currentRatio"] = None
    return fund


def compute_adv_map(price_data: dict[str, pd.DataFrame], window: int = 63) -> dict[str, float]:
    return {tkr: get_adv(df, window) for tkr, df in price_data.items()}


def fetch_all_fundamentals(tickers: list[str]) -> dict[str, dict[str, Any]]:
    """Parallel fundamentals fetch — ~10x faster than sequential."""
    result: dict[str, dict[str, Any]] = {}
    def _one(tkr):
        try:
            return tkr, fetch_fundamentals(yf.Ticker(tkr))
        except Exception:
            return tkr, {}
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
        for fut in as_completed([ex.submit(_one, t) for t in tickers]):
            tkr, data = fut.result()
            result[tkr] = data
    return result


def download_reference_prices(start_date: str) -> dict[str, pd.DataFrame]:
    logger.info("Downloading reference prices (SPY + sector ETFs)...")
    try:
        raw = yf.download(
            _REFERENCE_TICKERS, start=start_date, auto_adjust=True,
            group_by="ticker", progress=False,
        )
        result: dict[str, pd.DataFrame] = {}
        for tkr in _REFERENCE_TICKERS:
            try:
                df = raw[tkr].dropna(how="all")
                if not df.empty:
                    result[tkr] = df
            except KeyError:
                pass
        return result
    except Exception as exc:
        logger.warning("Reference price download failed: %s", exc)
        return {}


# =====================================================================
# INDICATORS — technical signal computation
# =====================================================================


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    if len(close) < period + 1:
        return pd.Series(np.nan, index=close.index)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(100.0)


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    if len(close) < slow + signal:
        nan_s = pd.Series(np.nan, index=close.index)
        return nan_s, nan_s, nan_s
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def compute_bb_pct(close: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.Series:
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    bw = (upper - lower).replace(0.0, np.nan)
    return ((close - lower) / bw).fillna(0.5)


def compute_volume_trend(volume: pd.Series, avg_period: int = 20) -> float:
    if len(volume) < avg_period:
        return 1.0
    avg_vol = volume.rolling(avg_period).mean().iloc[-1]
    if pd.isna(avg_vol) or avg_vol == 0:
        return 1.0
    return float(volume.iloc[-1] / avg_vol)


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def compute_volatility_percentile(close: pd.Series, vol_window: int = 20, lookback: int = 252) -> float:
    returns = close.pct_change()
    rolling_vol = returns.rolling(vol_window).std() * np.sqrt(252)
    rolling_vol = rolling_vol.dropna()
    if len(rolling_vol) < 2:
        return np.nan
    hist = rolling_vol.iloc[-lookback:]
    current = float(rolling_vol.iloc[-1])
    return float((hist < current).mean())


def compute_multi_horizon_momentum(close: pd.Series) -> dict[str, float]:
    result: dict[str, float] = {
        "mom_5d": np.nan, "mom_10d": np.nan, "mom_20d": np.nan, "mom_alignment": np.nan,
    }
    if len(close) < 21:
        return result
    result["mom_5d"] = float(close.iloc[-1] / close.iloc[-6] - 1) if len(close) >= 6 else np.nan
    result["mom_10d"] = float(close.iloc[-1] / close.iloc[-11] - 1) if len(close) >= 11 else np.nan
    result["mom_20d"] = float(close.iloc[-1] / close.iloc[-21] - 1) if len(close) >= 21 else np.nan
    vals = [v for v in [result["mom_5d"], result["mom_10d"], result["mom_20d"]] if not np.isnan(v)]
    if vals:
        result["mom_alignment"] = float(sum(1 for v in vals if v > 0) / len(vals))
    return result


def compute_momentum_acceleration(close: pd.Series) -> float:
    if len(close) < 25:
        return np.nan
    return float(
        (close.iloc[-1] / close.iloc[-11] - 1) - (close.iloc[-11] / close.iloc[-21] - 1)
    )


def compute_drift_consistency(close: pd.Series, window: int = 20) -> float:
    if len(close) < window + 1:
        return np.nan
    rets = close.pct_change().iloc[-window:]
    mu, sigma = float(rets.mean()), float(rets.std())
    if sigma == 0 or np.isnan(sigma):
        return np.nan
    return float(mu / sigma)


def compute_rsi_regime(rsi: pd.Series, window: int = 20) -> float:
    valid = rsi.dropna()
    if len(valid) < window:
        return np.nan
    return float((valid.iloc[-window:] > 50).mean())


def compute_gap_analysis(open_: pd.Series, close: pd.Series, window: int = 20) -> dict[str, float]:
    if len(close) < window + 2:
        return {"gap_up_count": 0, "gap_pct_avg": 0.0}
    gap_pct = (open_ - close.shift(1)) / close.shift(1)
    gap_ups = gap_pct.iloc[-window:].dropna()
    gap_ups = gap_ups[gap_ups > 0.005]
    return {
        "gap_up_count": len(gap_ups),
        "gap_pct_avg": float(gap_ups.mean()) if len(gap_ups) > 0 else 0.0,
    }


def compute_sharpe_63d(close: pd.Series) -> float:
    if len(close) < 65:
        return np.nan
    ret = close.iloc[-1] / close.iloc[-64] - 1
    vol = float(close.pct_change().iloc[-63:].std()) * np.sqrt(252)
    if vol == 0 or np.isnan(vol):
        return np.nan
    return float(ret / vol)


def compute_golden_cross(sma50: pd.Series, sma200: pd.Series, lookback: int = 5) -> dict[str, Any]:
    result: dict[str, Any] = {
        "is_golden_cross": False, "crossed_recently": False, "cross_direction": 0,
    }
    valid = sma50.notna() & sma200.notna()
    if valid.sum() < lookback + 1:
        return result
    s50, s200 = sma50[valid], sma200[valid]
    result["is_golden_cross"] = bool(s50.iloc[-1] > s200.iloc[-1])
    diff = s50 - s200
    recent_diff = diff.iloc[-(lookback + 1):]
    sign_changes = recent_diff.shift(1) * recent_diff < 0
    if sign_changes.any():
        result["crossed_recently"] = True
        last_cross_idx = sign_changes[sign_changes].index[-1]
        result["cross_direction"] = 1 if diff.loc[last_cross_idx] > 0 else -1
    return result


def compute_all_indicators(df: pd.DataFrame, cfg: Any) -> pd.DataFrame:
    close, volume = df["Close"], df["Volume"]
    high = df["High"] if "High" in df.columns else close
    low = df["Low"] if "Low" in df.columns else close
    open_ = df["Open"] if "Open" in df.columns else close

    ind = pd.DataFrame(index=df.index)
    ind["close"] = close
    ind["ret_5d"] = close.pct_change(5)
    ind["ret_63"] = close.pct_change(63)
    ind["ret_126"] = close.pct_change(126)
    ind["sma50"] = close.rolling(50).mean()
    ind["sma200"] = close.rolling(200).mean()
    ind["vol20"] = close.pct_change().rolling(20).std() * np.sqrt(252)
    ind["rsi"] = compute_rsi(close, period=cfg.rsi_period)
    _, _, macd_hist = compute_macd(
        close, fast=cfg.macd_fast, slow=cfg.macd_slow, signal=cfg.macd_signal_period,
    )
    ind["macd_hist"] = macd_hist
    ind["bb_pct"] = compute_bb_pct(close, period=cfg.bb_period, num_std=cfg.bb_std)
    ind["volume_trend"] = compute_volume_trend(volume, avg_period=cfg.volume_avg_period)

    cross = compute_golden_cross(ind["sma50"], ind["sma200"], lookback=cfg.golden_cross_lookback)
    ind["is_golden_cross"] = cross["is_golden_cross"]
    ind["crossed_recently"] = cross["crossed_recently"]
    ind["cross_direction"] = cross["cross_direction"]

    ind["atr"] = compute_atr(high, low, close, period=getattr(cfg, "atr_period", 14))
    ind["vol_percentile"] = compute_volatility_percentile(close)

    mhm = compute_multi_horizon_momentum(close)
    for k, v in mhm.items():
        ind[k] = v

    ind["mom_accel"] = compute_momentum_acceleration(close)
    ind["drift_consistency"] = compute_drift_consistency(close)
    ind["rsi_bull_regime"] = compute_rsi_regime(ind["rsi"])
    ind["sharpe_63d"] = compute_sharpe_63d(close)

    gaps = compute_gap_analysis(open_, close)
    ind["gap_up_count"] = gaps["gap_up_count"]
    ind["gap_pct_avg"] = gaps["gap_pct_avg"]

    # ── VALUE / DISCOUNT SIGNALS — for buying solid stocks at a discount ──
    current_price = float(close.iloc[-1])

    # Distance from 52-week high (negative = discounted; -0.20 = 20% below high)
    if len(close) >= 252:
        high_52w = float(close.iloc[-252:].max())
        ind["pct_from_52wk_high"] = (current_price - high_52w) / high_52w if high_52w > 0 else 0.0
    else:
        high_52w = float(close.max())
        ind["pct_from_52wk_high"] = (current_price - high_52w) / high_52w if high_52w > 0 else 0.0

    # Distance from all-time-high in the data
    ath = float(close.max())
    ind["pct_from_ath"] = (current_price - ath) / ath if ath > 0 else 0.0

    # Premium/discount to 200-SMA
    if len(close) >= 200:
        sma200_now = float(close.rolling(200).mean().iloc[-1])
        ind["pct_vs_sma200"] = (current_price - sma200_now) / sma200_now if sma200_now > 0 else 0.0
    else:
        ind["pct_vs_sma200"] = 0.0

    # 52-week range position (0 = at low, 1 = at high)
    if len(close) >= 252:
        low_52w = float(close.iloc[-252:].min())
        rng = high_52w - low_52w
        ind["range_position_52w"] = (current_price - low_52w) / rng if rng > 0 else 0.5
    else:
        ind["range_position_52w"] = 0.5

    return ind


def extract_latest_signals(ind: pd.DataFrame) -> dict[str, float]:
    row = ind.iloc[-1]
    signals: dict[str, float] = {}
    for col in ind.columns:
        val = row[col]
        if isinstance(val, (bool, np.bool_)):
            signals[col] = int(val)
        elif pd.isna(val):
            signals[col] = np.nan
        else:
            signals[col] = float(val)
    return signals


# =====================================================================
# MACRO — VIX, yield curve, dollar, commodities
# =====================================================================

_VIX_TICKER = "^VIX"
_TNX_TICKER = "^TNX"
_IRX_TICKER = "^IRX"
_DXY_TICKER = "DX-Y.NYB"
_OIL_TICKER = "CL=F"
_COPPER_TICKER = "HG=F"
_LOOKBACK_DAYS = 90


def _latest(ticker: str, lookback_days: int = _LOOKBACK_DAYS) -> tuple[float, float]:
    start = (date.today() - timedelta(days=lookback_days + 10)).isoformat()
    try:
        df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
        if df is None or df.empty:
            return np.nan, np.nan
        close = df["Close"].dropna()
        if len(close) < 2:
            return float(close.iloc[-1]), np.nan
        return float(close.iloc[-1].item()), float(close.iloc[0].item())
    except Exception:
        return np.nan, np.nan


def _pct_change(latest: float, prior: float) -> float:
    if np.isnan(latest) or np.isnan(prior) or prior == 0:
        return np.nan
    return (latest - prior) / abs(prior)


def fetch_vix() -> dict[str, Any]:
    latest, prior = _latest(_VIX_TICKER)
    if np.isnan(latest):
        return {
            "vix": None, "vix_regime": "UNKNOWN", "vix_3m_change": None,
            "vix_trending_up": False, "macro_score_contribution": 0.0, "flag": None,
        }
    vix_change = latest - prior if not np.isnan(prior) else np.nan
    vix_trending_up = (not np.isnan(vix_change)) and vix_change > 3.0

    if latest < 18:
        regime, score, flag = "CALM", 0.10, None
    elif latest < 25:
        regime, score, flag = "ELEVATED", 0.0, None
    elif latest < 35:
        regime, score, flag = "FEAR", -0.10, f"VIX {latest:.1f} — elevated market fear"
    elif latest < 45:
        regime, score, flag = "PANIC", -0.20, f"VIX {latest:.1f} — panic level"
    else:
        regime, score, flag = "CRISIS", -0.40, f"VIX {latest:.1f} — CRISIS level fear"

    if vix_trending_up and regime in ("FEAR", "PANIC", "CRISIS"):
        score -= 0.05

    return {
        "vix": round(latest, 2), "vix_regime": regime,
        "vix_3m_change": round(vix_change, 2) if not np.isnan(vix_change) else None,
        "vix_trending_up": vix_trending_up,
        "macro_score_contribution": float(np.clip(score, -1.0, 1.0)),
        "flag": flag,
    }


def fetch_yield_curve() -> dict[str, Any]:
    tnx_latest, tnx_prior = _latest(_TNX_TICKER)
    irx_latest, _ = _latest(_IRX_TICKER)
    yield_10y = tnx_latest if not np.isnan(tnx_latest) else None
    yield_3m = irx_latest if not np.isnan(irx_latest) else None
    spread: float | None = None
    curve_signal, flag, score = "UNKNOWN", None, 0.0

    if yield_10y is not None and yield_3m is not None:
        spread = round(yield_10y - yield_3m, 3)
        if spread > 1.0:
            curve_signal, score = "NORMAL", 0.05
        elif spread > 0.0:
            curve_signal, score, flag = "FLAT", -0.05, f"Yield curve flat ({spread:+.2f}pp)"
        elif spread > -0.50:
            curve_signal, score, flag = "INVERTED", -0.15, f"Yield curve INVERTED ({spread:+.2f}pp)"
        else:
            curve_signal, score, flag = (
                "DEEPLY_INVERTED", -0.25,
                f"Yield curve DEEPLY INVERTED ({spread:+.2f}pp)",
            )

    rate_change_3m: float | None = None
    rate_trend = "STABLE"
    if not np.isnan(tnx_latest) and not np.isnan(tnx_prior):
        rate_change_3m = round(tnx_latest - tnx_prior, 3)
        if rate_change_3m > 0.50:
            rate_trend = "RISING"
            score -= 0.10
        elif rate_change_3m < -0.50:
            rate_trend = "FALLING"
            score += 0.08

    return {
        "yield_10y": round(yield_10y, 3) if yield_10y is not None else None,
        "yield_3m": round(yield_3m, 3) if yield_3m is not None else None,
        "spread": spread, "curve_signal": curve_signal, "rate_trend": rate_trend,
        "rate_change_3m": rate_change_3m,
        "macro_score_contribution": float(np.clip(score, -1.0, 1.0)),
        "flag": flag.strip(" |") if flag else None,
    }


def fetch_dollar_trend() -> dict[str, Any]:
    latest, prior = _latest(_DXY_TICKER)
    if np.isnan(latest):
        return {
            "dxy_latest": None, "dxy_3m_change_pct": None,
            "dollar_signal": "UNKNOWN", "macro_score_contribution": 0.0, "flag": None,
        }
    change_pct = _pct_change(latest, prior)
    flag, score = None, 0.0
    if np.isnan(change_pct):
        signal = "UNKNOWN"
    elif change_pct > 0.04:
        signal, score, flag = "STRENGTHENING", -0.08, f"USD strengthening ({change_pct:+.1%} in 3m)"
    elif change_pct > 0.01:
        signal, score = "STABLE_STRONG", -0.02
    elif change_pct < -0.04:
        signal, score = "WEAKENING", 0.05
    else:
        signal = "STABLE"
    return {
        "dxy_latest": round(latest, 2),
        "dxy_3m_change_pct": round(change_pct, 4) if not np.isnan(change_pct) else None,
        "dollar_signal": signal,
        "macro_score_contribution": float(np.clip(score, -1.0, 1.0)),
        "flag": flag,
    }


def fetch_commodity_signals() -> dict[str, Any]:
    oil_latest, oil_prior = _latest(_OIL_TICKER)
    copper_latest, copper_prior = _latest(_COPPER_TICKER)
    oil_change = _pct_change(oil_latest, oil_prior)
    copper_change = _pct_change(copper_latest, copper_prior)

    oil_trend = (
        "UNKNOWN" if np.isnan(oil_change)
        else ("RISING" if oil_change > 0.06 else ("FALLING" if oil_change < -0.06 else "STABLE"))
    )
    copper_trend = (
        "UNKNOWN" if np.isnan(copper_change)
        else ("RISING" if copper_change > 0.06 else ("FALLING" if copper_change < -0.06 else "STABLE"))
    )

    flag, score = None, 0.0
    if copper_trend == "RISING" and oil_trend != "FALLING":
        commodity_grade, score = "GROWTH", 0.08
    elif copper_trend == "FALLING":
        commodity_grade, score, flag = "CONTRACTION", -0.12, "Copper falling — contraction signal"
    elif oil_trend == "RISING":
        commodity_grade, score, flag = "NEUTRAL", -0.04, "Oil rising — inflation pressure"
    else:
        commodity_grade = "NEUTRAL"

    return {
        "oil_latest": round(oil_latest, 2) if not np.isnan(oil_latest) else None,
        "oil_3m_change_pct": round(oil_change, 4) if not np.isnan(oil_change) else None,
        "oil_trend": oil_trend,
        "copper_latest": round(copper_latest, 4) if not np.isnan(copper_latest) else None,
        "copper_3m_change_pct": round(copper_change, 4) if not np.isnan(copper_change) else None,
        "copper_trend": copper_trend,
        "commodity_grade": commodity_grade,
        "macro_score_contribution": float(np.clip(score, -1.0, 1.0)),
        "flag": flag,
    }


def fetch_macro_environment() -> dict[str, Any]:
    logger.info("Fetching macro environment...")
    vix_data = fetch_vix()
    yc_data = fetch_yield_curve()
    dollar_data = fetch_dollar_trend()
    commodity_data = fetch_commodity_signals()

    macro_score = float(np.clip(
        0.35 * vix_data.get("macro_score_contribution", 0.0)
        + 0.35 * yc_data.get("macro_score_contribution", 0.0)
        + 0.20 * dollar_data.get("macro_score_contribution", 0.0)
        + 0.10 * commodity_data.get("macro_score_contribution", 0.0),
        -1.0, 1.0,
    ))

    if macro_score >= 0.08:
        macro_grade = "TAILWIND"
    elif macro_score >= -0.05:
        macro_grade = "NEUTRAL"
    elif macro_score >= -0.20:
        macro_grade = "HEADWIND"
    else:
        macro_grade = "CRISIS"

    blocks_high = vix_data.get("vix_regime") == "CRISIS"
    all_flags = [
        f for f in [
            vix_data.get("flag"), yc_data.get("flag"),
            dollar_data.get("flag"), commodity_data.get("flag"),
        ] if f
    ]

    vix_val = vix_data.get("vix")
    vix_regime = vix_data.get("vix_regime", "UNKNOWN")
    spread = yc_data.get("spread")
    commodity_grade = commodity_data.get("commodity_grade", "UNKNOWN")
    vix_str = f"VIX {vix_val:.1f} ({vix_regime.lower()})" if vix_val else "VIX unknown"
    curve_str = (
        f"yield curve {yc_data.get('curve_signal', 'UNKNOWN').lower().replace('_', ' ')}"
        if spread is not None else "curve unknown"
    )
    summary = f"Macro {macro_grade}: {vix_str}, {curve_str}, commodities {commodity_grade.lower()}"

    return {
        "vix": vix_data, "yield_curve": yc_data, "dollar": dollar_data,
        "commodities": commodity_data, "macro_score": macro_score,
        "macro_grade": macro_grade, "blocks_high_confidence": blocks_high,
        "all_flags": all_flags, "summary": summary,
    }


# =====================================================================
# MARKET REGIME — SPY-based regime detection
# =====================================================================

MARKET_PROXY = "SPY"


class Regime(str, Enum):
    BULL_QUIET = "BULL_QUIET"
    BULL_VOLATILE = "BULL_VOLATILE"
    BEAR_QUIET = "BEAR_QUIET"
    BEAR_VOLATILE = "BEAR_VOLATILE"
    CRISIS = "CRISIS"


@dataclass
class RegimeState:
    regime: Regime
    spy_price: float
    spy_sma200: float
    spy_above_sma200: bool
    realised_vol: float
    spy_ret_63: float
    spy_ret_126: float
    description: str
    weight_overrides: dict[str, Any]
    confidence_override: float
    position_size_factor: float


_REGIME_PROFILES: dict[Regime, dict[str, Any]] = {
    Regime.BULL_QUIET: {
        "description": "Bull market, low volatility — standard strategy",
        "weight_overrides": {},
        "confidence_override": 0.75,
        "position_size_factor": 1.0,
    },
    Regime.BULL_VOLATILE: {
        "description": "Bull trend intact but volatility elevated",
        "weight_overrides": {
            "weights": {"technical": 0.35, "fundamental": 0.65},
            "technical_weights": {
                "momentum": 0.10, "trend": 0.08, "rsi": 0.07,
                "macd": 0.06, "bb_pct": 0.02, "volume": 0.02,
            },
        },
        "confidence_override": 0.75,
        "position_size_factor": 0.80,
    },
    Regime.BEAR_QUIET: {
        "description": "Downtrend underway — favour quality and value",
        "weight_overrides": {
            "weights": {"technical": 0.30, "fundamental": 0.70},
            "fundamental_weights": {
                "value": 0.18, "quality": 0.22, "growth": 0.08,
                "fcf_yield": 0.15, "leverage_penalty": -0.07,
            },
        },
        "confidence_override": 0.78,
        "position_size_factor": 0.65,
    },
    Regime.BEAR_VOLATILE: {
        "description": "Confirmed downtrend with elevated fear",
        "weight_overrides": {
            "weights": {"technical": 0.20, "fundamental": 0.80},
            "fundamental_weights": {
                "value": 0.20, "quality": 0.28, "growth": 0.05,
                "fcf_yield": 0.18, "leverage_penalty": -0.09,
            },
        },
        "confidence_override": 0.82,
        "position_size_factor": 0.45,
    },
    Regime.CRISIS: {
        "description": "Market crisis — only highest-conviction buys",
        "weight_overrides": {
            "weights": {"technical": 0.15, "fundamental": 0.85},
            "fundamental_weights": {
                "value": 0.20, "quality": 0.35, "growth": 0.02,
                "fcf_yield": 0.20, "leverage_penalty": -0.12,
            },
        },
        "confidence_override": 0.85,
        "position_size_factor": 0.25,
    },
}


def classify_regime(
    spy_price: float, spy_sma200: float, realised_vol: float, vol_crisis: float = 0.35,
) -> Regime:
    if realised_vol >= vol_crisis:
        return Regime.CRISIS
    if spy_price > spy_sma200:
        return Regime.BULL_QUIET if realised_vol < 0.15 else Regime.BULL_VOLATILE
    else:
        return Regime.BEAR_QUIET if realised_vol < 0.20 else Regime.BEAR_VOLATILE


def detect_regime(
    lookback_days: int = 252, vol_window: int = 20, sma_window: int = 200,
) -> RegimeState:
    try:
        spy_df = yf.download(
            MARKET_PROXY, period=f"{lookback_days + 50}d",
            auto_adjust=True, progress=False,
        )
        if spy_df.empty or len(spy_df) < sma_window:
            raise ValueError("Insufficient SPY data")
    except Exception as exc:
        logger.error("SPY download failed: %s — defaulting to BULL_QUIET", exc)
        return _fallback_regime()

    close = spy_df["Close"].squeeze()
    spy_price = float(close.iloc[-1])
    spy_sma200 = float(close.rolling(sma_window).mean().iloc[-1])
    realised_vol = float(close.pct_change().rolling(vol_window).std().iloc[-1] * np.sqrt(252))
    spy_ret_63 = float(close.pct_change(63).iloc[-1]) if len(close) >= 64 else 0.0
    spy_ret_126 = float(close.pct_change(126).iloc[-1]) if len(close) >= 127 else 0.0

    regime = classify_regime(spy_price, spy_sma200, realised_vol)
    profile = _REGIME_PROFILES[regime]

    state = RegimeState(
        regime=regime, spy_price=spy_price, spy_sma200=spy_sma200,
        spy_above_sma200=spy_price > spy_sma200, realised_vol=realised_vol,
        spy_ret_63=spy_ret_63, spy_ret_126=spy_ret_126,
        description=profile["description"],
        weight_overrides=profile["weight_overrides"],
        confidence_override=profile["confidence_override"],
        position_size_factor=profile["position_size_factor"],
    )
    logger.info(
        "Market regime: %s | SPY %.2f vs SMA200 %.2f | Vol: %.1f%%",
        regime.value, spy_price, spy_sma200, realised_vol * 100,
    )
    return state


def apply_regime_to_weights(base_weights: dict[str, Any], regime_state: RegimeState) -> dict[str, Any]:
    merged = copy.deepcopy(base_weights)
    for section, overrides in regime_state.weight_overrides.items():
        if section in merged and isinstance(overrides, dict):
            merged[section].update(overrides)
        else:
            merged[section] = overrides
    return merged


def _fallback_regime() -> RegimeState:
    profile = _REGIME_PROFILES[Regime.BULL_QUIET]
    return RegimeState(
        regime=Regime.BULL_QUIET, spy_price=float("nan"), spy_sma200=float("nan"),
        spy_above_sma200=True, realised_vol=0.12, spy_ret_63=0.0, spy_ret_126=0.0,
        description="Fallback — using standard weights", weight_overrides={},
        confidence_override=profile["confidence_override"],
        position_size_factor=profile["position_size_factor"],
    )


# =====================================================================
# NEWS — VADER sentiment with financial lexicon
# =====================================================================

_analyzer = SentimentIntensityAnalyzer()
_FINANCIAL_LEXICON: dict[str, float] = {
    "beats": 2.0, "misses": -2.0, "miss": -2.0, "beat": 2.0,
    "exceeds": 1.8, "shortfall": -2.0, "layoffs": -2.5, "layoff": -2.5,
    "bankruptcy": -4.0, "default": -3.0, "probe": -2.0, "investigation": -1.8,
    "fraud": -4.0, "recall": -2.5, "downgrade": -2.5, "upgrade": 2.5,
    "raised": 1.5, "lowered": -1.5, "guidance": 0.5, "dividend": 1.0,
    "buyback": 1.5, "dilution": -2.0, "dilutive": -2.0, "acquisition": 1.0,
    "merger": 1.0, "breakup": -1.0, "spinoff": 0.5, "rally": 2.0,
    "plunge": -2.5, "surge": 2.0, "tumble": -2.0, "crash": -3.0,
    "soar": 2.0, "slump": -2.0, "halted": -2.5, "suspended": -2.0,
    "delisted": -4.0, "tariff": -1.5, "tariffs": -1.5, "sanction": -2.0,
    "sanctions": -2.0, "lawsuit": -2.0, "settlement": -1.0, "fine": -1.5,
}
for _word, _score in _FINANCIAL_LEXICON.items():
    _analyzer.lexicon[_word] = _score


def fetch_news(ticker: str, max_age_hours: int = 72) -> list[dict[str, Any]]:
    try:
        raw = yf.Ticker(ticker).news or []
    except Exception:
        return []
    now_ts = datetime.now(tz=timezone.utc).timestamp()
    results: list[dict[str, Any]] = []
    for item in raw:
        title = item.get("title", "").strip()
        if not title:
            continue
        pub_ts = item.get("providerPublishTime", 0)
        age_hours = (now_ts - pub_ts) / 3600 if pub_ts else float("inf")
        if age_hours > max_age_hours:
            continue
        results.append({
            "title": title,
            "publisher": item.get("publisher", "unknown"),
            "published_at": datetime.fromtimestamp(pub_ts, tz=timezone.utc) if pub_ts else None,
            "age_hours": age_hours,
        })
    return results


def score_headline(title: str) -> float:
    return float(_analyzer.polarity_scores(title)["compound"])


def aggregate_news_sentiment(
    headlines: list[dict[str, Any]], recency_decay: float = 0.15,
) -> dict[str, Any]:
    if not headlines:
        return {
            "score": 0.0, "n_headlines": 0, "n_negative": 0, "n_positive": 0,
            "worst_headline": "", "best_headline": "", "signal": "NEUTRAL",
        }
    scored = []
    for item in headlines:
        age = item.get("age_hours", 24.0)
        weight = float(2 ** (-recency_decay * age))
        s = score_headline(item["title"])
        scored.append((weight, s, item["title"]))

    total_weight = sum(w for w, _, _ in scored)
    weighted_score = sum(w * s for w, s, _ in scored) / (total_weight + 1e-9)
    scores_only = [s for _, s, _ in scored]
    worst_title = scored[scores_only.index(min(scores_only))][2]
    best_title = scored[scores_only.index(max(scores_only))][2]
    n_neg = sum(1 for s in scores_only if s < -0.2)
    n_pos = sum(1 for s in scores_only if s > 0.2)

    if weighted_score > 0.15:
        signal = "BULLISH"
    elif weighted_score < -0.15:
        signal = "BEARISH"
    else:
        signal = "NEUTRAL"

    return {
        "score": round(weighted_score, 4), "n_headlines": len(headlines),
        "n_negative": n_neg, "n_positive": n_pos,
        "worst_headline": worst_title, "best_headline": best_title,
        "signal": signal,
    }


def fetch_and_score_all(
    tickers: list[str], max_age_hours: int = 72, recency_decay: float = 0.15,
) -> dict[str, dict[str, Any]]:
    """Parallel news fetch & sentiment scoring."""
    results: dict[str, dict[str, Any]] = {}
    def _one(tkr):
        try:
            return tkr, aggregate_news_sentiment(
                fetch_news(tkr, max_age_hours=max_age_hours),
                recency_decay=recency_decay,
            )
        except Exception:
            return tkr, {"score": 0.0, "n_headlines": 0, "n_negative": 0,
                         "n_positive": 0, "worst_headline": "", "best_headline": "",
                         "signal": "NEUTRAL"}
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
        for fut in as_completed([ex.submit(_one, t) for t in tickers]):
            tkr, data = fut.result()
            results[tkr] = data
    return results


# =====================================================================
# LEADERSHIP — CEO tenure, dual transitions, CEO news sentiment
# =====================================================================

_CEO_TITLES = {
    "chief executive officer", "ceo", "president & ceo",
    "ceo & president", "co-ceo", "executive chairman", "president and ceo",
}
_CFO_TITLES = {"chief financial officer", "cfo", "chief financial officer (cfo)"}
_COO_TITLES = {"chief operating officer", "coo"}


def fetch_officers(ticker_obj: yf.Ticker) -> list[dict[str, Any]]:
    try:
        info = ticker_obj.info or {}
        officers = info.get("companyOfficers", [])
        return officers if isinstance(officers, list) else []
    except Exception:
        return []


def identify_key_executives(officers: list[dict[str, Any]]) -> dict[str, dict[str, Any] | None]:
    result: dict[str, dict[str, Any] | None] = {"ceo": None, "cfo": None, "coo": None}
    for officer in officers:
        title_raw = str(officer.get("title", "")).lower().strip()
        if result["ceo"] is None and any(t in title_raw for t in _CEO_TITLES):
            result["ceo"] = officer
        elif result["cfo"] is None and any(t in title_raw for t in _CFO_TITLES):
            result["cfo"] = officer
        elif result["coo"] is None and any(t in title_raw for t in _COO_TITLES):
            result["coo"] = officer
    return result


def assess_ceo_tenure(ceo: dict[str, Any] | None) -> dict[str, Any]:
    if ceo is None:
        return {
            "name": "Unknown", "title": "Unknown",
            "estimated_tenure_years": None, "is_new": False,
            "pay_usd": None, "age": None,
        }
    name = ceo.get("name", "Unknown")
    title = ceo.get("title", "Unknown")
    pay = ceo.get("totalPay")
    age = ceo.get("age")
    fiscal_year = ceo.get("fiscalYear")
    current_year = datetime.now().year
    estimated_tenure_years: float | None = None
    is_new = False
    if fiscal_year is not None:
        try:
            fy = int(fiscal_year)
            estimated_tenure_years = float(current_year - fy)
            is_new = fy >= current_year - 1
        except (ValueError, TypeError):
            pass
    return {
        "name": name, "title": title,
        "estimated_tenure_years": estimated_tenure_years,
        "is_new": is_new,
        "pay_usd": float(pay) if pay is not None else None,
        "age": int(age) if age is not None else None,
    }


def fetch_ceo_news_sentiment(
    ceo_name: str, ticker: str, max_age_hours: int = 168,
) -> dict[str, Any]:
    _empty = {
        "score": 0.0, "n_headlines": 0, "n_negative": 0, "n_positive": 0,
        "worst_headline": "", "best_headline": "", "signal": "NEUTRAL",
    }
    if not ceo_name or ceo_name == "Unknown":
        return _empty
    try:
        raw = yf.Ticker(ticker).news or []
    except Exception:
        return _empty
    now_ts = datetime.now(tz=timezone.utc).timestamp()
    last_name = ceo_name.split()[-1].lower() if ceo_name != "Unknown" else ""
    first_name = ceo_name.split()[0].lower() if len(ceo_name.split()) > 1 else ""
    ceo_headlines = []
    for item in raw:
        title = item.get("title", "").strip()
        if not title:
            continue
        pub_ts = item.get("providerPublishTime", 0)
        age_hours = (now_ts - pub_ts) / 3600 if pub_ts else float("inf")
        if age_hours > max_age_hours:
            continue
        title_lower = title.lower()
        name_match = (
            (first_name in title_lower and last_name in title_lower)
            or (last_name in title_lower and (
                "ceo" in title_lower or "chief" in title_lower or "executive" in title_lower
            ))
        )
        if name_match:
            ceo_headlines.append({"title": title, "age_hours": age_hours})
    if not ceo_headlines:
        return _empty
    return aggregate_news_sentiment(ceo_headlines)


def analyze_leadership(ticker_obj: yf.Ticker, ticker: str) -> dict[str, Any]:
    officers = fetch_officers(ticker_obj)
    executives = identify_key_executives(officers)
    ceo_info = assess_ceo_tenure(executives["ceo"])
    ceo_news = fetch_ceo_news_sentiment(ceo_info["name"], ticker)
    time.sleep(0.1)

    flags: list[str] = []
    score_components: list[float] = []

    if ceo_info["is_new"]:
        flags.append(f"NEW CEO: {ceo_info['name']} appears recently appointed")
        score_components.append(-0.30)
    else:
        score_components.append(0.10)

    cfo_raw = executives["cfo"]
    cfo_is_new = False
    if cfo_raw is not None:
        fy = cfo_raw.get("fiscalYear")
        if fy is not None:
            try:
                cfo_is_new = int(fy) >= datetime.now().year - 1
            except (ValueError, TypeError):
                pass

    dual_transition = ceo_info["is_new"] and cfo_is_new
    if dual_transition:
        flags.append("DUAL LEADERSHIP TRANSITION: Both CEO and CFO recently replaced")
        score_components.append(-0.20)

    ceo_news_score = float(ceo_news.get("score", 0.0))
    ceo_news_signal = ceo_news.get("signal", "NEUTRAL")
    n_ceo_headlines = int(ceo_news.get("n_headlines", 0))
    if n_ceo_headlines > 0:
        if ceo_news_signal == "BEARISH":
            flags.append(f"NEGATIVE CEO COVERAGE: {ceo_info['name']} score {ceo_news_score:+.2f}")
            score_components.append(ceo_news_score * 0.25)
        elif ceo_news_signal == "BULLISH":
            flags.append(f"POSITIVE CEO COVERAGE: {ceo_info['name']} score {ceo_news_score:+.2f}")
            score_components.append(ceo_news_score * 0.25)

    leadership_score = float(sum(score_components) / max(len(score_components), 1))
    leadership_score = max(-1.0, min(1.0, leadership_score))
    summary = f"{ceo_info['name']} — {'new CEO' if ceo_info['is_new'] else 'stable tenure'}"

    return {
        "ceo": ceo_info, "cfo": cfo_raw, "coo": executives["coo"],
        "ceo_news": ceo_news, "dual_transition_risk": dual_transition,
        "leadership_score": leadership_score, "flags": flags, "summary": summary,
    }


def fetch_all_leadership(tickers: list[str]) -> dict[str, dict[str, Any]]:
    """Parallel leadership analysis."""
    results: dict[str, dict[str, Any]] = {}
    def _one(tkr):
        try:
            return tkr, analyze_leadership(yf.Ticker(tkr), tkr)
        except Exception:
            return tkr, {"leadership_score": 0.0, "flags": [], "summary": "N/A",
                         "ceo": {"name": "Unknown"}, "cfo": None, "coo": None,
                         "ceo_news": {}, "dual_transition_risk": False}
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
        for fut in as_completed([ex.submit(_one, t) for t in tickers]):
            tkr, data = fut.result()
            results[tkr] = data
    return results


# =====================================================================
# COMPANY ANALYSIS — moat, industry growth, CEO reinvestment
# =====================================================================

_HIGH_GROWTH_INDUSTRIES: set[str] = {
    "Semiconductors", "Semiconductor Equipment & Materials",
    "Software—Application", "Software—Infrastructure",
    "Internet Content & Information", "Electronic Gaming & Multimedia",
    "Biotechnology", "Medical Devices",
    "Drug Manufacturers—Specialty & Generic", "Diagnostics & Research",
    "Health Information Services", "Solar", "Renewable Utilities",
    "Electrical Equipment & Parts", "Communication Equipment",
    "Aerospace & Defense", "Scientific & Technical Instruments",
    "Computer Hardware", "Information Technology Services", "Data Storage",
}
_MEDIUM_GROWTH_INDUSTRIES: set[str] = {
    "Healthcare Plans", "Medical Care Facilities",
    "Drug Manufacturers—General", "Consumer Electronics",
    "Specialty Retail", "Apparel Retail", "Home Improvement Retail",
    "Restaurants", "Financial Data & Stock Exchanges", "Credit Services",
    "Insurance—Life", "Insurance—Property & Casualty",
    "Banks—Regional", "Banks—Diversified", "Auto Manufacturers",
    "Auto Parts", "Oil & Gas Equipment & Services",
    "Oil & Gas Exploration & Production", "Specialty Chemicals",
    "Waste Management", "Industrial Distribution", "Trucking",
    "Airlines", "Lodging", "Entertainment", "Broadcasting",
}
_LOW_GROWTH_INDUSTRIES: set[str] = {
    "Coal", "Thermal Coal", "Tobacco", "Print Media", "Publishing",
    "Department Stores", "Discount Stores",
    "Integrated Telecommunication Services", "Wireline Telecom",
    "Oil & Gas Refining & Marketing", "Conventional Utilities",
    "Electric Utilities", "Gas Utilities", "Staffing & Employment Services",
    "Paper & Paper Products", "Aluminum", "Steel",
}
_HIGH_GROWTH_SECTORS: set[str] = {"Technology", "Healthcare"}
_LOW_GROWTH_SECTORS: set[str] = {"Energy", "Utilities", "Basic Materials"}


def analyze_ceo_reinvestment(ticker_obj: yf.Ticker, info: dict[str, Any]) -> dict[str, Any]:
    rd_intensity: float | None = None
    rd_signal, capex_signal, buyback_signal = "UNKNOWN", "UNKNOWN", "UNKNOWN"
    growth_capex_ratio: float | None = None
    flags: list[str] = []
    score_parts: list[float] = []
    sector = info.get("sector", "")

    # R&D intensity
    try:
        inc = ticker_obj.income_stmt
        if inc is not None and not inc.empty:
            revenue_row, rd_row = None, None
            for key in ["Total Revenue", "TotalRevenue", "Revenue"]:
                if key in inc.index:
                    revenue_row = inc.loc[key]
                    break
            for key in ["Research And Development", "ResearchAndDevelopment", "Research Development", "R&D Expense"]:
                if key in inc.index:
                    rd_row = inc.loc[key]
                    break
            if revenue_row is not None and rd_row is not None:
                rev = _safe(revenue_row.iloc[0])
                rd = abs(_safe(rd_row.iloc[0]))
                if not np.isnan(rev) and not np.isnan(rd) and rev > 0:
                    rd_intensity = rd / rev
                    high_rd = 0.10 if sector in {"Technology", "Healthcare"} else 0.05
                    if rd_intensity >= high_rd:
                        rd_signal = "HIGH"
                        flags.append(f"HIGH R&D INTENSITY: {rd_intensity:.1%} of revenue")
                        score_parts.append(0.5)
                    elif rd_intensity >= 0.03:
                        rd_signal = "MODERATE"
                        score_parts.append(0.1)
                    else:
                        rd_signal = "LOW"
                        score_parts.append(-0.1)
    except Exception:
        pass

    # CapEx vs depreciation
    try:
        cf = ticker_obj.cashflow
        if cf is not None and not cf.empty:
            capex, dep = None, None
            for key in ["Capital Expenditure", "CapitalExpenditure", "Purchase Of PPE", "Purchase of Property Plant And Equipment"]:
                if key in cf.index:
                    capex = abs(_safe(cf.loc[key].iloc[0]))
                    break
            for key in ["Depreciation", "DepreciationAndAmortization", "Depreciation Amortization Depletion", "Depreciation And Amortization"]:
                if key in cf.index:
                    dep = abs(_safe(cf.loc[key].iloc[0]))
                    break
            if capex is not None and dep is not None and not np.isnan(capex) and not np.isnan(dep) and dep > 0:
                growth_capex_ratio = capex / dep
                if growth_capex_ratio >= 1.5:
                    capex_signal = "EXPANDING"
                    flags.append(f"GROWTH CAPEX: {growth_capex_ratio:.1f}x depreciation")
                    score_parts.append(0.4)
                elif growth_capex_ratio >= 1.0:
                    capex_signal = "MAINTAINING"
                    score_parts.append(0.1)
                else:
                    capex_signal = "UNDERINVESTING"
                    score_parts.append(-0.3)
    except Exception:
        pass

    # Share buyback / dilution
    try:
        bs = ticker_obj.balance_sheet
        if bs is not None and not bs.empty:
            for key in ["Common Stock", "Ordinary Shares Number", "Share Issued", "Shares Outstanding"]:
                if key in bs.index:
                    row = bs.loc[key]
                    if isinstance(row, pd.Series) and len(row) >= 2:
                        curr_shares = _safe(row.iloc[0])
                        prev_shares = _safe(row.iloc[1])
                        if not np.isnan(curr_shares) and not np.isnan(prev_shares) and prev_shares > 0:
                            change_pct = (curr_shares - prev_shares) / prev_shares
                            if change_pct < -0.02:
                                buyback_signal = "BUYING_BACK"
                                flags.append(f"SHARE BUYBACKS: shares down {abs(change_pct):.1%}")
                                score_parts.append(0.3)
                            elif change_pct > 0.03:
                                buyback_signal = "DILUTING"
                                flags.append(f"SHARE DILUTION: shares up {change_pct:.1%}")
                                score_parts.append(-0.2)
                            else:
                                buyback_signal = "NEUTRAL"
                                score_parts.append(0.0)
                        break
    except Exception:
        pass

    reinvestment_score = float(np.mean(score_parts)) if score_parts else 0.0
    reinvestment_score = max(-1.0, min(1.0, reinvestment_score))
    return {
        "rd_intensity": rd_intensity, "rd_signal": rd_signal,
        "growth_capex_ratio": growth_capex_ratio, "capex_signal": capex_signal,
        "buyback_signal": buyback_signal, "reinvestment_score": reinvestment_score,
        "flags": flags,
    }


def score_industry_growth_potential(
    fund: dict[str, Any], sector_revenue_growths: dict[str, list[float]] | None = None,
) -> dict[str, Any]:
    industry = fund.get("industry") or ""
    sector = fund.get("sector") or ""
    rev_growth = _safe(fund.get("revenueGrowth"))
    flags: list[str] = []
    growth_tier, growth_score = "UNKNOWN", 0.0

    if industry in _HIGH_GROWTH_INDUSTRIES or sector in _HIGH_GROWTH_SECTORS:
        growth_tier, growth_score = "HIGH", 1.0
        flags.append(f"HIGH-GROWTH INDUSTRY: {industry or sector}")
    elif industry in _LOW_GROWTH_INDUSTRIES or sector in _LOW_GROWTH_SECTORS:
        growth_tier, growth_score = "LOW", -0.5
        flags.append(f"LOW-GROWTH INDUSTRY: {industry or sector}")
    elif industry in _MEDIUM_GROWTH_INDUSTRIES or industry or sector:
        growth_tier, growth_score = "MEDIUM", 0.0

    peer_outperformance: float | None = None
    if sector_revenue_growths and sector in sector_revenue_growths:
        peer_vals = [v for v in sector_revenue_growths[sector] if not np.isnan(v)]
        if peer_vals and not np.isnan(rev_growth):
            sector_median_growth = float(np.median(peer_vals))
            peer_outperformance = float(rev_growth - sector_median_growth)

    return {
        "industry": industry, "sector": sector, "growth_tier": growth_tier,
        "growth_score": growth_score, "peer_outperformance": peer_outperformance,
        "flags": flags,
    }


def score_competitive_moat(
    fund: dict[str, Any], sector_medians: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    gross_margin = _safe(fund.get("grossMargins"))
    operating_margin = _safe(fund.get("operatingMargins"))
    roe = _safe(fund.get("returnOnEquity"))
    sector = fund.get("sector") or "__universe__"
    flags: list[str] = []
    score_parts: list[float] = []

    s_med: dict[str, float] = {}
    if sector_medians:
        s_med = sector_medians.get(sector, sector_medians.get("__universe__", {}))

    gross_margin_vs_sector: float | None = None
    roe_vs_sector: float | None = None

    if not np.isnan(gross_margin):
        s_gross = _safe(s_med.get("grossMargins", np.nan))
        if not np.isnan(s_gross) and s_gross > 0:
            gross_margin_vs_sector = gross_margin - s_gross
            if gross_margin_vs_sector > 0.10:
                score_parts.append(0.6)
            elif gross_margin_vs_sector > 0.03:
                score_parts.append(0.2)
            elif gross_margin_vs_sector < -0.10:
                score_parts.append(-0.4)
            else:
                score_parts.append(0.0)
        else:
            if gross_margin > 0.50:
                score_parts.append(0.5)
            elif gross_margin > 0.30:
                score_parts.append(0.2)
            elif gross_margin < 0.10:
                score_parts.append(-0.2)

    if not np.isnan(roe):
        s_roe = _safe(s_med.get("returnOnEquity", np.nan))
        if not np.isnan(s_roe):
            roe_vs_sector = roe - s_roe
            if roe > 0.20 and roe_vs_sector > 0.05:
                score_parts.append(0.5)
            elif roe > 0.15:
                score_parts.append(0.2)
            elif roe < 0:
                score_parts.append(-0.5)
            else:
                score_parts.append(0.0)
        else:
            if roe > 0.20:
                score_parts.append(0.4)
            elif roe > 0.12:
                score_parts.append(0.1)
            elif roe < 0:
                score_parts.append(-0.5)

    if not np.isnan(operating_margin):
        s_op = _safe(s_med.get("operatingMargins", np.nan))
        if not np.isnan(s_op):
            if operating_margin - s_op > 0.05:
                score_parts.append(0.3)
            elif operating_margin - s_op < -0.05:
                score_parts.append(-0.2)
        else:
            if operating_margin > 0.20:
                score_parts.append(0.3)
            elif operating_margin < 0:
                score_parts.append(-0.2)

    if not score_parts:
        return {
            "moat_grade": "UNKNOWN", "moat_score": 0.0,
            "gross_margin": gross_margin if not np.isnan(gross_margin) else None,
            "operating_margin": operating_margin if not np.isnan(operating_margin) else None,
            "roe": roe if not np.isnan(roe) else None,
            "gross_margin_vs_sector": gross_margin_vs_sector,
            "roe_vs_sector": roe_vs_sector, "flags": flags,
        }

    moat_score = max(-1.0, min(1.0, float(np.mean(score_parts))))
    if moat_score >= 0.45:
        moat_grade = "WIDE"
    elif moat_score >= 0.10:
        moat_grade = "NARROW"
    elif moat_score >= -0.15:
        moat_grade = "NONE"
    else:
        moat_grade = "DISADVANTAGED"

    return {
        "moat_grade": moat_grade, "moat_score": moat_score,
        "gross_margin": gross_margin if not np.isnan(gross_margin) else None,
        "operating_margin": operating_margin if not np.isnan(operating_margin) else None,
        "roe": roe if not np.isnan(roe) else None,
        "gross_margin_vs_sector": gross_margin_vs_sector,
        "roe_vs_sector": roe_vs_sector, "flags": flags,
    }


def analyze_company(
    ticker_obj: yf.Ticker, ticker: str, fund: dict[str, Any],
    sector_medians=None, sector_revenue_growths=None,
) -> dict[str, Any]:
    reinvestment = analyze_ceo_reinvestment(ticker_obj, fund)
    industry_growth = score_industry_growth_potential(fund, sector_revenue_growths)
    moat = score_competitive_moat(fund, sector_medians)

    company_score = float(
        0.40 * moat.get("moat_score", 0.0)
        + 0.35 * industry_growth.get("growth_score", 0.0)
        + 0.25 * reinvestment.get("reinvestment_score", 0.0)
    )
    company_score = max(-1.0, min(1.0, company_score))

    if company_score >= 0.50:
        company_grade = "EXCEPTIONAL"
    elif company_score >= 0.20:
        company_grade = "STRONG"
    elif company_score >= -0.10:
        company_grade = "ADEQUATE"
    else:
        company_grade = "WEAK"

    all_flags = reinvestment.get("flags", []) + industry_growth.get("flags", []) + moat.get("flags", [])
    summary = f"{industry_growth.get('industry') or ticker} — {company_grade.lower()} profile"

    return {
        "reinvestment": reinvestment, "industry_growth": industry_growth,
        "moat": moat, "company_score": company_score,
        "company_grade": company_grade, "all_flags": all_flags, "summary": summary,
    }


def fetch_all_company_analyses(
    tickers: list[str], all_fundamentals: dict[str, dict[str, Any]],
    sector_medians=None,
) -> dict[str, dict[str, Any]]:
    sector_revenue_growths: dict[str, list[float]] = {}
    for fund in all_fundamentals.values():
        sector = fund.get("sector") or "__universe__"
        rev_g = fund.get("revenueGrowth")
        if rev_g is not None:
            try:
                sector_revenue_growths.setdefault(sector, []).append(float(rev_g))
            except (TypeError, ValueError):
                pass

    results: dict[str, dict[str, Any]] = {}
    _empty = {
        "reinvestment": {"reinvestment_score": 0.0, "flags": []},
        "industry_growth": {"growth_tier": "UNKNOWN", "growth_score": 0.0, "flags": []},
        "moat": {"moat_grade": "UNKNOWN", "moat_score": 0.0, "flags": []},
        "company_score": 0.0, "company_grade": "ADEQUATE",
        "all_flags": [], "summary": "Company analysis unavailable",
    }
    def _one(tkr):
        try:
            return tkr, analyze_company(
                yf.Ticker(tkr), tkr, all_fundamentals.get(tkr, {}),
                sector_medians=sector_medians,
                sector_revenue_growths=sector_revenue_growths,
            )
        except Exception:
            return tkr, _empty
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
        for fut in as_completed([ex.submit(_one, t) for t in tickers]):
            tkr, data = fut.result()
            results[tkr] = data
    return results


# =====================================================================
# PRICE STRUCTURE — trend, pullbacks, bases, breakouts
# =====================================================================


def compute_higher_highs_higher_lows(high: pd.Series, low: pd.Series, window: int = 20) -> int:
    if len(high) < window or len(low) < window:
        return 0
    half = window // 2
    hh = int(high.iloc[-half:].max() > high.iloc[-window:-half].max())
    hl = int(low.iloc[-half:].min() > low.iloc[-window:-half].min())
    if hh and hl:
        return 1
    elif not hh and not hl:
        return -1
    return 0


def compute_trend_slope(close: pd.Series, window: int = 63) -> float:
    if len(close) < window:
        return np.nan
    y = np.log(close.iloc[-window:].values.astype(float))
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        return np.nan
    x = np.arange(len(y), dtype=float)
    return float(np.polyfit(x, y, 1)[0]) * 252


def compute_ma_distances(
    close: pd.Series, atr: float, windows: list[int] | None = None,
) -> dict[str, float]:
    if windows is None:
        windows = [50, 100, 200]
    result = {f"ma_dist_{w}": np.nan for w in windows}
    if atr == 0 or np.isnan(_safe_float(atr)):
        return result
    current_price = float(close.iloc[-1])
    for w in windows:
        if len(close) >= w:
            ma = float(close.rolling(w).mean().iloc[-1])
            if not np.isnan(ma) and atr > 0:
                result[f"ma_dist_{w}"] = (current_price - ma) / atr
    return result


def compute_pullback_depth(close: pd.Series, window: int = 20) -> float:
    if len(close) < window:
        return np.nan
    wc = close.iloc[-window:]
    peak = float(wc.max())
    if peak == 0:
        return np.nan
    return float((peak - float(wc.min())) / peak)


def compute_trend_efficiency(close: pd.Series, window: int = 20) -> float:
    if len(close) < window + 1:
        return np.nan
    wc = close.iloc[-window - 1:]
    net = abs(float(wc.iloc[-1]) - float(wc.iloc[0]))
    path = float(wc.diff().abs().sum())
    if path == 0:
        return np.nan
    return float(min(net / path, 1.0))


def compute_base_analysis(
    close, high, low, volume, window=40,
    max_range_pct=0.15, min_duration=15, volatility_contraction_threshold=0.80,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "in_base": False, "base_duration_days": 0, "base_tightness": np.nan,
        "volume_dryup": False, "volatility_contraction": np.nan,
    }
    if len(close) < window:
        return result

    w_high = high.iloc[-window:]
    w_low = low.iloc[-window:]
    w_close = close.iloc[-window:]
    w_vol = volume.iloc[-window:]

    max_h = float(w_high.max())
    min_l = float(w_low.min())
    avg_p = float(w_close.mean())
    if avg_p == 0:
        return result

    range_pct = (max_h - min_l) / avg_p
    result["base_tightness"] = range_pct

    base_band = max_range_pct * avg_p
    band_center = avg_p
    base_days = 0
    for i in range(len(w_close) - 1, -1, -1):
        if (abs(float(w_high.iloc[i]) - band_center) <= base_band
                and abs(float(w_low.iloc[i]) - band_center) <= base_band):
            base_days += 1
        else:
            break

    result["base_duration_days"] = base_days
    result["in_base"] = (range_pct <= max_range_pct) and (base_days >= min_duration)

    half = window // 2
    recent_vol = float(w_vol.iloc[-half:].mean())
    prior_vol = float(w_vol.iloc[:half].mean())
    if prior_vol > 0:
        result["volume_dryup"] = bool(recent_vol < prior_vol * 0.85)

    rets = w_close.pct_change().dropna()
    if len(rets) >= 4:
        recent_std = float(rets.iloc[-(len(rets) // 2):].std())
        prior_std = float(rets.iloc[:(len(rets) // 2)].std())
        if prior_std > 0:
            result["volatility_contraction"] = float(recent_std / prior_std)

    return result


def compute_breakout_signals(
    close, high, low, volume, lookback=20, strength_min=1.5, volume_min=1.5,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "is_breakout": False, "breakout_strength": np.nan,
        "breakout_volume_ratio": np.nan, "retest_holding": False,
    }
    min_bars = lookback + 2
    if len(close) < min_bars or len(high) < min_bars:
        return result

    resistance = float(high.iloc[-(lookback + 1):-1].max())
    current_close = float(close.iloc[-1])
    result["is_breakout"] = bool(current_close > resistance)

    today_range = float(high.iloc[-1]) - float(low.iloc[-1])
    avg_range = float((high - low).iloc[-(lookback + 1):-1].mean())
    if avg_range > 0:
        result["breakout_strength"] = today_range / avg_range

    today_vol = float(volume.iloc[-1])
    avg_vol = float(volume.iloc[-(lookback + 1):-1].mean())
    if avg_vol > 0:
        result["breakout_volume_ratio"] = today_vol / avg_vol

    if len(close) >= 15 + lookback + 1:
        for days_ago in range(5, 16):
            idx = -(days_ago + 1)
            prior_start = -(days_ago + lookback + 1)
            prior_end = -(days_ago + 1)
            if abs(prior_start) > len(high):
                continue
            prior_resistance = float(high.iloc[prior_start:prior_end].max())
            if float(close.iloc[idx]) > prior_resistance and current_close > prior_resistance:
                result["retest_holding"] = True
                break

    return result


def compute_all_price_structure(df: pd.DataFrame, cfg: Any) -> dict[str, Any]:
    ps_cfg = getattr(cfg, "price_structure", None)
    atr_period = int(getattr(ps_cfg, "atr_period", 14)) if ps_cfg else 14

    close = df["Close"].dropna()
    high = df["High"].dropna() if "High" in df.columns else close
    low = df["Low"].dropna() if "Low" in df.columns else close
    volume = df["Volume"].dropna() if "Volume" in df.columns else pd.Series(np.ones(len(close)), index=close.index)

    shared_idx = close.index.intersection(high.index).intersection(low.index).intersection(volume.index)
    close, high, low, volume = close.loc[shared_idx], high.loc[shared_idx], low.loc[shared_idx], volume.loc[shared_idx]

    atr_series = compute_atr(high, low, close, period=atr_period)
    current_atr = _safe_float(atr_series.iloc[-1] if len(atr_series) > 0 else np.nan)

    result: dict[str, Any] = {}
    result["hh_hl"] = compute_higher_highs_higher_lows(high, low)
    result["trend_slope_63d"] = compute_trend_slope(close)
    result.update(compute_ma_distances(close, current_atr))
    result["pullback_depth"] = compute_pullback_depth(close)
    result["trend_efficiency"] = compute_trend_efficiency(close)
    result.update(compute_base_analysis(close, high, low, volume))
    result.update(compute_breakout_signals(close, high, low, volume))
    return result


def fetch_all_price_structure(tickers, price_data, cfg) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for tkr in tickers:
        df = price_data.get(tkr)
        if df is None or df.empty or len(df) < 30:
            results[tkr] = {}
            continue
        try:
            results[tkr] = compute_all_price_structure(df, cfg)
        except Exception:
            results[tkr] = {}
    return results


# =====================================================================
# RELATIVE STRENGTH — RS vs SPY, sector ETFs, percentile ranking
# =====================================================================

_SECTOR_ETF_MAP: dict[str, str] = {
    "Technology": "XLK", "Healthcare": "XLV", "Financial Services": "XLF",
    "Consumer Cyclical": "XLY", "Consumer Defensive": "XLP",
    "Industrials": "XLI", "Energy": "XLE", "Utilities": "XLU",
    "Real Estate": "XLRE", "Basic Materials": "XLB",
    "Communication Services": "XLC",
}
_ALL_SECTOR_ETFS = list(_SECTOR_ETF_MAP.values())


def _align_series(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
    shared = a.index.intersection(b.index)
    return a.loc[shared], b.loc[shared]


def _linear_slope(series: pd.Series) -> float:
    if len(series) < 2:
        return np.nan
    y = series.dropna().values.astype(float)
    if len(y) < 2:
        return np.nan
    mean_y = np.mean(y)
    if mean_y == 0:
        return np.nan
    return float(np.polyfit(np.arange(len(y), dtype=float), y, 1)[0]) / mean_y


def compute_rs_vs_spy(stock_close, spy_close, windows=None) -> dict[str, float]:
    if windows is None:
        windows = [21, 63, 126]
    result = {
        "rs_ratio": np.nan, "rs_vs_spy_21d": np.nan,
        "rs_vs_spy_63d": np.nan, "rs_vs_spy_126d": np.nan,
        "rs_slope_63d": np.nan,
    }
    sc, sp = _align_series(stock_close, spy_close)
    if len(sc) < 10:
        return result
    ratio = (sc / sp.replace(0, np.nan)).dropna()
    if len(ratio) < 2:
        return result
    result["rs_ratio"] = float(ratio.iloc[-1] / ratio.iloc[0]) if ratio.iloc[0] != 0 else np.nan
    for w in windows:
        key = f"rs_vs_spy_{w}d"
        if len(ratio) >= w + 1:
            prior = float(ratio.iloc[-(w + 1)])
            if prior != 0:
                result[key] = (float(ratio.iloc[-1]) - prior) / abs(prior)
    if len(ratio) >= 64:
        result["rs_slope_63d"] = _linear_slope(ratio.iloc[-63:])
    return result


def compute_rs_vs_sector(stock_close, sector_close, windows=None) -> dict[str, float]:
    if windows is None:
        windows = [21, 63]
    result = {
        "rs_vs_sector_21d": np.nan, "rs_vs_sector_63d": np.nan,
        "rs_slope_sector": np.nan,
    }
    sc, se = _align_series(stock_close, sector_close)
    if len(sc) < 10:
        return result
    ratio = (sc / se.replace(0, np.nan)).dropna()
    for w in windows:
        key = f"rs_vs_sector_{w}d"
        if len(ratio) >= w + 1:
            prior = float(ratio.iloc[-(w + 1)])
            if prior != 0:
                result[key] = (float(ratio.iloc[-1]) - prior) / abs(prior)
    if len(ratio) >= 22:
        result["rs_slope_sector"] = _linear_slope(ratio.iloc[-21:])
    return result


def compute_rs_percentile(ticker: str, all_rs_63d: dict[str, float]) -> float:
    this_rs = all_rs_63d.get(ticker)
    if this_rs is None or np.isnan(_safe(this_rs)):
        return np.nan
    vals = [v for v in all_rs_63d.values() if v is not None and not np.isnan(_safe(v))]
    if len(vals) < 2:
        return np.nan
    return float((np.array(vals, dtype=float) < float(this_rs)).mean())


def compute_momentum_persistence(rs_21d, rs_63d, rs_126d) -> float:
    vals = [(v, not np.isnan(_safe(v))) for v in [rs_21d, rs_63d, rs_126d]]
    available = [(v, ok) for v, ok in vals if ok]
    if not available:
        return np.nan
    return float(sum(1 for v, _ in available if v > 0) / len(available))


def compute_sector_ranking(sector_etf_prices, spy_close, window=63) -> dict[str, int]:
    scores: dict[str, float] = {}
    for etf, prices in sector_etf_prices.items():
        se, sp = _align_series(prices, spy_close)
        if len(se) < window + 1:
            scores[etf] = np.nan
            continue
        ratio = (se / sp.replace(0, np.nan)).dropna()
        if len(ratio) >= window + 1:
            prior = float(ratio.iloc[-(window + 1)])
            scores[etf] = (float(ratio.iloc[-1]) - prior) / abs(prior) if prior != 0 else np.nan
        else:
            scores[etf] = np.nan
    ranked = sorted(
        scores.keys(),
        key=lambda e: -scores[e] if not np.isnan(_safe(scores[e])) else float("inf"),
    )
    return {etf: i + 1 for i, etf in enumerate(ranked)}


def compute_all_rs_for_ticker(
    ticker, stock_close, spy_close, sector_etf_close,
    all_rs_63d, sector_rank=None, sector_etf_ticker=None,
) -> dict[str, Any]:
    rs_spy = compute_rs_vs_spy(stock_close, spy_close)
    rs_sector: dict[str, float] = {}
    if sector_etf_close is not None and len(sector_etf_close) > 10:
        rs_sector = compute_rs_vs_sector(stock_close, sector_etf_close)
    rs_pct = compute_rs_percentile(ticker, all_rs_63d)
    rs_persist = compute_momentum_persistence(
        rs_spy.get("rs_vs_spy_21d", np.nan),
        rs_spy.get("rs_vs_spy_63d", np.nan),
        rs_spy.get("rs_vs_spy_126d", np.nan),
    )
    sector_rank_val = np.nan
    if sector_rank and sector_etf_ticker and sector_etf_ticker in sector_rank:
        sector_rank_val = float(sector_rank[sector_etf_ticker])
    return {
        **rs_spy, **rs_sector,
        "rs_percentile": rs_pct, "rs_persistent": rs_persist,
        "sector_rank": sector_rank_val, "sector_etf": sector_etf_ticker or "",
    }


def fetch_all_relative_strength(
    tickers, all_fundamentals, price_data, reference_prices, sector_etf_map=None,
) -> dict[str, dict[str, Any]]:
    etf_map = sector_etf_map or _SECTOR_ETF_MAP
    spy_df = reference_prices.get("SPY")
    if spy_df is None or spy_df.empty:
        return {t: {"rs_vs_spy_63d": np.nan, "rs_percentile": np.nan, "rs_persistent": np.nan} for t in tickers}

    spy_close = spy_df["Close"].dropna()
    sector_closes: dict[str, pd.Series] = {}
    for etf in etf_map.values():
        df = reference_prices.get(etf)
        if df is not None and not df.empty:
            sector_closes[etf] = df["Close"].dropna()

    all_rs_63d: dict[str, float] = {}
    for tkr in tickers:
        df = price_data.get(tkr)
        if df is None or df.empty:
            continue
        rs = compute_rs_vs_spy(df["Close"].dropna(), spy_close)
        all_rs_63d[tkr] = rs.get("rs_vs_spy_63d", np.nan)

    sector_rank = compute_sector_ranking(sector_closes, spy_close)
    results: dict[str, dict[str, Any]] = {}
    for tkr in tickers:
        df = price_data.get(tkr)
        if df is None or df.empty:
            results[tkr] = {"rs_vs_spy_63d": np.nan, "rs_percentile": np.nan, "rs_persistent": np.nan}
            continue
        sector = all_fundamentals.get(tkr, {}).get("sector", "")
        etf_ticker = etf_map.get(sector)
        sector_etf_close = sector_closes.get(etf_ticker) if etf_ticker else None
        results[tkr] = compute_all_rs_for_ticker(
            tkr, df["Close"].dropna(), spy_close, sector_etf_close,
            all_rs_63d, sector_rank, etf_ticker,
        )
    return results


# =====================================================================
# VOLUME PROFILE — CMF, OBV, up/down volume, float turnover
# =====================================================================


def compute_chaikin_money_flow(close, high, low, volume, window=20) -> float:
    if len(close) < window:
        return np.nan
    h = high.iloc[-window:]
    lo = low.iloc[-window:]
    c = close.iloc[-window:]
    v = volume.iloc[-window:]
    hl_range = (h - lo).replace(0, np.nan)
    mfm = ((c - lo) - (h - c)) / hl_range
    sum_vol = float(v.sum())
    if sum_vol == 0 or np.isnan(sum_vol):
        return np.nan
    return float((mfm * v).sum() / sum_vol)


def compute_up_down_volume_ratio(close, volume, window=20) -> float:
    if len(close) < window + 1:
        return np.nan
    w_close = close.iloc[-(window + 1):]
    w_vol = volume.iloc[-(window + 1):]
    delta = w_close.diff().dropna()
    w_vol_trimmed = w_vol.iloc[1:]
    up_vol = float(w_vol_trimmed[delta > 0].sum())
    down_vol = float(w_vol_trimmed[delta < 0].sum())
    if down_vol == 0:
        return np.nan if up_vol == 0 else 3.0
    return float(up_vol / down_vol)


def compute_volume_trend_slope(volume, window=20) -> float:
    if len(volume) < window:
        return np.nan
    y = volume.iloc[-window:].values.astype(float)
    y = y[~np.isnan(y)]
    if len(y) < 2:
        return np.nan
    mean_y = np.mean(y)
    if mean_y == 0:
        return np.nan
    return float(np.polyfit(np.arange(len(y), dtype=float), y, 1)[0]) / mean_y


def compute_volume_spikes(close, volume, window=20, spike_threshold=2.0) -> dict[str, Any]:
    result: dict[str, Any] = {"vol_spike_count": 0, "vol_spike_bullish": False}
    if len(volume) < window + 1:
        return result
    w_vol = volume.iloc[-(window + 1):]
    w_close = close.iloc[-(window + 1):]
    avg_vol = float(w_vol.iloc[:-1].mean())
    if avg_vol == 0:
        return result
    delta = w_close.diff().dropna()
    w_vol_recent = w_vol.iloc[1:]
    is_spike = w_vol_recent > spike_threshold * avg_vol
    spike_count = int(is_spike.sum())
    result["vol_spike_count"] = spike_count
    if spike_count > 0:
        result["vol_spike_bullish"] = bool(
            int((is_spike & (delta > 0)).sum()) > int((is_spike & (delta < 0)).sum())
        )
    return result


def compute_float_turnover(volume, float_shares, window=20) -> float:
    if float_shares is None:
        return np.nan
    fv = _safe_float(float_shares)
    if np.isnan(fv) or fv == 0 or len(volume) < window:
        return np.nan
    return float(float(volume.iloc[-window:].sum()) / fv)


def compute_obv_trend(close, volume, window=20) -> float:
    if len(close) < window + 1:
        return np.nan
    obv = (np.sign(close.diff()).fillna(0) * volume).cumsum()
    y = obv.iloc[-window:].values.astype(float)
    if len(y) < 2:
        return np.nan
    mean_y = abs(np.mean(y))
    x = np.arange(len(y), dtype=float)
    slope = float(np.polyfit(x, y, 1)[0])
    return slope / mean_y if mean_y != 0 else slope


def compute_all_volume_profile(df, float_shares, cfg) -> dict[str, Any]:
    close = df["Close"].dropna()
    high = df["High"].dropna() if "High" in df.columns else close
    low = df["Low"].dropna() if "Low" in df.columns else close
    volume = df["Volume"].dropna() if "Volume" in df.columns else pd.Series(np.ones(len(close)), index=close.index)

    shared_idx = close.index.intersection(high.index).intersection(low.index).intersection(volume.index)
    close, high, low, volume = close.loc[shared_idx], high.loc[shared_idx], low.loc[shared_idx], volume.loc[shared_idx]

    result: dict[str, Any] = {}
    result["cmf"] = compute_chaikin_money_flow(close, high, low, volume)
    result["ud_volume_ratio"] = compute_up_down_volume_ratio(close, volume)
    result["volume_trend_slope"] = compute_volume_trend_slope(volume)
    result.update(compute_volume_spikes(close, volume))
    result["float_turnover"] = compute_float_turnover(volume, float_shares)
    result["obv_slope"] = compute_obv_trend(close, volume)
    return result


def fetch_all_volume_profile(tickers, price_data, all_fundamentals, cfg) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for tkr in tickers:
        df = price_data.get(tkr)
        if df is None or df.empty or len(df) < 25:
            results[tkr] = {}
            continue
        float_shares = all_fundamentals.get(tkr, {}).get("floatShares")
        try:
            results[tkr] = compute_all_volume_profile(df, float_shares, cfg)
        except Exception:
            results[tkr] = {}
    return results


# =====================================================================
# RISK CHECKS — earnings, short interest, insider, analyst, dilution
# =====================================================================


def check_earnings_proximity(ticker_obj: yf.Ticker) -> dict[str, Any]:
    try:
        cal = ticker_obj.calendar
        if cal is None or (isinstance(cal, pd.DataFrame) and cal.empty):
            return {"next_earnings": None, "days_until_earnings": None, "risk_level": "UNKNOWN", "flag": None}
        if isinstance(cal, pd.DataFrame):
            earnings_col = [c for c in cal.columns if "Earnings" in str(c)]
            if not earnings_col:
                return {"next_earnings": None, "days_until_earnings": None, "risk_level": "UNKNOWN", "flag": None}
            raw_date = cal[earnings_col[0]].iloc[0]
        elif isinstance(cal, dict):
            raw_date = cal.get("Earnings Date") or cal.get("earnings_date")
            if isinstance(raw_date, (list, pd.DatetimeIndex)):
                raw_date = raw_date[0] if len(raw_date) > 0 else None
        else:
            raw_date = None
        if raw_date is None or (isinstance(raw_date, float) and np.isnan(raw_date)):
            return {"next_earnings": None, "days_until_earnings": None, "risk_level": "UNKNOWN", "flag": None}
        next_date = raw_date.date() if hasattr(raw_date, "date") else pd.Timestamp(raw_date).date()
        days_away = (next_date - date.today()).days
        if days_away < 0:
            return {"next_earnings": next_date, "days_until_earnings": days_away, "risk_level": "LOW", "flag": None}
        if days_away <= 7:
            return {"next_earnings": next_date, "days_until_earnings": days_away, "risk_level": "HIGH", "flag": f"EARNINGS IN {days_away} DAYS ({next_date})"}
        elif days_away <= 21:
            return {"next_earnings": next_date, "days_until_earnings": days_away, "risk_level": "MEDIUM", "flag": f"Earnings in {days_away} days ({next_date})"}
        return {"next_earnings": next_date, "days_until_earnings": days_away, "risk_level": "LOW", "flag": None}
    except Exception:
        return {"next_earnings": None, "days_until_earnings": None, "risk_level": "UNKNOWN", "flag": None}


def check_short_interest(info: dict[str, Any]) -> dict[str, Any]:
    short_ratio = info.get("shortRatio")
    short_pct = info.get("shortPercentOfFloat")
    try:
        short_ratio = float(short_ratio) if short_ratio is not None else None
        short_pct = float(short_pct) if short_pct is not None else None
    except (TypeError, ValueError):
        short_ratio = short_pct = None
    if short_ratio is None and short_pct is None:
        return {"short_ratio": None, "short_pct_float": None, "squeeze_potential": False, "risk_level": "UNKNOWN", "flag": None}
    ratio_high = short_ratio is not None and short_ratio > 10
    pct_high = short_pct is not None and short_pct > 0.20
    if ratio_high or pct_high:
        risk_level = "HIGH"
        flag = f"HIGH SHORT INTEREST: {short_pct:.0%} of float" if short_pct else f"HIGH SHORT: days-to-cover {short_ratio:.1f}"
    elif (short_pct is not None and 0.10 < short_pct <= 0.20) or (short_ratio is not None and short_ratio > 5):
        risk_level = "MEDIUM"
        flag = f"Elevated short interest: {short_pct:.0%}" if short_pct else f"Elevated short ratio: {short_ratio:.1f}"
    else:
        risk_level, flag = "LOW", None
    return {"short_ratio": short_ratio, "short_pct_float": short_pct, "squeeze_potential": pct_high, "risk_level": risk_level, "flag": flag}


def check_insider_activity(ticker_obj: yf.Ticker) -> dict[str, Any]:
    _unknown = {"net_shares_90d": 0.0, "n_buys": 0, "n_sells": 0, "total_buy_value": 0.0, "total_sell_value": 0.0, "signal": "UNKNOWN", "flag": None}
    try:
        txns = ticker_obj.insider_transactions
        if txns is None or (hasattr(txns, "empty") and txns.empty):
            return _unknown
    except Exception:
        return _unknown
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=90)
    try:
        date_col = "Start Date" if "Start Date" in txns.columns else ("Date" if "Date" in txns.columns else txns.columns[0])
        txns[date_col] = pd.to_datetime(txns[date_col], utc=True, errors="coerce")
        recent = txns[txns[date_col] >= cutoff]
    except Exception:
        recent = txns
    if recent.empty:
        return {**_unknown, "signal": "NEUTRAL", "flag": "No insider transactions in 90 days"}
    text_col = next((c for c in recent.columns if "transaction" in c.lower() or "type" in c.lower()), None)
    value_col = next((c for c in recent.columns if "value" in c.lower()), None)
    shares_col = next((c for c in recent.columns if "share" in c.lower()), None)
    buys = recent[recent[text_col].str.lower().str.contains("buy|purchase", na=False)] if text_col else pd.DataFrame()
    sells = recent[recent[text_col].str.lower().str.contains("sell|sale", na=False)] if text_col else pd.DataFrame()

    def _sum_col(df, col):
        if col is None or col not in df.columns:
            return 0.0
        try:
            return float(pd.to_numeric(df[col], errors="coerce").abs().sum())
        except Exception:
            return 0.0

    total_buy_value = _sum_col(buys, value_col)
    total_sell_value = _sum_col(sells, value_col)
    net_shares = _sum_col(buys, shares_col) - _sum_col(sells, shares_col)
    n_buys, n_sells = len(buys), len(sells)
    if n_buys >= 2 and total_buy_value > total_sell_value * 1.5:
        signal, flag = "BULLISH", f"INSIDER BUYING: {n_buys} purchases (~${total_buy_value:,.0f})"
    elif n_sells >= 3 and total_sell_value > total_buy_value * 2:
        signal, flag = "BEARISH", f"CLUSTER INSIDER SELLING: {n_sells} sales (~${total_sell_value:,.0f})"
    else:
        signal, flag = "NEUTRAL", None
    return {"net_shares_90d": net_shares, "n_buys": n_buys, "n_sells": n_sells, "total_buy_value": total_buy_value, "total_sell_value": total_sell_value, "signal": signal, "flag": flag}


def check_analyst_consensus(ticker_obj: yf.Ticker, info: dict[str, Any]) -> dict[str, Any]:
    current_price = info.get("currentPrice") or info.get("regularMarketPrice")
    target_mean = info.get("targetMeanPrice")
    target_high = info.get("targetHighPrice")
    target_low = info.get("targetLowPrice")
    n_analysts = info.get("numberOfAnalystOpinions", 0)
    recommendation = str(info.get("recommendationKey", "")).upper()
    try:
        current_price = float(current_price) if current_price else None
        target_mean = float(target_mean) if target_mean else None
        n_analysts = int(n_analysts) if n_analysts else 0
    except (TypeError, ValueError):
        current_price = target_mean = None
        n_analysts = 0
    upside = (target_mean - current_price) / current_price if current_price and target_mean and current_price > 0 else None
    _rec_map = {"STRONG_BUY": "STRONG_BUY", "BUY": "BUY", "HOLD": "HOLD", "UNDERPERFORM": "SELL", "SELL": "SELL", "": "UNKNOWN"}
    consensus = _rec_map.get(recommendation, "UNKNOWN")
    recent_upgrades = recent_downgrades = 0
    try:
        recs = ticker_obj.recommendations
        if recs is not None and not recs.empty:
            cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=30)
            if recs.index.tz is None:
                recs.index = recs.index.tz_localize("UTC")
            recent_recs = recs[recs.index >= cutoff]
            if "To Grade" in recent_recs.columns and "From Grade" in recent_recs.columns:
                _buy_grades = {"Strong Buy", "Buy", "Outperform", "Overweight", "Positive"}
                _sell_grades = {"Sell", "Underperform", "Underweight", "Negative"}
                for _, row in recent_recs.iterrows():
                    to_g = str(row.get("To Grade", ""))
                    from_g = str(row.get("From Grade", ""))
                    if to_g in _buy_grades and from_g not in _buy_grades:
                        recent_upgrades += 1
                    elif to_g in _sell_grades and from_g not in _sell_grades:
                        recent_downgrades += 1
    except Exception:
        pass
    flag = None
    if recent_downgrades >= 2:
        flag = f"ANALYST DOWNGRADES: {recent_downgrades} downgrades in 30 days"
    elif upside is not None and upside < 0:
        flag = f"ABOVE ANALYST TARGET: price exceeds mean target ({upside:.1%})"
    elif recent_upgrades >= 2:
        flag = f"ANALYST UPGRADES: {recent_upgrades} upgrades in 30 days"
    return {"current_price": current_price, "target_mean": target_mean, "target_high": target_high, "target_low": target_low, "n_analysts": n_analysts, "upside_to_mean": upside, "recent_upgrades": recent_upgrades, "recent_downgrades": recent_downgrades, "consensus": consensus, "flag": flag}


def check_dilution_risk(info: dict[str, Any]) -> dict[str, Any]:
    shares_out = info.get("sharesOutstanding")
    float_shares = info.get("floatShares")
    insider_pct = info.get("heldPercentInsiders")
    inst_pct = info.get("heldPercentInstitutions")
    try:
        inst_pct = float(inst_pct) if inst_pct else None
        insider_pct = float(insider_pct) if insider_pct else None
    except (TypeError, ValueError):
        inst_pct = insider_pct = None
    flag = None
    if inst_pct is not None and inst_pct > 0.90:
        flag = f"HIGH INSTITUTIONAL CROWDING: {inst_pct:.0%} institutional ownership"
    elif insider_pct is not None and insider_pct < 0.01:
        flag = "LOW INSIDER OWNERSHIP: insiders own <1%"
    return {"shares_outstanding": shares_out, "float_shares": float_shares, "insider_pct": insider_pct, "institution_pct": inst_pct, "flag": flag}


def run_all_risk_checks(ticker_obj: yf.Ticker, info: dict[str, Any]) -> dict[str, Any]:
    earnings = check_earnings_proximity(ticker_obj)
    short = check_short_interest(info)
    insider = check_insider_activity(ticker_obj)
    analyst = check_analyst_consensus(ticker_obj, info)
    dilution = check_dilution_risk(info)
    all_flags = [check.get("flag") for check in [earnings, short, insider, analyst, dilution] if check.get("flag")]
    high_triggers = sum([
        earnings.get("risk_level") == "HIGH", short.get("risk_level") == "HIGH",
        insider.get("signal") == "BEARISH", analyst.get("consensus") == "SELL",
        analyst.get("recent_downgrades", 0) >= 2,
    ])
    medium_triggers = sum([
        earnings.get("risk_level") == "MEDIUM", short.get("risk_level") == "MEDIUM",
        analyst.get("upside_to_mean") is not None and analyst["upside_to_mean"] < 0,
    ])
    if high_triggers >= 2:
        overall_risk, risk_score = "HIGH", -0.30 - (high_triggers - 2) * 0.10
    elif high_triggers == 1 or medium_triggers >= 2:
        overall_risk, risk_score = "MEDIUM", -0.15
    else:
        overall_risk, risk_score = "LOW", 0.0
    if insider.get("signal") == "BULLISH":
        risk_score = min(0.0, risk_score + 0.10)
    return {"earnings": earnings, "short_interest": short, "insider": insider, "analyst": analyst, "dilution": dilution, "risk_flags": all_flags, "overall_risk": overall_risk, "risk_score": max(-1.0, min(0.0, risk_score))}


def fetch_all_risk_checks(tickers, all_fundamentals) -> dict[str, dict[str, Any]]:
    """Parallel risk checks."""
    results: dict[str, dict[str, Any]] = {}
    _empty = {"earnings": {}, "short_interest": {}, "insider": {}, "analyst": {},
              "dilution": {}, "risk_flags": [], "overall_risk": "UNKNOWN", "risk_score": 0.0}
    def _one(tkr):
        try:
            return tkr, run_all_risk_checks(yf.Ticker(tkr), all_fundamentals.get(tkr, {}))
        except Exception:
            return tkr, _empty
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
        for fut in as_completed([ex.submit(_one, t) for t in tickers]):
            tkr, data = fut.result()
            results[tkr] = data
    return results


# =====================================================================
# FINANCIAL HEALTH — Piotroski F-Score + Altman Z-Score
# =====================================================================


def _get_row(df: pd.DataFrame, *keys: str) -> float:
    for key in keys:
        if key in df.index:
            val = df.loc[key]
            if isinstance(val, pd.Series):
                val = val.iloc[0]
            f = _safe_float(val)
            if not np.isnan(f):
                return f
    return np.nan


def _latest_two(df: pd.DataFrame, *keys: str) -> tuple[float, float]:
    for key in keys:
        if key in df.index:
            row = df.loc[key]
            if isinstance(row, pd.Series) and len(row) >= 2:
                return _safe_float(row.iloc[0]), _safe_float(row.iloc[1])
            elif isinstance(row, pd.Series) and len(row) == 1:
                return _safe_float(row.iloc[0]), np.nan
    return np.nan, np.nan


def compute_piotroski_score(ticker_obj: yf.Ticker) -> dict[str, Any]:
    components = {k: None for k in [
        "roa_positive", "ocf_positive", "roa_improving", "accruals_low",
        "leverage_decreasing", "liquidity_improving", "no_dilution",
        "gross_margin_improving", "asset_turnover_improving",
    ]}
    try:
        bs = ticker_obj.balance_sheet
        inc = ticker_obj.income_stmt
        cf = ticker_obj.cashflow
        if bs is None or inc is None or cf is None or bs.empty or inc.empty or cf.empty:
            raise ValueError("Missing statements")

        total_assets_curr, total_assets_prev = _latest_two(bs, "Total Assets", "TotalAssets")
        long_debt_curr, long_debt_prev = _latest_two(bs, "Long Term Debt", "LongTermDebt")
        current_assets_curr, current_assets_prev = _latest_two(bs, "Current Assets", "CurrentAssets", "Total Current Assets")
        current_liab_curr, current_liab_prev = _latest_two(bs, "Current Liabilities", "CurrentLiabilities", "Total Current Liabilities")
        shares_curr, shares_prev = _latest_two(bs, "Common Stock", "Ordinary Shares Number", "Share Issued")
        net_income_curr, _ = _latest_two(inc, "Net Income", "NetIncome")
        revenue_curr, revenue_prev = _latest_two(inc, "Total Revenue", "TotalRevenue", "Revenue")
        gross_profit_curr, gross_profit_prev = _latest_two(inc, "Gross Profit", "GrossProfit")
        ocf_curr, _ = _latest_two(cf, "Operating Cash Flow", "OperatingCashFlow", "Cash From Operations", "Total Cash From Operating Activities")

        roa = np.nan
        if not np.isnan(net_income_curr) and not np.isnan(total_assets_curr) and total_assets_curr != 0:
            roa = net_income_curr / total_assets_curr
            components["roa_positive"] = 1 if roa > 0 else 0
            ni_prev = _safe_float(inc.loc[next((k for k in ["Net Income", "NetIncome"] if k in inc.index), inc.index[0])].iloc[1] if len(inc.columns) > 1 else np.nan)
            if not np.isnan(total_assets_prev) and total_assets_prev != 0 and not np.isnan(ni_prev):
                roa_prev = ni_prev / total_assets_prev
                components["roa_improving"] = 1 if roa > roa_prev else 0

        if not np.isnan(ocf_curr):
            components["ocf_positive"] = 1 if ocf_curr > 0 else 0
        if not np.isnan(ocf_curr) and not np.isnan(total_assets_curr) and total_assets_curr != 0 and not np.isnan(roa):
            components["accruals_low"] = 1 if ocf_curr / total_assets_curr > roa else 0
        if not np.isnan(long_debt_curr) and not np.isnan(long_debt_prev):
            components["leverage_decreasing"] = 1 if long_debt_curr < long_debt_prev else 0
        if not (np.isnan(current_assets_curr) or np.isnan(current_liab_curr) or current_liab_curr == 0):
            cr_curr = current_assets_curr / current_liab_curr
            if not (np.isnan(current_assets_prev) or np.isnan(current_liab_prev) or current_liab_prev == 0):
                components["liquidity_improving"] = 1 if cr_curr > current_assets_prev / current_liab_prev else 0
        if not np.isnan(shares_curr) and not np.isnan(shares_prev) and shares_prev != 0:
            components["no_dilution"] = 1 if shares_curr <= shares_prev * 1.02 else 0
        if not (np.isnan(gross_profit_curr) or np.isnan(revenue_curr) or revenue_curr == 0):
            gm_curr = gross_profit_curr / revenue_curr
            if not (np.isnan(gross_profit_prev) or np.isnan(revenue_prev) or revenue_prev == 0):
                components["gross_margin_improving"] = 1 if gm_curr > gross_profit_prev / revenue_prev else 0
        if not (np.isnan(revenue_curr) or np.isnan(total_assets_curr) or total_assets_curr == 0):
            at_curr = revenue_curr / total_assets_curr
            if not (np.isnan(revenue_prev) or np.isnan(total_assets_prev) or total_assets_prev == 0):
                components["asset_turnover_improving"] = 1 if at_curr > revenue_prev / total_assets_prev else 0
    except Exception:
        pass

    available = [v for v in components.values() if v is not None]
    if not available:
        return {"score": None, "score_pct": None, "available_criteria": 0, "grade": "INSUFFICIENT_DATA", "components": components, "flag": None}
    raw_score = sum(available)
    n_avail = len(available)
    score_9 = round(raw_score * 9 / n_avail)
    if score_9 >= 7:
        grade, flag = "STRONG", None
    elif score_9 >= 4:
        grade, flag = "NEUTRAL", None
    else:
        grade, flag = "WEAK", f"PIOTROSKI F-SCORE WEAK ({score_9}/9)"
    return {"score": score_9, "score_pct": raw_score / n_avail, "available_criteria": n_avail, "grade": grade, "components": components, "flag": flag}


def compute_altman_z_score(ticker_obj: yf.Ticker, info: dict[str, Any]) -> dict[str, Any]:
    sector = info.get("sector", "")
    financial_sector = sector in {"Financial Services", "Financial", "Banks", "Insurance"}
    components = {"X1": np.nan, "X2": np.nan, "X3": np.nan, "X4": np.nan, "X5": np.nan}
    try:
        bs = ticker_obj.balance_sheet
        inc = ticker_obj.income_stmt
        if bs is None or inc is None or bs.empty or inc.empty:
            raise ValueError("Missing")
        total_assets = _get_row(bs, "Total Assets", "TotalAssets")
        current_assets = _get_row(bs, "Current Assets", "CurrentAssets", "Total Current Assets")
        current_liab = _get_row(bs, "Current Liabilities", "CurrentLiabilities", "Total Current Liabilities")
        retained_earnings = _get_row(bs, "Retained Earnings", "RetainedEarnings")
        total_liab = _get_row(bs, "Total Liabilities Net Minority Interest", "Total Liabilities", "TotalLiabilities")
        revenue = _get_row(inc, "Total Revenue", "TotalRevenue", "Revenue")
        ebit = _get_row(inc, "EBIT", "Operating Income", "OperatingIncome")
        mkt_cap = _safe_float(info.get("marketCap"))
        if not np.isnan(total_assets) and total_assets > 0:
            wc = current_assets - current_liab
            if not np.isnan(wc):
                components["X1"] = wc / total_assets
            if not np.isnan(retained_earnings):
                components["X2"] = retained_earnings / total_assets
            if not np.isnan(ebit):
                components["X3"] = ebit / total_assets
            if not np.isnan(total_liab) and total_liab > 0 and not np.isnan(mkt_cap):
                components["X4"] = mkt_cap / total_liab
            if not np.isnan(revenue):
                components["X5"] = revenue / total_assets
    except Exception:
        pass
    available_x = {k: v for k, v in components.items() if not np.isnan(v)}
    if len(available_x) < 3:
        return {"z_score": None, "zone": "UNKNOWN", "components": components, "reliable": False, "flag": None}
    weights = {"X1": 1.2, "X2": 1.4, "X3": 3.3, "X4": 0.6, "X5": 1.0}
    z = sum(weights[k] * v for k, v in available_x.items())
    if z > 2.99:
        zone, flag = "SAFE", None
    elif z > 1.81:
        zone, flag = "GREY", f"ALTMAN Z-SCORE GREY ZONE ({z:.2f})"
    else:
        zone, flag = "DISTRESS", f"ALTMAN Z-SCORE DISTRESS ({z:.2f})"
    if financial_sector and flag:
        flag += " [Less reliable for financials]"
    return {"z_score": round(z, 3), "zone": zone, "components": components, "reliable": not financial_sector, "flag": flag}


def compute_balance_sheet_quality(ticker_obj: yf.Ticker, info: dict[str, Any]) -> dict[str, Any]:
    """Deep balance-sheet, cash-flow, and profitability quality check.

    Distinguishes *good* debt (low interest, well-covered) from *bad* debt
    (high leverage, poor coverage, burning cash).  Returns a grade from
    EXCELLENT to DANGEROUS and a list of human-readable flags.
    """
    flags: list[str] = []
    score_parts: list[float] = []

    # --- Debt-to-Equity ratio -------------------------------------------
    de = _safe(info.get("debtToEquity"))  # yfinance gives this as a %
    if not np.isnan(de):
        de_ratio = de / 100.0  # convert to decimal
        if de_ratio < 0.30:
            score_parts.append(0.8)
            flags.append(f"LOW DEBT: D/E {de_ratio:.2f} — very conservative balance sheet")
        elif de_ratio < 0.80:
            score_parts.append(0.4)
        elif de_ratio < 1.50:
            score_parts.append(0.0)
            flags.append(f"MODERATE DEBT: D/E {de_ratio:.2f}")
        elif de_ratio < 3.00:
            score_parts.append(-0.5)
            flags.append(f"HIGH DEBT: D/E {de_ratio:.2f} — leverage is elevated")
        else:
            score_parts.append(-1.0)
            flags.append(f"DANGEROUS DEBT: D/E {de_ratio:.2f} — extremely leveraged")

    # --- Interest coverage (EBIT / interest expense) --------------------
    try:
        inc = ticker_obj.income_stmt
        if inc is not None and not inc.empty:
            ebit = np.nan
            interest = np.nan
            for key in ["EBIT", "Operating Income", "OperatingIncome"]:
                if key in inc.index:
                    ebit = _safe_float(inc.loc[key].iloc[0])
                    break
            for key in ["Interest Expense", "InterestExpense", "Interest Expense Non Operating"]:
                if key in inc.index:
                    interest = abs(_safe_float(inc.loc[key].iloc[0]))
                    break
            if not np.isnan(ebit) and not np.isnan(interest) and interest > 0:
                coverage = ebit / interest
                if coverage >= 8.0:
                    score_parts.append(0.7)
                    flags.append(f"STRONG INTEREST COVERAGE: {coverage:.1f}x — debt easily serviceable")
                elif coverage >= 4.0:
                    score_parts.append(0.3)
                elif coverage >= 2.0:
                    score_parts.append(-0.2)
                    flags.append(f"THIN INTEREST COVERAGE: {coverage:.1f}x — limited margin of safety")
                elif coverage >= 1.0:
                    score_parts.append(-0.6)
                    flags.append(f"WEAK INTEREST COVERAGE: {coverage:.1f}x — barely covering debt payments")
                else:
                    score_parts.append(-1.0)
                    flags.append(f"CANNOT COVER INTEREST: {coverage:.1f}x — earnings don't cover debt costs")
    except Exception:
        pass

    # --- Current ratio (short-term liquidity) ---------------------------
    cr = _safe(info.get("currentRatio"))
    if not np.isnan(cr):
        if cr >= 2.0:
            score_parts.append(0.6)
        elif cr >= 1.5:
            score_parts.append(0.3)
        elif cr >= 1.0:
            score_parts.append(0.0)
        elif cr >= 0.7:
            score_parts.append(-0.4)
            flags.append(f"LOW CURRENT RATIO: {cr:.2f} — may struggle to pay short-term bills")
        else:
            score_parts.append(-0.8)
            flags.append(f"CRITICAL CURRENT RATIO: {cr:.2f} — serious short-term liquidity risk")

    # --- Free cash flow (is the business generating real cash?) ---------
    fcf = _safe(info.get("freeCashFlow"))
    mkt_cap = _safe(info.get("marketCap"))
    revenue = _safe(info.get("totalRevenue"))
    if not np.isnan(fcf):
        if fcf > 0:
            score_parts.append(0.5)
            if not np.isnan(mkt_cap) and mkt_cap > 0:
                fcf_yield = fcf / mkt_cap
                if fcf_yield >= 0.06:
                    flags.append(f"STRONG FREE CASH FLOW: {fcf_yield:.1%} FCF yield")
                    score_parts.append(0.3)
        else:
            score_parts.append(-0.7)
            flags.append("NEGATIVE FREE CASH FLOW — company is burning cash")

    # --- Operating cash flow vs net income (earnings quality) -----------
    opcf = _safe(info.get("operatingCashflow"))
    net_income = _safe(info.get("netIncomeToCommon"))
    if not np.isnan(opcf) and not np.isnan(net_income) and net_income > 0:
        ocf_ratio = opcf / net_income
        if ocf_ratio >= 1.0:
            score_parts.append(0.4)
        elif ocf_ratio >= 0.5:
            score_parts.append(0.0)
        else:
            score_parts.append(-0.5)
            flags.append(f"POOR EARNINGS QUALITY: operating cash flow only {ocf_ratio:.0%} of net income")

    # --- Profitability check --------------------------------------------
    profit_margin = _safe(info.get("profitMargins"))
    operating_margin = _safe(info.get("operatingMargins"))
    roe = _safe(info.get("returnOnEquity"))
    profitable = True

    if not np.isnan(net_income):
        if net_income <= 0:
            score_parts.append(-0.8)
            flags.append("NOT PROFITABLE — company is losing money")
            profitable = False
        else:
            score_parts.append(0.5)

    if not np.isnan(profit_margin):
        if profit_margin >= 0.20:
            score_parts.append(0.6)
            flags.append(f"HIGH PROFIT MARGIN: {profit_margin:.1%}")
        elif profit_margin >= 0.10:
            score_parts.append(0.3)
        elif profit_margin >= 0.0:
            score_parts.append(0.0)
        else:
            score_parts.append(-0.6)

    if not np.isnan(operating_margin):
        if operating_margin >= 0.25:
            score_parts.append(0.5)
        elif operating_margin >= 0.10:
            score_parts.append(0.2)
        elif operating_margin < 0:
            score_parts.append(-0.5)
            flags.append(f"NEGATIVE OPERATING MARGIN: {operating_margin:.1%} — core business unprofitable")

    if not np.isnan(roe):
        if roe >= 0.20:
            score_parts.append(0.5)
        elif roe >= 0.10:
            score_parts.append(0.2)
        elif roe < 0:
            score_parts.append(-0.6)
            flags.append("NEGATIVE ROE — destroying shareholder value")

    # --- P/E sanity check -----------------------------------------------
    pe_fwd = _safe(info.get("forwardPE"))
    pe_trail = _safe(info.get("trailingPE"))
    pe = pe_fwd if not np.isnan(pe_fwd) else pe_trail
    if not np.isnan(pe):
        if pe < 0:
            score_parts.append(-0.8)
            flags.append("NEGATIVE P/E — company has negative earnings")
        elif pe > 100:
            score_parts.append(-0.4)
            flags.append(f"EXTREME P/E: {pe:.0f} — priced for perfection")
        elif pe > 50:
            score_parts.append(-0.1)
            flags.append(f"HIGH P/E: {pe:.0f}")
        elif pe > 25:
            score_parts.append(0.1)
        elif pe > 10:
            score_parts.append(0.4)
            flags.append(f"REASONABLE P/E: {pe:.0f}")
        else:
            score_parts.append(0.3)  # Very low P/E could be a value trap

    # --- Debt vs cash (net debt position) --------------------------------
    try:
        bs = ticker_obj.balance_sheet
        if bs is not None and not bs.empty:
            total_debt = np.nan
            cash = np.nan
            for key in ["Total Debt", "TotalDebt", "Long Term Debt", "LongTermDebt"]:
                if key in bs.index:
                    total_debt = abs(_safe_float(bs.loc[key].iloc[0]))
                    break
            for key in ["Cash And Cash Equivalents", "CashAndCashEquivalents", "Cash Cash Equivalents And Short Term Investments"]:
                if key in bs.index:
                    cash = _safe_float(bs.loc[key].iloc[0])
                    break
            if not np.isnan(total_debt) and not np.isnan(cash):
                net_debt = total_debt - cash
                if net_debt <= 0:
                    score_parts.append(0.7)
                    flags.append(f"NET CASH POSITION: more cash (${cash/1e9:.1f}B) than debt (${total_debt/1e9:.1f}B)")
                elif not np.isnan(revenue) and revenue > 0:
                    net_debt_to_rev = net_debt / revenue
                    if net_debt_to_rev < 0.5:
                        score_parts.append(0.3)
                    elif net_debt_to_rev < 1.5:
                        score_parts.append(0.0)
                    elif net_debt_to_rev < 3.0:
                        score_parts.append(-0.4)
                        flags.append(f"HIGH NET DEBT: {net_debt_to_rev:.1f}x revenue")
                    else:
                        score_parts.append(-0.8)
                        flags.append(f"EXTREME NET DEBT: {net_debt_to_rev:.1f}x revenue — dangerous leverage")
    except Exception:
        pass

    # --- Final grade -----------------------------------------------------
    if not score_parts:
        return {
            "bs_score": 0.0, "bs_grade": "UNKNOWN", "profitable": True,
            "flags": flags, "blocks_high": False,
        }

    bs_score = max(-1.0, min(1.0, float(np.mean(score_parts))))
    if bs_score >= 0.45:
        bs_grade = "EXCELLENT"
    elif bs_score >= 0.20:
        bs_grade = "GOOD"
    elif bs_score >= -0.05:
        bs_grade = "FAIR"
    elif bs_score >= -0.40:
        bs_grade = "POOR"
    else:
        bs_grade = "DANGEROUS"

    # Block HIGH confidence if financials are truly bad
    blocks_high = bs_grade == "DANGEROUS" or not profitable

    return {
        "bs_score": bs_score, "bs_grade": bs_grade, "profitable": profitable,
        "flags": flags, "blocks_high": blocks_high,
    }


def compute_health_scores(ticker_obj: yf.Ticker, info: dict[str, Any]) -> dict[str, Any]:
    piotroski = compute_piotroski_score(ticker_obj)
    altman = compute_altman_z_score(ticker_obj, info)
    balance_sheet = compute_balance_sheet_quality(ticker_obj, info)
    score_parts: list[float] = []

    p_grade = piotroski.get("grade", "INSUFFICIENT_DATA")
    if p_grade == "STRONG":
        score_parts.append(0.8)
    elif p_grade == "NEUTRAL":
        score_parts.append(0.2)
    elif p_grade == "WEAK":
        score_parts.append(-0.8)

    a_zone = altman.get("zone", "UNKNOWN")
    if a_zone == "SAFE":
        score_parts.append(0.6)
    elif a_zone == "GREY":
        score_parts.append(0.0)
    elif a_zone == "DISTRESS":
        score_parts.append(-1.0)

    # Add balance sheet quality (weighted more heavily)
    bs_score = float(balance_sheet.get("bs_score", 0.0))
    score_parts.append(bs_score)
    score_parts.append(bs_score)  # double-weight balance sheet quality

    if not score_parts:
        all_flags = [f for f in [piotroski.get("flag"), altman.get("flag")] if f] + balance_sheet.get("flags", [])
        return {
            "piotroski": piotroski, "altman": altman, "balance_sheet": balance_sheet,
            "health_score": 0.0, "health_grade": "UNKNOWN", "flags": all_flags,
        }

    health_score = max(-1.0, min(1.0, float(np.mean(score_parts))))
    if health_score >= 0.5:
        health_grade = "EXCELLENT"
    elif health_score >= 0.1:
        health_grade = "GOOD"
    elif health_score >= -0.3:
        health_grade = "FAIR"
    else:
        health_grade = "POOR"

    all_flags = [f for f in [piotroski.get("flag"), altman.get("flag")] if f] + balance_sheet.get("flags", [])
    return {
        "piotroski": piotroski, "altman": altman, "balance_sheet": balance_sheet,
        "health_score": health_score, "health_grade": health_grade, "flags": all_flags,
    }


def fetch_all_health_scores(tickers, all_fundamentals) -> dict[str, dict[str, Any]]:
    """Parallel health scores (Piotroski + Altman + balance sheet quality)."""
    results: dict[str, dict[str, Any]] = {}
    _empty = {
        "piotroski": {"grade": "UNKNOWN", "score": None, "flag": None},
        "altman": {"zone": "UNKNOWN", "z_score": None, "flag": None},
        "balance_sheet": {"bs_score": 0.0, "bs_grade": "UNKNOWN",
                          "profitable": True, "flags": [], "blocks_high": False},
        "health_score": 0.0, "health_grade": "UNKNOWN", "flags": [],
    }
    def _one(tkr):
        try:
            return tkr, compute_health_scores(yf.Ticker(tkr), all_fundamentals.get(tkr, {}))
        except Exception:
            return tkr, _empty
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
        for fut in as_completed([ex.submit(_one, t) for t in tickers]):
            tkr, data = fut.result()
            results[tkr] = data
    return results


# =====================================================================
# SCORER — cross-sectional z-scoring + composite score
# =====================================================================


def cross_sectional_zscore(signal_matrix: pd.DataFrame, winsorize_pct: float = 0.05) -> pd.DataFrame:
    result = signal_matrix.copy().astype(float)
    for col in result.columns:
        series = result[col].dropna()
        if series.empty:
            result[col] = 0.0
            continue
        if winsorize_pct > 0:
            lo = series.quantile(winsorize_pct)
            hi = series.quantile(1.0 - winsorize_pct)
            result[col] = result[col].clip(lower=lo, upper=hi)
        std = result[col].std(skipna=True)
        mean = result[col].mean(skipna=True)
        if std == 0 or pd.isna(std):
            result[col] = 0.0
        else:
            result[col] = (result[col] - mean) / std
        result[col] = result[col].clip(-3.0, 3.0).fillna(0.0)
    return result


def compute_fundamental_signals(fund: dict[str, Any]) -> dict[str, float]:
    """Value-tilted fundamental signals.

    Reward LOW P/E, LOW P/B, LOW P/S, HIGH FCF yield — i.e., classic value.
    """
    pe = _safe(fund.get("forwardPE")) or _safe(fund.get("trailingPE"))
    peg = _safe(fund.get("pegRatio"))
    pb = _safe(fund.get("priceToBook"))
    ps = _safe(fund.get("priceToSalesTrailing12Months"))
    roe = _safe(fund.get("returnOnEquity"))
    margin = _safe(fund.get("profitMargins"))
    cr = _safe(fund.get("currentRatio"))
    rev_g = _safe(fund.get("revenueGrowth"))
    earn_g = _safe(fund.get("earningsGrowth"))
    fcf = _safe(fund.get("freeCashFlow"))
    mkt_cap = _safe(fund.get("marketCap"))
    de = _safe(fund.get("debtToEquity"))

    # P/E — more aggressive: PE 10 → 0.67, PE 20 → 0.33, PE 30 → 0, PE 50 → -0.67
    value_signal = np.nan if np.isnan(pe) or pe <= 0 else float(np.clip(1.0 - pe / 30.0, -1.0, 1.0))
    # PEG — PEG 1.0 → 0.5, PEG 1.5 → 0.25, PEG 2.0 → 0
    peg_signal = np.nan if np.isnan(peg) or peg <= 0 else float(np.clip(1.0 - peg / 2.0, -1.0, 1.0))
    # P/B — PB 1.0 → 0.67, PB 2.0 → 0.33, PB 3.0 → 0
    pb_signal = np.nan if np.isnan(pb) or pb <= 0 else float(np.clip(1.0 - pb / 3.0, -1.0, 1.0))
    # P/S — PS 1 → 0.75, PS 2 → 0.5, PS 4 → 0
    ps_signal = np.nan if np.isnan(ps) or ps <= 0 else float(np.clip(1.0 - ps / 4.0, -1.0, 1.0))

    roe_norm = float(np.clip(roe / 0.30, 0.0, 1.0)) if not np.isnan(roe) else np.nan
    margin_norm = float(np.clip(margin / 0.30, 0.0, 1.0)) if not np.isnan(margin) else np.nan
    cr_norm = float(np.clip((cr - 0.5) / 2.0, 0.0, 1.0)) if not np.isnan(cr) else np.nan

    quality_parts = [(v, w) for v, w in [(roe_norm, 0.4), (margin_norm, 0.4), (cr_norm, 0.2)] if not (v is None or np.isnan(v))]
    quality_score = float(sum(v * w for v, w in quality_parts) / sum(w for _, w in quality_parts)) if quality_parts else np.nan

    growth_parts = [g for g in [rev_g, earn_g] if not np.isnan(g)]
    growth_signal = float(np.clip(np.mean(growth_parts), -1.0, 1.0)) if growth_parts else np.nan

    # FCF yield — heavily reward 5%+ yield (classic value play). 5% → 0.5, 10% → 1.0
    if not np.isnan(fcf) and not np.isnan(mkt_cap) and mkt_cap > 0:
        raw_fcf_yield = fcf / mkt_cap
        fcf_yield = float(np.clip(raw_fcf_yield * 10.0, -1.0, 1.0))
    else:
        fcf_yield = np.nan

    leverage_flag = float(np.clip(de / 200.0, 0.0, 1.0)) if not np.isnan(de) else 0.0

    # Combined value composite — average of available value metrics
    value_components = [v for v in [value_signal, peg_signal, pb_signal, ps_signal] if not np.isnan(v)]
    value_composite = float(np.mean(value_components)) if value_components else np.nan

    return {
        "value_signal": value_signal, "peg_signal": peg_signal,
        "pb_signal": pb_signal, "ps_signal": ps_signal,
        "value_composite": value_composite,
        "quality_score": quality_score, "growth_signal": growth_signal,
        "fcf_yield": fcf_yield, "leverage_flag": leverage_flag,
    }


def build_signal_matrix(all_raw_signals, all_fundamentals, all_news_sentiment=None) -> pd.DataFrame:
    rows: dict[str, dict[str, float]] = {}
    for tkr in all_raw_signals:
        tech = all_raw_signals[tkr]
        fund_signals = compute_fundamental_signals(all_fundamentals.get(tkr, {}))
        news_score = 0.0
        if all_news_sentiment and tkr in all_news_sentiment:
            raw_news = all_news_sentiment[tkr].get("score", 0.0)
            news_score = float(raw_news) if raw_news is not None else 0.0
        rows[tkr] = {**tech, **fund_signals, "news_score": news_score}
    return pd.DataFrame(rows).T


def score_asset(
    ticker, z_scores, raw_signals, raw_fundamentals, weights,
    min_confidence_score=0.85, news_sentiment=None, health_scores=None,
    risk_checks=None, leadership=None, company_analysis=None,
    macro_data=None, relative_strength=None, price_structure=None,
    volume_profile=None,
) -> dict[str, Any]:
    tw = weights.get("technical_weights", {})
    fw = weights.get("fundamental_weights", {})

    def z(key):
        v = z_scores.get(key, 0.0)
        return 0.0 if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)

    # Technical score components — VALUE TILTED
    # Momentum is REDUCED in weight; we don't want to chase rallies
    momentum_z = np.nanmean([z("ret_63"), z("ret_126")])
    trend_z = float(z("is_golden_cross")) + float(z("cross_direction")) * 0.5
    macd_z = z("macd_hist")
    vol_sign = 1.0 if raw_signals.get("ret_5d", 0) >= 0 else -1.0
    volume_z = z("volume_trend") * vol_sign

    # RSI — INVERTED for value: reward oversold/healthy (30-55), penalize overbought (70+)
    raw_rsi = float(raw_signals.get("rsi", 50.0))
    if raw_rsi < 30:
        rsi_value_score = 1.0       # deeply oversold = great buy zone
    elif raw_rsi < 45:
        rsi_value_score = 0.6       # oversold-ish
    elif raw_rsi < 55:
        rsi_value_score = 0.3       # healthy mid-range
    elif raw_rsi < 65:
        rsi_value_score = -0.2      # getting heated
    elif raw_rsi < 75:
        rsi_value_score = -0.6      # overbought
    else:
        rsi_value_score = -1.0      # extreme overbought = avoid

    # BB %B — INVERTED: reward LOW (near lower band = oversold), penalize HIGH (near upper band)
    raw_bb = float(raw_signals.get("bb_pct", 0.5))
    if raw_bb < 0.20:
        bb_value_score = 1.0        # at lower band = great
    elif raw_bb < 0.40:
        bb_value_score = 0.5
    elif raw_bb < 0.60:
        bb_value_score = 0.0        # neutral
    elif raw_bb < 0.80:
        bb_value_score = -0.4       # extended
    else:
        bb_value_score = -1.0       # at upper band = expensive

    # ── DISCOUNT FROM HIGHS — the key value signal ──
    pct_from_52wk_high = float(raw_signals.get("pct_from_52wk_high", 0.0))
    pct_vs_sma200 = float(raw_signals.get("pct_vs_sma200", 0.0))
    range_position = float(raw_signals.get("range_position_52w", 0.5))

    # Discount score: best when 15-35% below 52-week high (sweet spot for value)
    if pct_from_52wk_high > -0.02:
        discount_score = -1.0       # within 2% of 52w high — NO DISCOUNT, AT PREMIUM
    elif pct_from_52wk_high > -0.08:
        discount_score = -0.5       # within 8% of high — still expensive
    elif pct_from_52wk_high > -0.15:
        discount_score = 0.2        # mild pullback
    elif pct_from_52wk_high > -0.25:
        discount_score = 0.8        # nice 15-25% pullback — value zone
    elif pct_from_52wk_high > -0.40:
        discount_score = 1.0        # 25-40% off — deep value (if fundamentals strong)
    elif pct_from_52wk_high > -0.55:
        discount_score = 0.5        # 40-55% off — possible value trap, be careful
    else:
        discount_score = -0.3       # >55% off — likely broken business

    # Range position penalty: if at top of 52-week range, penalize
    if range_position > 0.90:
        range_penalty = -0.5
    elif range_position > 0.80:
        range_penalty = -0.2
    else:
        range_penalty = 0.0

    # SMA premium penalty: if trading 25%+ above 200-SMA, it's overextended
    if pct_vs_sma200 > 0.30:
        sma_premium_penalty = -0.4  # 30%+ above 200-SMA = overheated
    elif pct_vs_sma200 > 0.15:
        sma_premium_penalty = -0.1
    else:
        sma_premium_penalty = 0.0

    technical_score = float(
        tw.get("momentum", 0.04) * momentum_z          # was 0.10 — REDUCED
        + tw.get("trend", 0.05) * trend_z
        + tw.get("rsi", 0.06) * rsi_value_score        # was 0.02 — INCREASED & inverted
        + tw.get("macd", 0.02) * macd_z
        + tw.get("bb_pct", 0.05) * bb_value_score      # was 0.01 — INCREASED & inverted
        + tw.get("volume", 0.01) * volume_z
        + tw.get("discount", 0.15) * discount_score    # NEW — major value signal
        + range_penalty
        + sma_premium_penalty
    )

    # Relative strength — DOWNWEIGHTED so we don't chase RS leaders at highs
    rs_percentile_raw: float = np.nan
    if relative_strength:
        rs_percentile_raw = _safe(relative_strength.get("rs_percentile"))
        rs_63d_val = _safe(relative_strength.get("rs_vs_spy_63d"))
        rs_persist_val = _safe(relative_strength.get("rs_persistent"))
        rs_63d_z = z("rs_vs_spy_63d") if "rs_vs_spy_63d" in z_scores else (float(np.clip(rs_63d_val * 5, -3, 3)) if not np.isnan(rs_63d_val) else 0.0)
        rs_pct_z = z("rs_percentile") if "rs_percentile" in z_scores else (float((rs_percentile_raw - 0.5) * 6) if not np.isnan(rs_percentile_raw) else 0.0)
        rs_persist_z = z("rs_persistent") if "rs_persistent" in z_scores else (float((rs_persist_val - 0.5) * 4) if not np.isnan(rs_persist_val) else 0.0)
        technical_score += tw.get("relative_strength", 0.03) * float(np.nanmean([rs_63d_z, rs_pct_z, rs_persist_z]))

    # Price structure
    if price_structure:
        ps_z = float(np.nanmean([
            z("trend_slope_63d") if "trend_slope_63d" in z_scores else 0.0,
            float(_safe(price_structure.get("hh_hl"), 0.0)),
            z("trend_efficiency") if "trend_efficiency" in z_scores else 0.0,
            z("sharpe_63d") if "sharpe_63d" in z_scores else 0.0,
        ]))
        technical_score += tw.get("price_structure", 0.03) * ps_z

    # Volume profile
    if volume_profile:
        vp_z = float(np.nanmean([
            z("cmf") if "cmf" in z_scores else 0.0,
            z("ud_volume_ratio") if "ud_volume_ratio" in z_scores else 0.0,
            z("obv_slope") if "obv_slope" in z_scores else 0.0,
        ]))
        technical_score += tw.get("volume_profile", 0.03) * vp_z

    # NO breakout bonus — breakouts mean it already moved
    breakout_bonus = 0.0

    # No RS laggard penalty — value plays often have weak recent RS
    rs_laggard_penalty = 0.0

    # rsi_z and bb_z for confidence-counting (use the value-flipped version)
    rsi_z = rsi_value_score
    bb_z = bb_value_score

    # Fundamental score
    fund_signals = compute_fundamental_signals(raw_fundamentals)

    def fs(key):
        v = fund_signals.get(key, np.nan)
        return 0.0 if (v is None or np.isnan(v)) else float(v)

    # Value composite — uses PE + PEG + PB + PS averaged. Heavy weight for value.
    value = fs("value_composite") if not np.isnan(fund_signals.get("value_composite", np.nan)) else fs("value_signal")
    fundamental_score = float(
        fw.get("value", 0.25) * value                      # was 0.12 — DOUBLED
        + fw.get("quality", 0.18) * fs("quality_score")    # quality matters more for value plays
        + fw.get("growth", 0.08) * fs("growth_signal")     # was 0.13 — REDUCED (value > growth)
        + fw.get("fcf_yield", 0.18) * fs("fcf_yield")      # was 0.10 — boosted heavily
        + fw.get("leverage_penalty", -0.08) * fs("leverage_flag")
    )

    # Adjustments
    news_raw_score, news_signal = 0.0, "NEUTRAL"
    if news_sentiment:
        news_raw_score = float(news_sentiment.get("score", 0.0))
        news_signal = news_sentiment.get("signal", "NEUTRAL")
    news_adjustment = news_raw_score * 0.10 if news_raw_score < 0 else news_raw_score * 0.05

    health_adjustment, health_grade, altman_zone = 0.0, "UNKNOWN", "UNKNOWN"
    bs_blocks_high = False
    if health_scores:
        health_grade = health_scores.get("health_grade", "UNKNOWN")
        altman_zone = health_scores.get("altman", {}).get("zone", "UNKNOWN")
        h_score = float(health_scores.get("health_score", 0.0))
        health_adjustment = h_score * 0.10 if h_score >= 0 else h_score * 0.20
        # Balance sheet quality gate
        bs_data = health_scores.get("balance_sheet", {})
        bs_blocks_high = bool(bs_data.get("blocks_high", False))

    risk_adjustment, overall_risk = 0.0, "UNKNOWN"
    if risk_checks:
        overall_risk = risk_checks.get("overall_risk", "UNKNOWN")
        risk_adjustment = float(risk_checks.get("risk_score", 0.0)) * 0.15

    leadership_adjustment = 0.0
    if leadership:
        leadership_adjustment = float(leadership.get("leadership_score", 0.0)) * 0.05

    company_adjustment, company_grade, growth_tier, moat_grade = 0.0, "ADEQUATE", "UNKNOWN", "UNKNOWN"
    if company_analysis:
        c_score = float(company_analysis.get("company_score", 0.0))
        company_grade = company_analysis.get("company_grade", "ADEQUATE")
        growth_tier = company_analysis.get("industry_growth", {}).get("growth_tier", "UNKNOWN")
        moat_grade = company_analysis.get("moat", {}).get("moat_grade", "UNKNOWN")
        company_adjustment = c_score * 0.10 if c_score >= 0 else c_score * 0.08

    macro_adjustment, macro_grade, macro_blocks_high = 0.0, "NEUTRAL", False
    if macro_data:
        m_score = float(macro_data.get("macro_score", 0.0))
        macro_grade = macro_data.get("macro_grade", "NEUTRAL")
        macro_blocks_high = bool(macro_data.get("blocks_high_confidence", False))
        macro_adjustment = m_score * 0.06 if m_score >= 0 else m_score * 0.12

    # Composite — raw weighted sum
    w_tech = weights.get("weights", {}).get("technical", 0.45)
    w_fund = weights.get("weights", {}).get("fundamental", 0.55)
    raw_composite = float(
        w_tech * technical_score + w_fund * fundamental_score
        + news_adjustment + health_adjustment + risk_adjustment
        + leadership_adjustment + company_adjustment + macro_adjustment
        + breakout_bonus + rs_laggard_penalty
    )

    # Rescale to 0-1 range using sigmoid-like transform.
    # Empirically, raw_composite spans roughly [-0.5, +0.6] across the universe.
    # We map -0.4 → 0.10, 0.0 → 0.50, +0.4 → 0.90 so good stocks naturally
    # land in the 0.75-0.95 range that the threshold system expects.
    composite_score = float(1.0 / (1.0 + np.exp(-5.0 * raw_composite)))

    # Confidence
    positive_technical_signals = sum([
        momentum_z > 0, trend_z > 0, rsi_z > 0, macd_z > 0, bb_z > 0, volume_z > 0,
        bool(relative_strength and _safe(relative_strength.get("rs_vs_spy_63d"), -1) > 0),
        bool(price_structure and _safe(price_structure.get("trend_slope_63d"), -1) > 0),
        bool(volume_profile and _safe(volume_profile.get("cmf"), -1) > 0),
    ])
    news_blocks_high = news_signal == "BEARISH" and news_raw_score < -0.40
    health_blocks_high = altman_zone == "DISTRESS"
    risk_blocks_high = overall_risk == "HIGH"
    # Earnings within 30 days = block. Earnings gaps are noise vs. the strategy.
    earnings_within_window = has_earnings_within(risk_checks or {}, days=30)

    # ── FALLING KNIFE / VALUE TRAP FILTERS ─────────────────────────────
    # Industry must NOT be in a structural decline (coal, tobacco, etc.)
    industry_blocks = growth_tier == "LOW"
    # Moat must NOT be disadvantaged (losing competitive position)
    moat_blocks = moat_grade == "DISADVANTAGED"

    # Revenue must NOT be shrinking >5% YoY (we want growing or stable revenue)
    rev_growth = _safe(raw_fundamentals.get("revenueGrowth"))
    revenue_shrinking = (not np.isnan(rev_growth)) and rev_growth < -0.05

    # Earnings must NOT be collapsing (>20% decline if data is available)
    earn_growth = _safe(raw_fundamentals.get("earningsGrowth"))
    earnings_collapsing = (not np.isnan(earn_growth)) and earn_growth < -0.30

    # Piotroski WEAK = fundamentals deteriorating (block)
    piotroski_grade = "UNKNOWN"
    if health_scores:
        piotroski_grade = health_scores.get("piotroski", {}).get("grade", "UNKNOWN")
    fundamentals_deteriorating = piotroski_grade == "WEAK"

    # Leadership turmoil = block
    leadership_score_val = float(leadership.get("leadership_score", 0.0)) if leadership else 0.0
    dual_transition = bool(leadership and leadership.get("dual_transition_risk", False))
    leadership_blocks = leadership_score_val < -0.20 or dual_transition

    # Falling knife: down >55% from 52w high = something likely broken
    pct_from_52wk_high = float(raw_signals.get("pct_from_52wk_high", 0.0))
    deeply_broken = pct_from_52wk_high < -0.55

    # Free-fall detection: 63-day return < -25% AND below 200-SMA AND no stabilization
    ret_63 = _safe(raw_signals.get("ret_63"))
    pct_vs_sma200 = float(raw_signals.get("pct_vs_sma200", 0.0))
    free_fall = (
        not np.isnan(ret_63) and ret_63 < -0.25
        and pct_vs_sma200 < -0.10
        and not bool(raw_signals.get("is_golden_cross", False))
    )

    # Insider cluster selling = vote of no confidence
    insider_signal = "NEUTRAL"
    if risk_checks:
        insider_signal = risk_checks.get("insider", {}).get("signal", "NEUTRAL")
    insider_dumping = insider_signal == "BEARISH"

    value_trap_blocks = (
        industry_blocks or moat_blocks or revenue_shrinking
        or earnings_collapsing or fundamentals_deteriorating
        or leadership_blocks or deeply_broken or free_fall
        or insider_dumping
    )

    # Build flags list for transparency in report
    trap_reasons: list[str] = []
    if industry_blocks: trap_reasons.append("LOW-growth industry")
    if moat_blocks: trap_reasons.append("disadvantaged moat")
    if revenue_shrinking: trap_reasons.append(f"revenue shrinking {rev_growth:.0%}")
    if earnings_collapsing: trap_reasons.append(f"earnings down {earn_growth:.0%}")
    if fundamentals_deteriorating: trap_reasons.append("Piotroski WEAK (fundamentals deteriorating)")
    if leadership_blocks: trap_reasons.append("leadership turmoil")
    if deeply_broken: trap_reasons.append(f"down {pct_from_52wk_high:.0%} from high (likely broken)")
    if free_fall: trap_reasons.append("free-fall pattern")
    if insider_dumping: trap_reasons.append("insider cluster selling")

    # Quality gates that ALWAYS apply
    if (composite_score >= min_confidence_score and positive_technical_signals >= 3
            and fundamental_score > 0.0 and not news_blocks_high
            and not health_blocks_high and not risk_blocks_high
            and not macro_blocks_high and not bs_blocks_high
            and not value_trap_blocks
            and not earnings_within_window):
        confidence = "HIGH"
    elif composite_score >= 0.60 and not value_trap_blocks:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    return {
        "ticker": ticker, "technical_score": technical_score,
        "fundamental_score": fundamental_score, "news_score": news_raw_score,
        "composite_score": composite_score, "confidence": confidence,
        "z_scores": z_scores, "raw_signals": raw_signals,
        "raw_fundamentals": raw_fundamentals,
        "news_sentiment": news_sentiment or {},
        "health_scores": health_scores or {},
        "risk_checks": risk_checks or {},
        "leadership": leadership or {},
        "company_analysis": company_analysis or {},
        "macro_data": macro_data or {},
        "relative_strength": relative_strength or {},
        "price_structure": price_structure or {},
        "volume_profile": volume_profile or {},
        "breakout_bonus": breakout_bonus,
        "rs_laggard_penalty": rs_laggard_penalty,
        "value_trap_blocks": value_trap_blocks,
        "trap_reasons": trap_reasons,
    }


def compute_sector_medians(fundamentals_by_ticker) -> dict[str, dict[str, float]]:
    _METRICS = ["trailingPE", "forwardPE", "priceToBook", "profitMargins", "returnOnEquity", "debtToEquity"]
    sector_buckets: dict[str, list[dict]] = {}
    for fund in fundamentals_by_ticker.values():
        sector = fund.get("sector") or "__universe__"
        sector_buckets.setdefault(sector, []).append(fund)
    all_funds = list(fundamentals_by_ticker.values())

    def _median(funds, metric):
        vals = [_safe(f.get(metric)) for f in funds]
        vals = [v for v in vals if not np.isnan(v)]
        return float(np.median(vals)) if vals else np.nan

    universe_medians = {m: _median(all_funds, m) for m in _METRICS}
    result: dict[str, dict[str, float]] = {"__universe__": universe_medians}
    for sector, funds in sector_buckets.items():
        result[sector] = {m: _median(funds, m) for m in _METRICS} if len(funds) >= 3 else universe_medians
    return result


# =====================================================================
# EXPLANATION — generates pros/cons/verdict + trade plan
# =====================================================================


def generate_explanation(
    ticker, score_bundle, sector_medians, thresholds=None, regime_state=None,
) -> dict[str, Any]:
    t = {
        "momentum_strong": 0.15, "rsi_healthy_low": 35.0, "rsi_healthy_high": 55.0,
        "rsi_overbought": 70.0, "rsi_oversold": 25.0, "volume_trend_strong": 1.3,
        "bb_pct_oversold": 0.25, "bb_pct_extended": 0.85, "roe_strong": 0.15,
        "peg_cheap": 1.5, "peg_expensive": 2.5, "pe_high": 40.0,
        "debt_equity_high": 150.0, "revenue_growth_strong": 0.10,
        "fcf_yield_positive": 0.03, "fcf_yield_negative": -0.02,
        "beta_high": 1.5, "profit_margin_strong": 0.15,
    }
    if thresholds:
        t.update(thresholds)

    sig = score_bundle.get("raw_signals", {})
    fund = score_bundle.get("raw_fundamentals", {})
    composite = score_bundle.get("composite_score", 0.0)
    news = score_bundle.get("news_sentiment", {})
    pros: list[str] = []
    cons: list[str] = []

    # Regime
    if regime_state is not None:
        regime_name = str(regime_state.regime.value)
        if regime_name == "BULL_QUIET":
            pros.insert(0, f"+ Market regime: {regime_name} — {regime_state.description}")
        elif regime_name in ("BEAR_VOLATILE", "CRISIS"):
            cons.append(f"- Market regime: {regime_name} — {regime_state.description}")

    # Technical
    ret_126 = _safe(sig.get("ret_126"))
    if not np.isnan(ret_126) and ret_126 > t["momentum_strong"]:
        pros.append(f"+ Strong momentum: {ret_126:+.0%} over 6 months")
    elif not np.isnan(ret_126) and ret_126 > 0.05:
        pros.append(f"+ Positive momentum: {ret_126:+.0%} over 6 months")
    if bool(sig.get("is_golden_cross", 0)):
        pros.append("+ Above 200-day SMA with SMA50 in uptrend")
    rsi = _safe(sig.get("rsi"))
    if not np.isnan(rsi) and t["rsi_healthy_low"] <= rsi <= t["rsi_healthy_high"]:
        pros.append(f"+ RSI {rsi:.0f} — healthy range")
    macd_hist = _safe(sig.get("macd_hist"))
    if not np.isnan(macd_hist) and macd_hist > 0:
        pros.append("+ MACD histogram positive")

    # Fundamental
    roe = _safe(fund.get("returnOnEquity"))
    if not np.isnan(roe) and roe > t["roe_strong"]:
        pros.append(f"+ ROE {roe:.0%}")
    peg = _safe(fund.get("pegRatio"))
    if not np.isnan(peg) and peg < t["peg_cheap"]:
        pros.append(f"+ PEG ratio {peg:.1f}")
    rev_g = _safe(fund.get("revenueGrowth"))
    if not np.isnan(rev_g) and rev_g > t["revenue_growth_strong"]:
        pros.append(f"+ Revenue growing {rev_g:+.0%} YoY")

    # Technical cons
    if not np.isnan(rsi) and rsi > t["rsi_overbought"]:
        cons.append(f"- RSI {rsi:.0f} — overbought")
    if not np.isnan(macd_hist) and macd_hist < 0:
        cons.append("- MACD histogram negative")
    if not np.isnan(ret_126) and ret_126 < -0.05:
        cons.append(f"- Negative 6-month momentum: {ret_126:+.0%}")

    # Fundamental cons
    pe = _safe(fund.get("forwardPE")) or _safe(fund.get("trailingPE"))
    if not np.isnan(pe) and pe > t["pe_high"]:
        cons.append(f"- High P/E of {pe:.0f}")
    de = _safe(fund.get("debtToEquity"))
    if not np.isnan(de) and de > t["debt_equity_high"]:
        cons.append(f"- Debt-to-equity {de:.0f}%")

    # News
    news_score = float(news.get("score", 0.0)) if news else 0.0
    news_signal_str = news.get("signal", "NEUTRAL") if news else "NEUTRAL"
    n_total = int(news.get("n_headlines", 0)) if news else 0
    if n_total == 0:
        cons.append("- No recent news found")
    elif news_signal_str == "BULLISH":
        pros.append(f"+ News sentiment BULLISH (score {news_score:+.2f})")
    elif news_signal_str == "BEARISH":
        cons.append(f"- News sentiment BEARISH (score {news_score:+.2f})")

    # Health
    health = score_bundle.get("health_scores", {})
    if health:
        h_grade = health.get("health_grade", "UNKNOWN")
        if h_grade in ("EXCELLENT", "GOOD"):
            pros.append(f"+ Financial health {h_grade}")
        elif h_grade in ("FAIR", "POOR"):
            cons.append(f"- Financial health {h_grade}")

    # Risk flags
    risk = score_bundle.get("risk_checks", {})
    if risk:
        for flag in risk.get("risk_flags", []):
            if "INSIDER BUYING" in flag or "UPGRADES" in flag:
                pros.append(f"+ {flag}")
            else:
                cons.append(f"- {flag}")

    # Relative strength
    rs_data = score_bundle.get("relative_strength", {})
    if rs_data:
        rs_63d = _safe(rs_data.get("rs_vs_spy_63d"))
        if not np.isnan(rs_63d) and rs_63d > 0.05:
            pros.append(f"+ Strong RS vs SPY: {rs_63d:+.0%} over 3 months")
        elif not np.isnan(rs_63d) and rs_63d < -0.05:
            cons.append(f"- RS vs SPY: {rs_63d:+.0%} — underperforming")

    # Verdict
    n_pros, n_cons = len(pros), len(cons)
    if composite >= 0.80 and n_pros >= 2 * max(n_cons, 1):
        verdict = "Strong buy — multiple factors align with high conviction."
    elif composite >= 0.65 and n_pros > n_cons:
        verdict = "Buy — favourable risk/reward."
    elif n_pros > n_cons:
        verdict = "Moderate buy — positive but watch the cons."
    elif n_pros == n_cons:
        verdict = "Mixed signals — monitor closely."
    else:
        verdict = "Risk outweighs reward — avoid or reduce."

    net_signal = "BUY" if n_pros >= 2 * max(n_cons, 1) and composite > 0.60 else ("HOLD" if composite > 0.40 else "AVOID")
    return {"pros": pros, "cons": cons, "verdict": verdict, "net_signal": net_signal}


def generate_trade_plan(score_bundle, cfg=None) -> dict[str, Any]:
    sig = score_bundle.get("raw_signals", {})
    ps = score_bundle.get("price_structure", {})
    current_close = _safe(sig.get("close"))
    current_atr = _safe(sig.get("atr"))

    atr_stop_pct: float | None = None
    if not np.isnan(current_close) and not np.isnan(current_atr) and current_close > 0:
        atr_stop_pct = (2.0 * current_atr) / current_close

    target_1_pct = atr_stop_pct * 1.5 if atr_stop_pct else None
    target_2_pct = atr_stop_pct * 2.5 if atr_stop_pct else None

    is_bo = bool(ps.get("is_breakout", False))
    retest = bool(ps.get("retest_holding", False))
    in_base = bool(ps.get("in_base", False))

    if is_bo:
        entry_note = "Breakout entry — buy on close above resistance"
    elif retest:
        entry_note = "Retest entry — prior breakout acting as support"
    elif in_base:
        entry_note = "Base setup — wait for volume-confirmed breakout"
    else:
        entry_note = "Trend continuation — buy near current price or on pullback"

    return {
        "entry_note": entry_note, "atr_stop_pct": atr_stop_pct,
        "structural_stop_note": "Below key support level",
        "target_1_pct": target_1_pct, "target_2_pct": target_2_pct,
        "time_stop_days": 30,
        "scaling_note": "Add if RS vs SPY exceeds +5% since entry",
        "risk_reward": 2.5 if target_2_pct else None,
    }


def generate_sell_alerts(
    held_positions, score_bundles, regime_state=None, sell_cfg=None,
) -> list[dict[str, Any]]:
    news_threshold = -0.30
    regime_sell_on = ["CRISIS", "BEAR_VOLATILE"]
    rsi_sell = 78.0
    if sell_cfg is not None:
        if hasattr(sell_cfg, "news_score_threshold"):
            news_threshold = float(sell_cfg.news_score_threshold)
        if hasattr(sell_cfg, "regime_sell_on"):
            regime_sell_on = list(sell_cfg.regime_sell_on)

    alerts: list[dict[str, Any]] = []
    for tkr in held_positions:
        if tkr not in score_bundles:
            continue
        bundle = score_bundles[tkr]
        sig = bundle.get("raw_signals", {})
        news = bundle.get("news_sentiment", {})
        reasons: list[str] = []
        urgency_flags: list[int] = []

        news_score = float(news.get("score", 0.0)) if news else 0.0
        if news_score < news_threshold:
            reasons.append(f"Negative news sentiment: {news_score:+.2f}")
            urgency_flags.append(2 if news_score < -0.50 else 1)

        if regime_state is not None:
            current_regime = str(regime_state.regime.value)
            if current_regime in regime_sell_on:
                reasons.append(f"Market regime is {current_regime}: {regime_state.description}")
                urgency_flags.append(2 if current_regime == "CRISIS" else 1)

        rsi = float(sig.get("rsi", 50.0))
        if rsi > rsi_sell:
            reasons.append(f"RSI {rsi:.0f} — overbought")
            urgency_flags.append(1)

        macd_hist = float(sig.get("macd_hist", 0.0))
        if macd_hist < 0:
            reasons.append("MACD histogram negative")
            urgency_flags.append(1)

        if not reasons:
            continue

        urgency = "URGENT" if 2 in urgency_flags else "CAUTION"
        recommendation = "EXIT position immediately" if urgency == "URGENT" else "Consider reducing or tighter stop-loss"
        alerts.append({
            "ticker": tkr, "reasons": reasons,
            "urgency": urgency, "recommendation": recommendation,
        })
    return alerts


def build_report(
    ranked_candidates, explanations, report_date, output_dir,
    min_confidence_score=0.80, sell_alerts=None, regime_state=None,
    macro_data=None,
) -> str:
    lines: list[str] = []
    divider = "-" * 64

    lines.append(f"=== STOCK RECOMMENDATIONS [{report_date}] ===")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Universe: {len(ranked_candidates)} HIGH-confidence picks")

    if regime_state is not None:
        r = regime_state
        lines.append(f"\n  MARKET REGIME: {r.regime.value}")
        lines.append(f"  {r.description}")
        lines.append(f"  SPY: {r.spy_price:.2f} | 200-SMA: {r.spy_sma200:.2f} | Vol: {r.realised_vol:.0%}")

    if macro_data:
        lines.append(f"\n  MACRO: {macro_data.get('macro_grade', 'NEUTRAL')}")
        lines.append(f"  {macro_data.get('summary', '')}")

    lines.append("")

    if sell_alerts:
        lines.append("=" * 64)
        lines.append("  SELL / REDUCE ALERTS")
        lines.append("=" * 64)
        for alert in sell_alerts:
            lines.append(f"\n  [{alert['urgency']}] {alert['ticker']}")
            for reason in alert["reasons"]:
                lines.append(f"    * {reason}")
            lines.append(f"  -> {alert['recommendation']}")
        lines.append(f"\n{divider}\n")

    if not ranked_candidates:
        lines.append("No high-conviction picks today.")
        lines.append(f"Confidence threshold: {min_confidence_score:.0%}")
        report = "\n".join(lines)
        _write_report(report, output_dir, report_date)
        return report

    for rank, (bundle, expl) in enumerate(zip(ranked_candidates, explanations), start=1):
        tkr = bundle["ticker"]
        score = bundle["composite_score"]
        conf = bundle["confidence"]
        fund = bundle.get("raw_fundamentals", {})

        lines.append(f"#{rank}  {tkr:<6}  (Score: {score:.2f} | Confidence: {conf})")
        lines.append(f"    Sector: {fund.get('sector', 'Unknown')}  |  Industry: {fund.get('industry', 'Unknown')}")

        lines.append("\n    WHY IT WILL INCREASE:")
        if expl.get("pros"):
            lines.append("      Pros:")
            for p in expl["pros"]:
                lines.append(f"        {p}")
        if expl.get("cons"):
            lines.append("      Cons:")
            for c in expl["cons"]:
                lines.append(f"        {c}")
        lines.append(f"\n      Verdict [{expl.get('net_signal', '')}]: {expl.get('verdict', '')}")

        trade = apply_slippage_to_trade(generate_trade_plan(bundle))
        lines.append(f"\n    TRADE PLAN (slippage-adjusted, 0.2% in/out):")
        lines.append(f"      Entry:   {trade['entry_note']}")
        if trade["atr_stop_pct"] is not None:
            lines.append(f"      Stop:    -{trade['atr_stop_pct']:.1%} below entry (2x ATR + slip)")
            lines.append(f"      Target1: +{trade['target_1_pct']:.1%} (net of costs)")
            lines.append(f"      Target2: +{trade['target_2_pct']:.1%} (net of costs)")
        lines.append(f"      Time:    Re-evaluate in {trade['time_stop_days']} days")
        # Show liquidity warning if low
        adv = bundle.get("raw_signals", {}).get("avg_dollar_volume", 0)
        if adv and adv < 10_000_000:
            lines.append(f"      ⚠ Liquidity: ${adv/1e6:.1f}M/day avg — keep position size small")
        lines.append(f"\n{divider}\n")

    lines.append(f"NOTE: Only stocks scoring >= {min_confidence_score:.0%} composite confidence shown.")
    lines.append("This is algorithmic output. Always do your own due diligence.")
    report = "\n".join(lines)
    _write_report(report, output_dir, report_date)
    return report


def _write_report(report: str, output_dir, report_date: str) -> None:
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"report_{report_date}.txt"
        path.write_text(report, encoding="utf-8")
        logger.info("Report written to %s", path)
    except Exception as exc:
        logger.warning("Could not write report to disk: %s", exc)


# =====================================================================
# BACKTEST (train) — historical portfolio simulation
# =====================================================================


class Portfolio:
    def __init__(self, initial_capital: float) -> None:
        self.cash = initial_capital
        self.positions: dict[str, dict] = {}
        self.history: list[float] = []
        self.entry_dates: dict[str, int] = {}

    def value(self, prices: dict[str, float]) -> float:
        total = self.cash
        for tkr, pos in self.positions.items():
            total += pos["shares"] * prices.get(tkr, 0.0)
        return total

    def rebalance(self, target_alloc, prices, adv_map, transaction_cost, slippage, max_adv_participation, turnover_penalty):
        pv = self.value(prices)
        penalised = {}
        for tkr, weight in target_alloc.items():
            penalised[tkr] = weight * (1 - turnover_penalty) if tkr in self.positions else weight
        for tkr, tw in penalised.items():
            price = prices.get(tkr, np.nan)
            if np.isnan(price) or price <= 0:
                continue
            target_value = pv * tw
            current_shares = self.positions.get(tkr, {}).get("shares", 0.0)
            diff = target_value - current_shares * price
            adv = adv_map.get(tkr, np.nan)
            max_trade = (adv * price * max_adv_participation) if not np.isnan(adv) else abs(diff)
            trade_value = float(np.clip(abs(diff), 0.0, max_trade))
            if diff > 0:
                cost = trade_value * (1 + slippage)
                fee = cost * transaction_cost
                if self.cash >= cost + fee:
                    self.positions.setdefault(tkr, {"shares": 0.0})
                    self.positions[tkr]["shares"] += trade_value / price
                    self.cash -= cost + fee
                    self.entry_dates.setdefault(tkr, len(self.history))
            else:
                shares_to_sell = min(trade_value / price, current_shares)
                if shares_to_sell > 0:
                    proceeds = shares_to_sell * price * (1 - slippage)
                    self.cash += proceeds - proceeds * transaction_cost
                    self.positions[tkr]["shares"] -= shares_to_sell
                    if self.positions[tkr]["shares"] <= 1e-8:
                        del self.positions[tkr]

    def apply_stops(self, prices, time_stop_days):
        for tkr in list(self.positions.keys()):
            if len(self.history) - self.entry_dates.get(tkr, 0) > time_stop_days:
                proceeds = self.positions[tkr]["shares"] * prices.get(tkr, 0.0)
                self.cash += proceeds
                del self.positions[tkr]


def run_backtest(cfg_path=None, debug=False) -> dict[str, float]:
    cfg = load_config(cfg_path)
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    universe = cfg.debug_universe if debug else cfg.universe
    start_date = cfg.debug_start_date if debug else cfg.start_date

    price_data = download_prices(universe, start_date)
    available = [t for t in universe if t in price_data]
    adv_map = compute_adv_map(price_data)

    all_indicators: dict[str, pd.DataFrame] = {}
    for tkr in available:
        df = price_data[tkr]
        if len(df) < 200:
            continue
        all_indicators[tkr] = compute_all_indicators(df, cfg)

    qualified = list(all_indicators.keys())
    all_fundamentals = fetch_all_fundamentals(qualified)
    weights = config_to_weights_dict(cfg)

    portfolio = Portfolio(cfg.initial_capital)
    portfolio_values: list[float] = []
    dates = all_indicators[qualified[0]].index

    for step, dt in enumerate(dates):
        if step % cfg.rebalance_days == 0 and step > 0:
            raw_signals_snap: dict[str, dict] = {}
            for tkr in qualified:
                ind_slice = all_indicators[tkr].loc[:dt].dropna(subset=["ret_126", "sma200"])
                if len(ind_slice) < 200:
                    continue
                raw_signals_snap[tkr] = extract_latest_signals(ind_slice)
            if len(raw_signals_snap) >= 2:
                signal_matrix = build_signal_matrix(raw_signals_snap, all_fundamentals)
                z_matrix = cross_sectional_zscore(signal_matrix)
                score_bundles: dict[str, dict] = {}
                for tkr in raw_signals_snap:
                    z_row = z_matrix.loc[tkr].to_dict() if tkr in z_matrix.index else {}
                    score_bundles[tkr] = score_asset(
                        ticker=tkr, z_scores=z_row,
                        raw_signals=raw_signals_snap[tkr],
                        raw_fundamentals=all_fundamentals[tkr],
                        weights=weights,
                        min_confidence_score=cfg.min_confidence_score,
                    )
                ranked = sorted(score_bundles.items(), key=lambda x: x[1]["composite_score"], reverse=True)
                top = [t for t, _ in ranked[:cfg.top_n]]
                prices = {
                    tkr: float(all_indicators[tkr].loc[dt, "close"])
                    for tkr in qualified if dt in all_indicators[tkr].index
                }
                portfolio.rebalance(
                    dict.fromkeys(top, cfg.risk_budget_per_asset),
                    prices, adv_map, cfg.transaction_cost, cfg.slippage,
                    cfg.max_adv_participation, cfg.turnover_penalty,
                )

        prices_today = {
            tkr: float(all_indicators[tkr].loc[dt, "close"])
            for tkr in qualified if dt in all_indicators[tkr].index
        }
        portfolio.apply_stops(prices_today, cfg.time_stop_days)
        portfolio_values.append(portfolio.value(prices_today))
        portfolio.history.append(portfolio_values[-1])

    return compute_metrics(portfolio_values)


# =====================================================================
# INFERENCE — live recommendation pipeline
# =====================================================================


# =====================================================================
# DATA QUALITY / LIQUIDITY / DIVERSIFICATION HELPERS
# =====================================================================

# Critical fundamental fields. If too many of these are missing, the stock
# is excluded — yfinance is unreliable and bad data drives bad picks.
_CRITICAL_FUND_FIELDS = [
    "marketCap", "trailingPE", "forwardPE", "profitMargins", "returnOnEquity",
    "revenueGrowth", "freeCashFlow", "totalRevenue", "debtToEquity",
    "currentRatio", "operatingMargins",
]


def assess_data_quality(fund: dict, *, min_present: int = 6) -> tuple[bool, int, list[str]]:
    """Returns (is_acceptable, fields_present, missing_fields).

    A stock needs at least `min_present` of the critical fundamental fields
    populated with sane values. Otherwise we can't trust the analysis.
    """
    missing: list[str] = []
    present = 0
    for f in _CRITICAL_FUND_FIELDS:
        v = fund.get(f)
        if v is None:
            missing.append(f)
            continue
        try:
            fv = float(v)
            if np.isnan(fv) or np.isinf(fv):
                missing.append(f)
            else:
                present += 1
        except (TypeError, ValueError):
            missing.append(f)
    return present >= min_present, present, missing


def compute_liquidity(price_df: pd.DataFrame, window: int = 20) -> dict[str, float]:
    """Average daily dollar volume + share volume over recent window.

    Filters out illiquid names where slippage would destroy the edge.
    """
    if price_df is None or price_df.empty or len(price_df) < window:
        return {"avg_dollar_volume": 0.0, "avg_share_volume": 0.0}
    recent = price_df.tail(window)
    close = recent["Close"]
    volume = recent["Volume"] if "Volume" in recent.columns else pd.Series([0.0])
    dollar_vol = (close * volume).mean()
    share_vol = volume.mean()
    return {
        "avg_dollar_volume": float(dollar_vol) if not pd.isna(dollar_vol) else 0.0,
        "avg_share_volume": float(share_vol) if not pd.isna(share_vol) else 0.0,
    }


def apply_sector_diversification(
    bundles: list[dict], all_fundamentals: dict, max_per_sector: int = 3,
) -> list[dict]:
    """Cap exposure to any single sector. Walks the ranked list and keeps
    at most `max_per_sector` picks per sector, preserving rank order.
    Prevents 5 semis from looking like 5 different bets when they're really one.
    """
    sector_counts: dict[str, int] = {}
    diversified: list[dict] = []
    for b in bundles:
        tkr = b["ticker"]
        sector = (all_fundamentals.get(tkr, {}).get("sector") or "Unknown")
        if sector_counts.get(sector, 0) >= max_per_sector:
            continue
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
        diversified.append(b)
    return diversified


def has_earnings_within(risk_checks: dict, days: int = 30) -> bool:
    """True if next earnings is within `days`. Avoids getting stopped by
    earnings gaps which are pure noise relative to the strategy."""
    if not risk_checks:
        return False
    earnings = risk_checks.get("earnings", {})
    days_until = earnings.get("days_until_earnings")
    if days_until is None:
        return False
    try:
        d = int(days_until)
    except (TypeError, ValueError):
        return False
    return 0 <= d <= days


def apply_slippage_to_trade(trade_plan: dict, *, slippage_pct: float = 0.002) -> dict:
    """Adjust trade plan for transaction costs / slippage.

    Default 0.20% in + 0.20% out = ~0.40% round-trip cost. Reduces stated
    targets and widens the effective stop so the plan reflects reality.
    """
    p = dict(trade_plan)
    if p.get("atr_stop_pct") is not None:
        p["atr_stop_pct"] = p["atr_stop_pct"] + slippage_pct
    if p.get("target_1_pct") is not None:
        p["target_1_pct"] = max(0.0, p["target_1_pct"] - 2 * slippage_pct)
    if p.get("target_2_pct") is not None:
        p["target_2_pct"] = max(0.0, p["target_2_pct"] - 2 * slippage_pct)
    p["slippage_applied_pct"] = slippage_pct
    return p


def log_run_results(
    run_date: str, regime: str, macro_grade: str, n_qualified: int,
    n_high: int, n_value_traps_blocked: int, top_picks: list[dict],
    output_dir: Path,
) -> None:
    """Append one CSV row per run for performance tracking over time.

    Lets you measure: HIGH gate fire rate, fallback usage, average score,
    sector mix, etc. Critical for evaluating whether the algo actually works.
    """
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / "run_log.csv"
        new_file = not log_path.exists()
        top_tickers = ",".join(b["ticker"] for b in top_picks[:5])
        top_scores = ",".join(f"{b['composite_score']:.3f}" for b in top_picks[:5])
        with open(log_path, "a", encoding="utf-8") as f:
            if new_file:
                f.write("date,regime,macro_grade,n_qualified,n_high,n_traps_blocked,"
                        "top5_tickers,top5_scores\n")
            f.write(f"{run_date},{regime},{macro_grade},{n_qualified},{n_high},"
                    f"{n_value_traps_blocked},{top_tickers},{top_scores}\n")
    except Exception as exc:
        logger.warning("Failed to log run results: %s", exc)


def run(cfg_path=None, debug=False, held_positions=None) -> None:
    """Run the full stock recommendation pipeline.

    Args:
        cfg_path: Optional path to YAML config (uses embedded default if None).
        debug: Quick test with 2 tickers when True.
        held_positions: Tickers you own — triggers sell alert analysis.
    """
    cfg = load_config(cfg_path)
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)

    universe = cfg.debug_universe if debug else cfg.universe
    start_date = cfg.debug_start_date if debug else cfg.start_date
    held = held_positions or []
    logger.info("Mode: %s | Universe: %s tickers", "DEBUG" if debug else "LIVE", len(universe))

    # Step 0 — Macro environment
    logger.info("Step 0: Fetching macro environment...")
    macro_env = fetch_macro_environment()
    logger.info("  Macro: %s", macro_env.get("macro_grade", "UNKNOWN"))

    # Step 1 — Market regime
    logger.info("Step 1: Detecting market regime...")
    regime_state = detect_regime(
        lookback_days=cfg.regime.lookback_days,
        vol_window=cfg.regime.vol_window,
        sma_window=cfg.regime.sma_window,
    )

    # Step 2 — Prices + indicators
    logger.info("Step 2: Downloading prices and computing indicators...")
    full_universe = list(set(universe) | set(held))
    price_data = download_prices(full_universe, start_date)
    all_raw_signals: dict[str, dict] = {}
    for tkr in [t for t in full_universe if t in price_data]:
        df = price_data[tkr]
        if len(df) < 200:
            continue
        all_raw_signals[tkr] = extract_latest_signals(compute_all_indicators(df, cfg))
    qualified = list(all_raw_signals.keys())
    logger.info("%d tickers qualified", len(qualified))

    # Step 2b — Reference prices
    logger.info("Step 2b: Reference prices...")
    reference_prices = download_reference_prices(start_date)

    # Step 2c — Price structure
    logger.info("Step 2c: Price structure signals...")
    all_ps = fetch_all_price_structure(qualified, price_data, cfg)

    # Step 3 — Fundamentals
    logger.info("Step 3: Fetching fundamentals...")
    all_fundamentals = fetch_all_fundamentals(qualified)

    # ── Data quality + liquidity filtering ─────────────────────────────
    # Drop stocks that yfinance returned thin/garbage data for, and drop
    # illiquid names where slippage would eat any edge.
    MIN_DOLLAR_VOLUME = 5_000_000  # $5M/day floor
    quality_filtered: list[str] = []
    rejected_data: list[str] = []
    rejected_liquidity: list[str] = []
    for tkr in qualified:
        fund = all_fundamentals.get(tkr, {})
        ok, n_present, missing = assess_data_quality(fund)
        if not ok:
            rejected_data.append(tkr)
            continue
        liq = compute_liquidity(price_data.get(tkr))
        if liq["avg_dollar_volume"] < MIN_DOLLAR_VOLUME:
            rejected_liquidity.append(tkr)
            continue
        # Stash liquidity onto the raw signals for later use
        all_raw_signals[tkr]["avg_dollar_volume"] = liq["avg_dollar_volume"]
        quality_filtered.append(tkr)
    logger.info(
        "Quality+liquidity filter: %d kept, %d rejected for thin data, %d for low liquidity",
        len(quality_filtered), len(rejected_data), len(rejected_liquidity),
    )
    if rejected_data:
        logger.debug("Thin-data rejected (sample): %s", rejected_data[:10])
    if rejected_liquidity:
        logger.debug("Low-liquidity rejected (sample): %s", rejected_liquidity[:10])
    # Use filtered list for everything downstream
    qualified = quality_filtered
    if not qualified:
        logger.error("No tickers passed data quality + liquidity filters.")
        return

    # Step 3b — Relative strength
    logger.info("Step 3b: Relative strength...")
    all_rs = fetch_all_relative_strength(
        qualified, all_fundamentals, price_data, reference_prices,
    )

    # Step 3c — Volume profile
    logger.info("Step 3c: Volume profile...")
    all_vp = fetch_all_volume_profile(qualified, price_data, all_fundamentals, cfg)

    # Step 4 — News
    logger.info("Step 4: News sentiment...")
    news_cfg = cfg.news
    all_news = fetch_and_score_all(
        qualified,
        max_age_hours=int(news_cfg.max_age_hours),
        recency_decay=float(news_cfg.recency_decay),
    )

    # Step 5 — Risk checks
    logger.info("Step 5: Risk checks...")
    all_risk = fetch_all_risk_checks(qualified, all_fundamentals)

    # Step 6 — Leadership
    logger.info("Step 6: Leadership analysis...")
    all_leadership = fetch_all_leadership(qualified)

    # Step 7 — Financial health
    logger.info("Step 7: Financial health scores...")
    all_health = fetch_all_health_scores(qualified, all_fundamentals)

    # Step 7b — Company analysis
    logger.info("Step 7b: Company analysis...")
    sector_medians = compute_sector_medians(all_fundamentals)
    all_company = fetch_all_company_analyses(qualified, all_fundamentals, sector_medians)

    # Step 8 — Score all tickers
    logger.info("Step 8: Scoring all tickers...")
    base_weights = config_to_weights_dict(cfg)
    adapted_weights = apply_regime_to_weights(base_weights, regime_state)
    min_conf = regime_state.confidence_override

    signal_matrix = build_signal_matrix(all_raw_signals, all_fundamentals, all_news)
    z_matrix = cross_sectional_zscore(signal_matrix)
    thresholds = vars(cfg.explanation_thresholds) if hasattr(cfg, "explanation_thresholds") else None

    score_bundles: dict[str, dict] = {}
    for tkr in qualified:
        z_row = z_matrix.loc[tkr].to_dict() if tkr in z_matrix.index else {}
        score_bundles[tkr] = score_asset(
            ticker=tkr, z_scores=z_row, raw_signals=all_raw_signals[tkr],
            raw_fundamentals=all_fundamentals[tkr], weights=adapted_weights,
            min_confidence_score=min_conf, news_sentiment=all_news.get(tkr),
            health_scores=all_health.get(tkr), risk_checks=all_risk.get(tkr),
            leadership=all_leadership.get(tkr), company_analysis=all_company.get(tkr),
            macro_data=macro_env, relative_strength=all_rs.get(tkr),
            price_structure=all_ps.get(tkr), volume_profile=all_vp.get(tkr),
        )

    # Sell alerts
    sell_alerts = generate_sell_alerts(
        held, score_bundles, regime_state,
        cfg.sell_signals if hasattr(cfg, "sell_signals") else None,
    )

    # Filter & rank
    ranked = sorted(
        [b for tkr, b in score_bundles.items() if tkr in set(universe)],
        key=lambda b: b["composite_score"], reverse=True,
    )
    high_conviction = [b for b in ranked if b["confidence"] == "HIGH"]

    # If no HIGH picks, fall back to top 20 quality candidates.
    # Same quality gates as HIGH confidence, just without the score threshold.
    n_value_traps_blocked = sum(1 for b in ranked if b.get("value_trap_blocks", False))
    if not high_conviction:
        logger.warning("No HIGH picks found (threshold=%.0f%%) — showing top 20 quality candidates", min_conf * 100)
        quality_passed = []
        for b in ranked:
            health = b.get("health_scores", {})
            altman_zone = health.get("altman", {}).get("zone", "UNKNOWN")
            bs = health.get("balance_sheet", {})
            bs_grade = bs.get("bs_grade", "UNKNOWN")
            profitable = bs.get("profitable", True)
            risk_level = b.get("risk_checks", {}).get("overall_risk", "UNKNOWN")
            risk_checks = b.get("risk_checks", {})
            # Apply ALL quality + value-trap gates (including earnings window)
            if (profitable
                    and altman_zone != "DISTRESS"
                    and bs_grade != "DANGEROUS"
                    and risk_level != "HIGH"
                    and not b.get("value_trap_blocks", False)
                    and not has_earnings_within(risk_checks, days=30)):
                quality_passed.append(b)
            if len(quality_passed) >= 30:  # collect more so we can diversify
                break
        high_conviction = quality_passed

    # ── Sector diversification: cap 3 per sector ──────────────────────
    n_high_pre_div = len(high_conviction)
    high_conviction = apply_sector_diversification(
        high_conviction, all_fundamentals, max_per_sector=3,
    )
    # Trim to top N after diversification
    high_conviction = high_conviction[:20]
    logger.info(
        "Picks: %d after sector diversification (was %d before, max 3/sector)",
        len(high_conviction), n_high_pre_div,
    )

    logger.info("Final scores (top 30):")
    for b in ranked[:30]:
        logger.info("  %s  %.3f  %s", b["ticker"], b["composite_score"], b["confidence"])

    # Explanations
    explanations = [
        generate_explanation(
            b["ticker"], b, sector_medians,
            thresholds=thresholds, regime_state=regime_state,
        )
        for b in high_conviction
    ]

    # Output dir — works on both Colab and local
    try:
        output_dir = Path(os.getcwd()) / "outputs" / "reports"
    except Exception:
        output_dir = Path("/tmp/stock_algo_reports")

    report = build_report(
        high_conviction, explanations, str(date.today()), output_dir,
        min_confidence_score=min_conf, sell_alerts=sell_alerts,
        regime_state=regime_state, macro_data=macro_env,
    )
    print("\n" + report)

    # ── Run logging — for tracking algo performance over time ─────────
    log_run_results(
        run_date=str(date.today()),
        regime=regime_state.regime.value,
        macro_grade=macro_env.get("macro_grade", "UNKNOWN") if macro_env else "UNKNOWN",
        n_qualified=len(qualified),
        n_high=sum(1 for b in ranked if b["confidence"] == "HIGH"),
        n_value_traps_blocked=n_value_traps_blocked,
        top_picks=high_conviction,
        output_dir=output_dir,
    )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Bulletproof stock recommendation engine")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--debug", action="store_true", help="Fast run on 2 tickers")
    parser.add_argument(
        "--held", nargs="+", default=[], metavar="TICKER",
        help="Currently held tickers — triggers sell alert analysis",
    )
    args = parser.parse_args()
    run(cfg_path=args.config, debug=args.debug, held_positions=args.held)


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    # Detect Colab / Jupyter: they pass kernel args that break argparse.
    # In that environment, users call run() directly — skip CLI parsing.
    import sys as _sys
    _in_notebook = any("jupyter" in a.lower() or "colab" in a.lower() or "ipykernel" in a.lower() for a in _sys.argv)
    if not _in_notebook:
        main()
