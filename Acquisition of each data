#必要なライブラリのインポート
from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.simplefilter('ignore')

!pip install yfinance
import yfinance as yf

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# 各データの取得

# 銘柄と日付範囲を指定します
tickers = ["BBDC4.SA", "BRAP4.SA", "CMIG4.SA", "GGBR4.SA", "GOAU4.SA", "GOLL4.SA", "ITSA4.SA", "ITUB4.SA"]
start_date = "2008-04-01"
end_date = "2008-05-02"

# 各銘柄に対してデータを取得し、それぞれ別のデータフレームに保存します
for ticker in tickers:
    globals()[f'{ticker.split(".")[0]}_df'] = yf.download(ticker, start=start_date, end=end_date)

# 取得したデータを確認します
for ticker in tickers:
    print(f"\n{ticker}")
    df_name = f'{ticker.split(".")[0]}_df'
    print(globals()[df_name].head())

BBDC4_df

# Volume列を削除します
BBDC4_df = BBDC4_df.drop(columns=['Volume'])

# 新たな目的変数Targetを作成します。Targetは1日後のAdj Close価格です
BBDC4_df['Target'] = BBDC4_df['Adj Close'].shift(-1)
BBDC4_df = BBDC4_df.drop(pd.Timestamp('2008-05-02'))

# 指定されたパスにBBDC4_dfをCSVとして保存します
file_path = 'YOURPATH'
BBDC4_df.to_csv(file_path)

BBDC4_df = pd.read_csv(file_path, index_col=0)

# データフレームの最初の数行を表示して確認します
BBDC4_df

# Volume列を削除します
BRAP4_df = BRAP4_df.drop(columns=['Adj Close'])

# 新たな目的変数Targetを作成します。Targetは1日後のAdj Close価格です
BRAP4_df['Target'] = BRAP4_df['Close'].shift(-1)
BRAP4_df = BRAP4_df.drop(pd.Timestamp('2008-05-02'))

# 指定されたパスにBRAP4_dfをCSVとして保存します
file_path = 'YOURPATH'
BRAP4_df.to_csv(file_path)

BRAP4_df = pd.read_csv(file_path, index_col=0)

# データフレームの最初の数行を表示して確認します
BRAP4_df

# Volume列を削除します
CMIG4_df = CMIG4_df.drop(columns=['Adj Close'])

# 新たな目的変数Targetを作成します。Targetは1日後のAdj Close価格です
CMIG4_df['Target'] = CMIG4_df['Close'].shift(-1)
CMIG4_df = CMIG4_df.drop(pd.Timestamp('2008-05-02'))

# 指定されたパスにCMIG4_dfをCSVとして保存します
file_path = 'YOURPATH'
CMIG4_df.to_csv(file_path)

CMIG4_df = pd.read_csv(file_path, index_col=0)

# データフレームの最初の数行を表示して確認します
CMIG4_df

# Volume列を削除します
GOAU4_df = GOAU4_df.drop(columns=['Adj Close'])

# 新たな目的変数Targetを作成します。Targetは1日後のAdj Close価格です
GOAU4_df['Target'] = GOAU4_df['Close'].shift(-1)
GOAU4_df = GOAU4_df.drop(pd.Timestamp('2008-05-02'))

# 指定されたパスにGOAU4_dfをCSVとして保存します
file_path = 'YOURPATH'
GOAU4_df.to_csv(file_path)

GOAU4_df = pd.read_csv(file_path, index_col=0)

# データフレームの最初の数行を表示して確認します
GOAU4_df

# Volume列を削除します
GOLL4_df = GOLL4_df.drop(columns=['Close'])

# 新たな目的変数Targetを作成します。Targetは1日後のAdj Close価格です
GOLL4_df['Target'] = GOLL4_df['Adj Close'].shift(-1)
GOLL4_df = GOLL4_df.drop(pd.Timestamp('2008-05-02'))

# 指定されたパスにGOLL4_dfをCSVとして保存します
file_path = 'YOURPATH'
GOLL4_df.to_csv(file_path)

GOLL4_df = pd.read_csv(file_path, index_col=0)

# データフレームの最初の数行を表示して確認します
GOLL4_df

# Volume列を削除します
ITSA4_df = ITSA4_df.drop(columns=['Adj Close'])

# 新たな目的変数Targetを作成します。Targetは1日後のAdj Close価格です
ITSA4_df['Target'] = ITSA4_df['Close'].shift(-1)
ITSA4_df = ITSA4_df.drop(pd.Timestamp('2008-05-02'))

# 指定されたパスにITSA4_dfをCSVとして保存します
file_path = 'YOURPATH'
ITSA4_df.to_csv(file_path)

ITSA4_df = pd.read_csv(file_path, index_col=0)

# データフレームの最初の数行を表示して確認します
ITSA4_df

# Volume列を削除します
ITUB4_df = ITUB4_df.drop(columns=['Adj Close'])

# 新たな目的変数Targetを作成します。Targetは1日後のAdj Close価格です
ITUB4_df['Target'] = ITUB4_df['Close'].shift(-1)
ITUB4_df = ITUB4_df.drop(pd.Timestamp('2008-05-02'))

# 指定されたパスにITUB4_dfをCSVとして保存します
file_path = 'YOURPATH'
ITUB4_df.to_csv(file_path)

ITUB4_df = pd.read_csv(file_path, index_col=0)

# データフレームの最初の数行を表示して確認します
ITUB4_df
