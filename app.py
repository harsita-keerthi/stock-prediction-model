# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# define starting and ending point of data
start = '2010-01-01'
end = '2019-12-31'

# scrape data from yahoo finance
df = yf.download('AAPL', start=start, end=end)
df.head()