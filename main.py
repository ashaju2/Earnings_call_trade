from nltk.sentiment.vader import SentimentIntensityAnalyzer
import requests
import pandas as pd
from pandas import read_csv
import datetime
from yahoo_earnings_calendar import YahooEarningsCalendar
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
from pandas.tseries.offsets import BDay
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, nearest_workday, \
    USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay, \
    USLaborDay, USThanksgivingDay


class USTradingCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday('NewYearsDay', month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday('USIndependenceDay', month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday('Christmas', month=12, day=25, observance=nearest_workday)
    ]


def get_trading_close_holidays(year):
    inst = USTradingCalendar()
    return inst.holidays(datetime.datetime(year-1, 12, 31), datetime.datetime(year, 12, 31))


def get_earnings_call(yec):
    date_from = datetime.date.today()
    date_to = datetime.date.today()
    print(yec.earnings_on(date_from))
    print(yec.earnings_between(date_from, date_to))


def get_past_earnings_call(yec, company):
    df = pd.DataFrame(yec.get_earnings_of(company))
    return df

def aapl_earnings_call():
    v = "We achieved an all-time revenue record of $111.4 billion. " \
        "We saw strong double-digit growth across every product category, " \
        "and we achieved all-time revenue records in each of our geographic segments."
    return v

def puase_main():
    sia = SentimentIntensityAnalyzer()


def main():
    demo = 'd286f23fd3d3c4fbd6cc5768c2a6388d'

    #data = read_csv('/Users/alenshaju/Downloads/SP500_tickers_100.csv')
    #companies = data['Ticker'].to_list()[:10]

    consumer_companies = ['TJX', 'NKE', 'TGT', 'HD', 'LOW', 'PG', 'WMT', 'COST', 'MDLZ', 'EL', 'KO', 'PEP', 'PM', 'MO', 'BKNG', 'MCD', 'SBUX']
    energy_companies = ['NEE', 'XOM', 'CVX']
    fig_companies = ['BLK', 'AXP', 'V', 'MA', 'PYPL', 'FIS', 'JPM', 'BAC', 'WFC', 'USB', 'SPGI', 'MS', 'SCHW', 'GS', 'BRK.B', 'AMT'] #C
    healthcare_companies = ['ABBV', 'AMGN', 'GILD', 'ABT', 'DHR', 'MDT', 'SYK', 'ISRG', 'CVS', 'CI', 'TMO', 'UNH', 'ANTM', 'JNJ', 'PFE', 'LLY', 'BMY']
    industrials_companies = ['BA', 'RTX', 'LMT', 'DE', 'UPS', 'TSLA', 'GM', 'CAT', 'HON', 'GE', 'MMM', 'LIN', 'UNP']
    tech_companies = ['ADBE', 'CRM', 'INTU', 'GOOG', 'GOOG.L', 'FB', 'AMZN', 'ACN', 'IBM', 'AMAT', 'LRCX', 'NVDA', 'INTC', 'AVGO', 'TXN', 'QCOM', 'MU', 'AMD', 'MSFT', 'ORCL', 'NOW', 'AAPL']
    mt_companies = ['CMCS.A', 'CHTR', 'CSCO', 'VZ', 'T', 'DIS', 'NFLX']

    companies = ['UAL']

    past_call_dict = {}
    yec = YahooEarningsCalendar()

    for company in companies:
        print("Ticker:", company)
        past_calls_df = get_past_earnings_call(yec, company)
        past_call_dict[company] = past_calls_df
    df_returns_scores = pd.DataFrame(columns=['Return', 'Score'])
    sia = SentimentIntensityAnalyzer()

    d = {}
    with open("/Users/alenshaju/Downloads/LoughranMcDonald_MasterDictionary_2018.txt") as f:
        for line in f:
            (key, val) = line.split()
            d[key] = float(val)
    sia.lexicon.update(d)
    excel_df = pd.DataFrame(columns = ['Ticker', 'Quarter', 'Sentiment Score', 'Returns'])

    for company in companies:
        print("For company: ", company)
        for i, row in past_call_dict[company].iterrows():
            date = datetime.datetime.strptime(row['startdatetime'], '%Y-%m-%dT%H:%M:%S.%fZ')
            quarter = pd.Timestamp(date).quarter
            year = date.year
            if year <= datetime.datetime.now().year:
                if year == datetime.datetime.now().year:
                    if quarter >= pd.Timestamp(datetime.datetime.now()).quarter:
                        continue
                transcript = requests.get(
                f'https://financialmodelingprep.com/api/v3/earning_call_transcript/{company}?quarter={quarter}&year={year}&apikey={demo}').json()

                if len(transcript) == 0:
                    continue

                transcript = transcript[0]['content'].split('\n')
                if not bool(len(pd.bdate_range(date, date))):
                    date = date - BDay(1)
                if (date + BDay(1)) in get_trading_close_holidays(year):
                    end_date = date + BDay(1)
                else:
                    end_date = date

                stock = yf.download(company, start=date, end=end_date + BDay(1) + datetime.timedelta(1), progress=False)
                price_change_rate = (stock['Adj Close'][1] / stock['Adj Close'][0]) - 1
                price_change_percent = price_change_rate * 100
                sentiment_score = sia.polarity_scores(transcript[0])['pos'] - sia.polarity_scores(transcript[0])['neg']
                print(transcript)
                print('score: ', sia.polarity_scores(transcript[0]))
                print("price change: ", price_change_rate)

                df_returns_scores = df_returns_scores.append({'Return': price_change_rate,
                                            'Score': sentiment_score}, ignore_index=True)
                excel_df = excel_df.append({'Ticker': company, "Date": date, 'Quarter': quarter, 'Sentiment Score': sentiment_score,
                                 'Returns': price_change_rate}, ignore_index=True)
            if i > 8:  # 10years - 4 quarters
                break

    excel_df.to_excel("/Users/alenshaju/Downloads/mt_excel_file_v1.xlsx")

    x = df_returns_scores.Score.values.reshape(-1,1)
    y = df_returns_scores.Return.values.reshape(-1,1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)

    support_vector_reg_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
    support_vector_reg_model.fit(x_train, y_train)

    y_pred = support_vector_reg_model.predict(x_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2_data = r2_score(y_test, y_pred)
    print("Root mean square error: ", rmse)
    print("R^2 score: ", r2_data)

    train_test_label = ['Training Data', 'Testing Data']
    model_color = ['m', 'c', 'g']

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 10), sharey=True)

    ###### Training Data ##########
    axes[0].plot(x_test, y_pred, color=model_color[0], lw=2, label='{} model'.format(train_test_label[0]))
    axes[0].scatter(x_train[np.setdiff1d(np.arange(len(x_train)), support_vector_reg_model.support_)],
                         y_train[np.setdiff1d(np.arange(len(x_train)), support_vector_reg_model.support_)],
                         facecolor="none", edgecolor=model_color[0], s=50,
                         label='Training data')
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
                        ncol=1, fancybox=True, shadow=True)

    ####### Testing Data #########
    axes[1].plot(x_test, y_pred, color=model_color[1], lw=2, label='{} model'.format(train_test_label[1]))
    axes[1].scatter(x_test[np.setdiff1d(np.arange(len(x_test)), support_vector_reg_model.support_)],
                         y_pred[np.setdiff1d(np.arange(len(x_test)), support_vector_reg_model.support_)],
                         facecolor="none", edgecolor=model_color[1], s=50,
                         label='Testing data')
    axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
                        ncol=1, fancybox=True, shadow=True)
    fig.text(0.5, 0.04, 'data', ha='center', va='center')
    fig.text(0.06, 0.5, 'target', ha='center', va='center', rotation='vertical')
    fig.suptitle("Support Vector Regression", fontsize=14)
    plt.show()

if __name__ == '__main__':
    main()

