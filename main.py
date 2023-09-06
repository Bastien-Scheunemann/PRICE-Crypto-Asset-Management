# -*- coding: utf-8 -*-


""" Import """
# !pip install ta
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from tabulate import tabulate
from scipy.stats import linregress
from sklearn.cluster import KMeans
import ta

""" Crypto liste to trade with """

crypto_liste = [['Bitcoin', 'BTC-USD'], ['Etherium', 'ETH-USD'], ['Chia', 'XCH-USD'],
                ['Cordano', 'ADA-USD'], ['Binance', 'BNB-USD'], ['Avalanche', 'AVAX-USD']]

end_date = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.today() - timedelta(days=60)).strftime('%Y-%m-%d')  # Start 60 days before today

for crypto in crypto_liste:
    data = yf.download(crypto[1], start=start_date, end=end_date, interval="1h")
    data = data.reset_index()
    crypto.append(data)

data = crypto_liste[0][2]
data = data['Close']

bitcoin_prices = []  # Get BTC price for the plot
index_liste = []

# Loop through the crypto list data to get the index for the plot
for index, row in crypto_liste[0][2].tail(720).iterrows():
    index_liste.append(index)
    bitcoin_prices.append(row['Close'])

"""Definition of the Asset CLass and the associated funtion"""


class Asset:
    """
    The Asset Class aims to define the crypto currency that the bot will trade and the associated functions
    """

    def __init__(self, name, symbol, quantity, price, datetime, index):
        """
        Initialization of the asset
        """
        self.name = name
        self.symbol = symbol
        self.quantity = quantity
        self.price = price
        self.datetime = datetime
        self.index = index

    def set_quantity(self, quantity):
        """
        Set the quantity of an asset and aslo helpful to modify it
        """
        self.quantity = quantity

    def volatility(self):
        """
        Return the volatility of the asset as the variance of the last hour asset prices
        """
        # Load data
        data = yf.download(self.symbol, interval="5m", period="1d")
        last_hour_data = data.tail(12)
        last_hour_returns = data['Close'].pct_change()

        # Volatility calculation
        volatility = last_hour_returns.std()
        return volatility

    def support_resistance(self, period, K=10):
        """
        Return the support and resistance levels for the traded asset
        """

        # K is the number of cluster
        for crypto in crypto_liste:
            if crypto[1] == self.symbol:
                data = crypto[2][self.index - period * 24: self.index]

        prices = np.array(data["Close"])
        kmeans = KMeans(n_clusters=K).fit(prices.reshape(-1, 1))
        clusters = kmeans.predict(prices.reshape(-1, 1))

        min_max_values = []

        for i in range(K):
            min_max_values.append([np.inf, -np.inf])

        for i in range(len(prices)):
            cluster = clusters[i]
            if prices[i] < min_max_values[cluster][0]:
                min_max_values[cluster][0] = prices[i]
            if prices[i] > min_max_values[cluster][1]:
                min_max_values[cluster][1] = prices[i]

        output = []
        s = sorted(min_max_values, key=lambda x: x[0])

        for i, (_min, _max) in enumerate(s):
            if i == 0:
                output.append(_min)
            if i == len(min_max_values) - 1:
                output.append(_max)
            else:
                output.append(sum([_max, s[i + 1][0]]) / 2)
        return output

    def PP_sup_res(self, period, K):
        """
        Return closest support and resistance of the last closed price, useful for the TP and Sl of the trading strategies
        """
        list_sup_res = self.support_resistance(period, K)
        price = self.price
        lower_bound, upper_bound = 0, 0

        # Find the lower and upper bounds for the input float within the list
        for i in range(len(list_sup_res)):

            if list_sup_res[i] <= price:
                lower_bound = list_sup_res[i]
            if list_sup_res[i] >= price:
                upper_bound = list_sup_res[i]
                break

        # Return the bounding numbers as a tuple
        return lower_bound, upper_bound

    def tendance_haussiere(self, period):  # period is a number of day
        """
        :param period: number of day
        :return: 1 the price are increasing 0 the price are not increasing
        """
        for crypto in crypto_liste:
            if crypto[1] == self.symbol:
                data = crypto[2][self.index - period * 24: self.index]
        dt = data['Close']
        t = [i for i in range(len(dt))]
        slope, intercept, r_value, p_value, std_err = linregress(t, dt)

        if slope > 0:
            return 1
        else:
            return 0

    def tendance_baissiere(self, period):  # period is a number of day
        """
        :param period: number of day
        :return: 1 the price are decreasing 0 the price are not decreasing
        """
        for crypto in crypto_liste:
            if crypto[1] == self.symbol:
                data = crypto[2][self.index - period * 24: self.index]

        dt = data['Close']
        t = [i for i in range(len(dt))]
        slope, intercept, r_value, p_value, std_err = linregress(t, dt)

        if slope < 0:
            return 1
        else:
            return 0

    def check_ma_50_under_200(self):
        """
        Check if the ma50 is below the ma 200 by returning 1 if it's true 0 otherwise
        """
        for crypto in crypto_liste:
            if crypto[1] == self.symbol:
                data = crypto[2][self.index - 40: self.index]

        # building up the indicator
        ma_50 = data['Close'].rolling(window=15).mean()
        ma_200 = data['Close'].rolling(window=30).mean()

        # add the column in the data frame
        data.loc[:, 'Signal'] = np.where(ma_50 > ma_200, 1, 0)

        # the signal is positif if there is an increasing crossing
        if data['Signal'].iloc[-1] > data['Signal'].iloc[-2]:
            return 1
        else:
            return 0

    def check_ma_50_above_200(self):

        for crypto in crypto_liste:
            if crypto[1] == self.symbol:
                data = crypto[2][self.index - 40: self.index]

        # building of the indicator
        ma_50 = data['Close'].rolling(window=15).mean()
        ma_200 = data['Close'].rolling(window=30).mean()

        # add the column in the data frame
        data.loc[:, 'Signal'] = np.where(ma_50 > ma_200, 1, 0)

        # the signal is positif if there is an increasing crossing
        if data['Signal'].iloc[-1] < data['Signal'].iloc[-2]:
            return 1
        else:
            return 0

    def check_ma_15_under_25(self):  # cet indicateur permet de déterminer une zone d'achat

        for crypto in crypto_liste:
            if crypto[1] == self.symbol:
                data = crypto[2][self.index - 30: self.index]

        # construction de l'indicateur
        me_15 = data['Close'].ewm(span=15).mean()
        me_25 = data['Close'].ewm(span=25).mean()

        # ajout de la colonne 'Signal' dans le DataFrame
        data.loc[:, 'Signal'] = np.where(me_15 > me_25, 1, 0)

        # le signal est positif s'il y a un croisement haussier
        if data['Signal'].iloc[-1] > data['Signal'].iloc[-2]:
            return 1
        else:
            return 0

    def check_ma_15_above_25(self):  # cet indicateur permet de déterminer une zone d'achat

        for crypto in crypto_liste:
            if crypto[1] == self.symbol:
                data = crypto[2][self.index - 30: self.index]

        # construction de l'indicateur
        me_15 = data['Close'].ewm(span=15).mean()
        me_25 = data['Close'].ewm(span=25).mean()

        # ajout de la colonne 'Signal' dans le DataFrame
        data.loc[:, 'Signal'] = np.where(me_15 > me_25, 1, 0)

        # le signal est positif s'il y a un croisement haussier
        if data['Signal'].iloc[-1] < data['Signal'].iloc[-2]:
            return 1
        else:
            return 0

    def check_stoch_indicator(self):  # indicator to get a buy signal

        # chargement des données
        for crypto in crypto_liste:

            if crypto[1] == self.symbol:
                data = crypto[2][self.index - 168: self.index]

        # construction de l'indicateur
        data.loc[:, '%K'] = (data['Close'] - data['Low']) / (data['High'] - data['Low'])
        data.loc[:, '%D'] = data['%K'].ewm(span=9).mean()

        # implémentation des signaux
        data.loc[:, 'Trading_Signal'] = np.where(data['%K'] > data['%D'], 1, 0)
        TS = np.array(data['Trading_Signal'])
        # le signal est positif s'il y a un croisement haussier
        if TS[-2] == 0 and TS[-1] == 1:
            return 1
        else:
            return 0

    # Define the RSI trading strategy function
    def rsi_strategy(self, rsi_oversold=30, rsi_overbought=70):

        for crypto in crypto_liste:

            if crypto[1] == self.symbol:
                stock_data = crypto[2][self.index - 30: self.index]

        # Calculate the RSI indicator using TA-Lib
        rsi_indicator = ta.momentum.RSIIndicator(stock_data['Close'], window=20)
        stock_data.loc[:, 'RSI'] = rsi_indicator.rsi()

        # Create a signal when the RSI crosses the oversold or overbought levels
        stock_data.loc[:, 'Signal'] = 0
        stock_data.loc[stock_data['RSI'] < rsi_oversold, 'Signal'] = 1
        stock_data.loc[stock_data['RSI'] > rsi_overbought, 'Signal'] = 0

        S = np.array(stock_data['Signal'])

        if S[-2] == 0 and S[-1] == 1:
            return 1
        else:
            return 0


"""Définition de la class Portfolio et des fonctions associées"""


class Portfolio:
    """
    Definition de la classe Portfolio qui a pour but de modelliser un
    portefeuille ainsi que les fonctions associées
    """

    def __init__(self, name, cash_init, index=0, assets=None, history=[]):
        self.name = name
        self.index = index
        self.cash = Asset('Cash', 'USD', cash_init, 1, '01-01-2023', self.index)
        self.assets = assets or []
        self.history = history
        self.asset_price_init = {}  # Exemple : {
        #                'BTC': {'name': 'Bitcoin', 'price': 27000},
        #                'ETH': {'name': 'Ethereum', 'price': 2000},
        #                'LTC': {'name': 'Litecoin', 'price': 150},
        #                'XRP': {'name': 'Ripple', 'price': 0.5}
        #            }
        # permet de concerver le prix d'achat
        # util pour avoir des rapports de performance

    def add_asset(self, asset, order_time):  # cette fonction ajoute une crypto au portefeuille

        cout = asset.quantity * asset.price
        c = self.cash.quantity

        done = False  # booléen qui permet de savoir si on a déjà la crypto que l'on veut ajouter dans le portfeuille

        for a in self.assets:

            if a.name == asset.name and c > cout:  # condition d'achat

                Qi = a.quantity
                Qf = asset.quantity
                # il faut définir le nouveau prix initial comme moyenne entre le prix initial et le nouveaux prix
                # d'achat
                new_price_init = self.asset_price_init[asset.symbol]['price'] * Qi / (Qi + Qf) + \
                                 asset.price * Qf / (Qi + Qf)
                # mise à jour de la quantité de l'asset
                a.set_quantity(Qi + Qf)
                # ajout à l'historique des trades
                self.history.append(['achat', asset, order_time])
                # mise à jour du cash
                self.cash.quantity = c - cout
                # mise à jour du prix d'achat
                self.asset_price_init[asset.symbol] = {'name': asset.name, 'price': new_price_init}
                done = True

        if c > cout and not done:  # si on a pas déjà acheté l'actif

            self.history.append(['achat', asset.symbol, order_time])
            self.assets.append(asset)
            self.cash.quantity = c - cout
            if self.assets == []:
                self.asset_price_init = {asset.symbol: {'name': asset.name, 'price': asset.price}}
            else:
                self.asset_price_init[asset.symbol] = {'name': asset.name, 'price': asset.price}

    def remove_asset(self, asset, order_time):
        # fonction qui retire un actif du portefeuille et met à jour l'historique des transactions
        gain = asset.quantity * asset.price
        c = self.cash.quantity
        self.cash.quantity = c + gain
        self.assets.remove(asset)
        self.history.append(['vente', asset, order_time])
        del self.asset_price_init[asset.symbol]

    def total_value(self):
        return sum(asset.quantity * asset.price for asset in self.assets) + self.cash.quantity * self.cash.price

    def tp_sl(self, asset, order_time):  # Fonction take profit et stop loss
        ##Take profit sans conditions de support/résistance pour l'instant (version naif)
        current_price = asset.price
        price_init = self.asset_price_init[asset.symbol]['price']

        # implémentation de la stratégie de TP et SL, arbitraire ici
        # if current_price > 1.15 * price_init:
        # self.remove_asset(asset, order_time)
        CI = (asset.check_ma_50_above_200() + asset.tendance_baissiere(20) + asset.check_ma_15_above_25()) / 3

        if current_price > 1.03 * price_init:
            #  asset1 = Asset(asset.name, asset.symbol, asset.quantity / 2, asset.price,
            #                asset.datetime, self.index)  # On divie la quantité par deux
            self.remove_asset(asset, order_time)
        # self.add_asset(asset1, order_time)

        ##Stop loss
        elif current_price < 0.985 * price_init:
            # asset1 = Asset(asset.name, asset.symbol, asset.quantity / 2, asset.price, asset.datetime, self.index)
            self.remove_asset(asset, order_time)
        # self.add_asset(asset1, order_time)

        # elif CI >= 0.5:
        # self.remove_asset(asset, order_time)

    # elif current_price < 0.90 * price_init:
    #    self.remove_asset(asset, order_time)

    def update(self):

        # Met à jour chaque asset du portfeuille au prix du marché
        for asset in self.assets:

            if asset.name != "Cash":  # On actualise pas le prix du dollars
                ticker = asset.symbol

                for crypto in crypto_liste:

                    if crypto[1] == ticker:
                        data = crypto[2]

                # Extraction du dernier prix de cloture en hourly
                last_closed_price = data["Close"][self.index]

                # Mise à jour du prix de l'actif
                asset.price = last_closed_price

    def risk(self, n):
        # Première étape : récupérer les données des actifs du portefeuille
        nb_asset = len(self.assets)
        prices = []

        if nb_asset == 0:
            return 0

        for asset in self.assets:

            for crypto in crypto_liste:

                if crypto[1] == asset.symbol:
                    data = crypto[2]
                    data = data['Close'][self.index - n:self.index]
                    prices.append(data.tolist())

        S = np.array(prices)
        # Deuxième étape : calculer la moyenne de chaque stocks
        M = np.mean(S, axis=1)

        # Troisème étape on retire la moyenne au dernier prix
        S_demeaned = S - M[:, np.newaxis]

        # Quatrième étape on calcul la covariance
        # A_pinv = np.linalg.pinv(S_demeaned)
        C = np.cov(S_demeaned)

        # Cinquième étape récupérer les pondérations des actifs dans le portefeuille
        W = np.ones(nb_asset)
        count = 0
        tot = self.total_value()
        for asset in self.assets:
            W[count] = asset.price * asset.quantity / tot
            count += 1

        # Calculate the expected portfolio return
        portfolio_return = np.dot(M, W)

        # Calculate the expected portfolio variance
        portfolio_variance = np.dot(np.dot(W.T, C), W)

        # Print the risk of the portfolio
        portfolio_risk = np.sqrt(portfolio_variance)

        return portfolio_risk

    def print_performance_1(self, time):

        print(f"Portfolio at time {time}:")
        headers = ["Name", "Symbol", "Volatility", "Quantity", "Price", "% of Pf"]

        data = []
        # Print the table using tabulate
        for asset in self.assets:
            data.append([asset.name, asset.symbol, asset.volatility(), asset.quantity, asset.price,
                         asset.quantity * asset.price / self.total_value() * 100])
        data.append([self.cash.name, self.cash.symbol, 'None', self.cash.quantity, self.cash.price,
                     self.cash.quantity / self.total_value() * 100])
        print(tabulate(data, headers=headers))
        print(f"Total value: {self.total_value()}\n")
        print(f"Risk value: {self.risk(10)}\n")

    def performance(self):

        headers = ["Name", "Symbol", "Volatility", "Quantity", "Price", "% of Pf"]
        data = []
        for asset in self.assets:
            data.append([asset.name, asset.symbol, asset.volatility(), asset.quantity, asset.price,
                         asset.quantity * asset.price / self.total_value() * 100])
        data.append([self.cash.name, self.cash.symbol, 'None', self.cash.quantity, self.cash.price,
                     self.cash.quantity / self.total_value() * 100])
        value_and_risk = [self.total_value(), self.risk(10)]

        return data, value_and_risk

    def performance2(self):
        data = {}
        name_asset_list = [['Bitcoin', 'BTC-USD'], ['Ethereum', 'ETH-USD'], ['Chia', 'XCH-USD'], ['Cordano', 'ADA-USD'], \
                           ['Binance', 'BNB-USD'], ['Avalanche', 'AVAX-USD']]
        for asset in self.assets:

            for a in name_asset_list:

                if a[1] == asset.symbol:
                    data[asset.symbol] = {"Name": asset.name, "Symbol": asset.symbol, "Volatility": asset.volatility(),
                                          "Quantity": asset.quantity, "Price": asset.price,
                                          "pPf": asset.quantity * asset.price / self.total_value() * 100}

                    name_asset_list.remove(a)

        for a in name_asset_list:
            data[a[1]] = {"Name": a[0], "Symbol": a[1], "Volatility": 0,
                          "Quantity": 0, "Price": 0,
                          "pPf": 0}

        data['cash'] = {"Name": self.cash.name, "Symbol": self.cash.symbol, "Volatility": 'None',
                        "Quantity": self.cash.quantity, "Price": self.cash.price,
                        "pPf": self.cash.quantity / self.total_value() * 100}

        return data


def taille_position(Entry_price, Stop_loss_Price, Account_risk, Account_size):
    SL_share = Entry_price - Stop_loss_Price
    Number_of_shares = (Account_risk * Account_size) / SL_share

    return Number_of_shares


"""Mise en service automatique de l'algorithme"""


def app(Pf):
    RR_ratio_best = 1.8
    asset_size_max = 0.15  # no more than 20% of one same asset
    risk_value = 0.02
    name_asset_list = [['Bitcoin', 'BTC-USD'], ['Etherium', 'ETH-USD'], ['Chia', 'XCH-USD'], ['Cordano', 'ADA-USD'], \
                       ['Binance', 'BNB-USD'], ['Avalanche', 'AVAX-USD']]

    date = index
    order_time = 1
    period = 5
    K = 6
    liste_indicateur_achat = []

    for a in name_asset_list:

        is_in = False

        for asset in Pf.assets:

            if asset.symbol == a[1]:
                # on veut une condition de sortie du trade
                is_in = True
                Pf.tp_sl(asset, order_time)

        if is_in == False:

            # on veut une condition d'entrée du trade avec SL, TP et taille de position
            ticker = a[1]
            data = []

            for crypto in crypto_liste:

                if crypto[1] == ticker:
                    data = crypto[2]

            # Extraction du dernier prix de cloture en hourly
            last_closed_price = data["Close"][Pf.index]

            # creation d'un nouvel asset
            new_asset = Asset(a[0], a[1], 0, last_closed_price, date, Pf.index)

            # obtention des supports et resistances
            sup, res = new_asset.PP_sup_res(period, K)
            # Return/risk ratio

            RR_ratio = (res - new_asset.price) / (new_asset.price - sup)  # 1.6 best number

            # calcul de la quantite a acheter en fonction du risque du portefeuille
            qty_asset = taille_position(new_asset.price, sup, res, risk_value, Pf.total_value())

            qty_max = min(asset_size_max * Pf.total_value() / new_asset.price, qty_asset)
            # if a[1] == 'BTC-USD' or a[1] == 'ETH-USD' :

            #   qty_max = 0.4 * Pf.total_value() / new_asset.price
            # qty = min(qty_asset, qty_max)

            new_asset.set_quantity(qty_max)  # 20% du portefeuille

            CI_achat = (new_asset.check_ma_50_under_200() + new_asset.check_stoch_indicator() + \
                        new_asset.tendance_haussiere(
                            20) + new_asset.rsi_strategy() + new_asset.check_ma_15_under_25()) / 5

            # CI_achat peut être interprété comme la probabilité que le stock monte et atteigne la première résistance
            expected_return = CI_achat * RR_ratio

            # print("je fais l'asset :" + new_asset.name)
            # print(expected_return, RR_ratio, CI_achat)

            # condition de rentabilité et de probabilité d'occurence suffisante
            if 2.2 >= RR_ratio >= RR_ratio_best and CI_achat >= 0.4:
                liste_indicateur_achat.append([expected_return, new_asset])

    # on tri de manière décroissante pour optimizer la performance
    sorted_list = sorted(liste_indicateur_achat, reverse=True)

    for a in sorted_list:
        # on achète
        Pf.add_asset(a[1], order_time)


Pf = Portfolio("My Portfolio", 1000000, index=0)

data_backtest = []
result_pf = []
risk_pf = []

"""for index in index_liste:

    print(index)
    print(Pf.total_value())
    print(Pf.print_performance_1())
    Pf.index = index
    Pf.update()
    app(Pf)
    d, vr = Pf.performance()
    data_backtest.append(d)
    result_pf.append(vr[0])
    risk_pf.append(vr[1])
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, (ax1, ax2, ax4, ax5) = plt.subplots(nrows=4, ncols=1, figsize=(12, 8))

portfolio_value = []
portfolio_value_dollar = []

bitcoin_shares = []
etherium_shares = []
chia_shares = []
cordano_shares = []
binance_shares = []
avalanche_shares = []
risk_values = []

index_values = []

bitcoin_prices2 = []

line1, = ax1.plot([], [], label='Portfolio Evolution (%)')
line2, = ax2.plot([], [], label='Bitcoin Shares')
line3, = ax2.plot([], [], label='Ethereum Shares')
line4, = ax2.plot([], [], label='Chia Shares')
line5, = ax2.plot([], [], label='Cordano Shares')
line6, = ax2.plot([], [], label='Binance Shares')
line7, = ax2.plot([], [], label='Avalanche Shares')
line8, = ax1.plot([], [], label='Bitcoin prices Evolution (%)')
line9, = ax4.plot([], [], label='Risk')
line10, = ax5.plot([], [], label='Prix du Pf en dollar')

ax1.set_xlabel('Index')
ax1.set_ylabel('Portfolio Value')
ax1.set_title('Portfolio Value Over Time')
ax1.legend()

ax2.set_xlabel('Index')
ax2.set_ylabel('Shares')
ax2.set_title('% Shares in the Pf Over Time')
ax2.legend()

# ax3.set_xlabel('Index')
# ax3.set_ylabel('Bitcoin price')
# ax3.set_title('Evolution of the Bitcoin')
# ax3.legend()

ax4.set_xlabel('Index')
ax4.set_ylabel('risk')
ax4.set_title('Evolution of the risk')
ax4.legend()

ax5.set_xlabel('Index')
ax5.set_ylabel('Pf $')
ax5.set_title('Evolution in $')
ax5.legend()


def animate(i):
    if i == 0:

        portfolio_value.append(0)
        bitcoin_prices2.append(0)
        index = index_liste[i]
        bitcoin_p = bitcoin_prices[i]
        Pf.index = index
        Pf.update()
        app(Pf)
        perf = Pf.performance2()
        risk_values.append(Pf.risk(10))
        bitcoin_shares.append(perf['BTC-USD']['pPf'])
        etherium_shares.append(perf['ETH-USD']['pPf'])
        chia_shares.append(perf['XCH-USD']['pPf'])
        cordano_shares.append(perf['ADA-USD']['pPf'])
        binance_shares.append(perf['BNB-USD']['pPf'])
        avalanche_shares.append(perf['AVAX-USD']['pPf'])
        portfolio_value_dollar.append(Pf.total_value())

        index_values.append(index)

    else:

        index = index_liste[i]
        bitcoin_p = bitcoin_prices[i]
        Pf.index = index
        Pf.update()
        app(Pf)
        perf = Pf.performance2()
        portfolio_value.append((Pf.total_value() - portfolio_value_dollar[i - 1]) / portfolio_value_dollar[i - 1])
        risk_values.append(Pf.risk(10))
        bitcoin_shares.append(perf['BTC-USD']['pPf'])
        etherium_shares.append(perf['ETH-USD']['pPf'])
        chia_shares.append(perf['XCH-USD']['pPf'])
        cordano_shares.append(perf['ADA-USD']['pPf'])
        binance_shares.append(perf['BNB-USD']['pPf'])
        avalanche_shares.append(perf['AVAX-USD']['pPf'])
        bitcoin_prices2.append((bitcoin_p - bitcoin_prices[i - 1]) / bitcoin_prices[i - 1])
        portfolio_value_dollar.append(Pf.total_value())

        index_values.append(index)

    line1.set_data(index_values, portfolio_value)
    # ax1.set_ylim(980000, 1010000)
    ax1.relim()
    ax1.autoscale_view()

    line2.set_data(index_values, bitcoin_shares)
    line3.set_data(index_values, etherium_shares)
    line4.set_data(index_values, chia_shares)
    line5.set_data(index_values, cordano_shares)
    line6.set_data(index_values, binance_shares)
    line7.set_data(index_values, avalanche_shares)
    line8.set_data(index_values, bitcoin_prices2)
    line9.set_data(index_values, risk_values)
    line10.set_data(index_values, portfolio_value_dollar)

    ax2.relim()
    ax2.autoscale_view()

    # ax3.relim()
    # ax3.autoscale_view()

    ax4.relim()
    ax4.autoscale_view()

    ax5.relim()
    ax5.autoscale_view()

    return line1, line2, line3, line4, line5, line6, line7, line8, line9, line10


ani = animation.FuncAnimation(fig, animate, frames=len(index_liste), interval=10)
plt.show()

# """# Exemple d'usage pour print_performance_1
"""Pf = Portfolio("My Portfolio", 10000)
date = datetime.today().strftime('%Y-%m-%d')

A1 = Asset("Bitcoin", "BTC-USD", 100, 10, date)
A2 = Asset("Ethereum", "ETH-USD", 50, 20, date)
A3 = Asset("Bitcoin", "BTC-USD", 10, 12, date)
A4 = Asset("Ethereum", "ETH-USD", 30, 18, date)

print("Les prix initiaux :", Pf.asset_price_init)
print('ajout des deux premiers assets:')
Pf.add_asset(A1, 1)
Pf.add_asset(A2, 1)
print(Pf.asset_price_init)
res = A1.support_resistance(20, K=7)

Pf.add_asset(A3, 1)
Pf.add_asset(A4, 1)

print(Pf.asset_price_init)


Pf.update()

Pf.print_performance_1(1)

print(A1.check_ma_50_under_200())
print(A2.check_stoch_indicator())
print(A1.tendance_haussiere(200))
print(A2.rsi_strategy())

Pf.tp_sl(A1, 1)
Pf.print_performance_1(1)
print("Pf total cash", Pf.cash.quantity)
print(Pf.risk(10))
"""
