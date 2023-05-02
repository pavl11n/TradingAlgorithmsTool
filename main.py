import pathlib
from tkinter import *
from tkinter import ttk, messagebox

from PIL import ImageTk, Image

import gym
import gym_anytrading
import os
import numpy as np
import pandas as pd
import quantstats as qs

from gym_anytrading.envs import StocksEnv
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3 import A2C, PPO, DQN, DDPG
from datetime import datetime
from matplotlib import pyplot as plt
from finta import TA

tickers = ["AAPL", "AMD", "FSLR", "INTC", "MSFT",
           "NFLX", "NVDA", "QQQ", "SPY", "TSLA"]

values = ["None", "None"]


def signals(env):
    start = env.frame_bound[0] - env.window_size
    end = env.frame_bound[1]
    prices = env.df.loc[:, 'Close'].to_numpy()[start:end]
    listt = ["Close", "Volume"]
    if not (values[0] == "None" and values[1] == "None"):
        if values[0] == "All":
            listt = ["Close", "Volume", "RSI", "SMA", "EMA", "WMA", "SMMA"]
        elif values[0] != "None":
            listt = ["Close", "Volume", values[0]]
        if values[1] != "None" and values[0] == "All":
            listt = ["Close", "Volume", "RSI", "SMA", "EMA", "WMA", "SMMA", values[1] + " Close"]
        elif values[1] != "None":
            listt = ["Close", "Volume", values[1] + " Close"]
    print(listt)
    signal_features = env.df.loc[:, listt].to_numpy()[start:end]
    return prices, signal_features


class Tool:

    def __init__(self, ticker, algorithm, r1, r2, r3, r4, timesteps):
        self.ticker = ticker
        self.algorithm = algorithm
        self.r1 = int(r1)
        self.r2 = int(r2)
        self.r3 = int(r3)
        self.r4 = int(r4)

        print("Ticker: " + self.ticker)
        print("Algorithm: " + self.algorithm)
        print("From: " + r1 + " To: " + r2)
        print("From2: " + r3 + " To2: " + r4)

        self.windowSize = 10

        self.data = pd.read_csv("./resources/" + self.ticker + ".csv")

        indicator = values[0]
        if indicator != "None":
            if indicator == "RSI" or indicator == "All":
                self.data['RSI'] = TA.RSI(self.data, 16)
            if indicator == "SMA" or indicator == "All":
                self.data['SMA'] = TA.SMA(self.data, 16)
            if indicator == "EMA" or indicator == "All":
                self.data['EMA'] = TA.EMA(self.data, 16)
            if indicator == "WMA" or indicator == "All":
                self.data['WMA'] = TA.WMA(self.data, 16)
            if indicator == "SMMA" or indicator == "All":
                self.data['SMMA'] = TA.SMMA(self.data, 16)
            self.data.fillna(0, inplace=True)

        ticker2 = values[1]
        if ticker2 != "None":
            data_1 = pd.read_csv("./resources/" + ticker2 + ".csv")

            data_1.rename(columns={'Close': ticker2 + " Close"}, inplace=True)
            data_1.rename(columns={'Volume': ticker2 + " Volume"}, inplace=True)
            self.data = pd.concat([self.data, data_1], axis=1)
            self.data = self.data.loc[:, ~self.data.columns.duplicated()]

        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data.set_index('Date', inplace=True)

        self.env2 = MyCustomEnv(df=self.data, window_size=self.windowSize, frame_bound=(self.r1, self.r2))

        env_maker = lambda: self.env2
        self.env = DummyVecEnv([env_maker])

        self.model = PPO('MlpPolicy', self.env, verbose=1)
        self.model.learn(total_timesteps=int(timesteps))

    def getProfit(self):
        return self.profit

    def plot(self, r3, r4):
        self.r3 = r3
        self.r4 = r4
        self.env = MyCustomEnv(df=self.data, window_size=self.windowSize, frame_bound=(self.r3, self.r4))
        obs = self.env.reset()

        while True:
            action, _states = self.model.predict(obs)
            obs, rewards, done, info = self.env.step(action)
            if done:
                print('info', info)
                self.profit = info.get('total_profit')
                break

        plt.figure(figsize=(15, 6), facecolor='w')
        plt.cla()
        self.env.render_all()
        plt.title("Agent's peformance", fontdict={'family': 'arial', 'size': 20})
        plt.ylabel('Stock Price', fontsize=18)
        plt.xlabel('Number of Trading Days', fontsize=16)
        plt.savefig("temp.jpg")
        image = Image.open("./temp.jpg")
        photoImage = ImageTk.PhotoImage(image)
        return photoImage

    def getPercentage(self):
        if self.profit < 1:
            return int((1 - self.profit) * 100), False
        else:
            return int((int(self.profit) - 1) * 100 + (self.profit % 1) * 100), True

    def getBounds(self):
        self.env = MyCustomEnv(df=self.data, window_size=self.windowSize, frame_bound=(self.r3, self.r4))
        self.bound1 = self.env.df.index[self.env.frame_bound[0]].strftime('%Y-%m-%d')
        self.bound2 = self.env.df.index[self.env.frame_bound[1]].strftime('%Y-%m-%d')
        print(
            f"Frame bounds: {self.env.df.index[self.env.frame_bound[0]]} - {self.env.df.index[self.env.frame_bound[1]]}")

    def isValidDate(self, date):
        try:
            date = pd.to_datetime(date)
        except Exception as e:
            return False
        if date < self.env.df.index[self.env.frame_bound[0]] or date > self.env.df.index[self.env.frame_bound[1]]:
            return False
        else:
            self.date = date
            return True

    def predict_action_on_date(self):
        date_idx = self.env.df.index.get_loc(self.date, method='nearest')

        listt = ["Close", "Volume"]
        if not (values[0] == "None" and values[1] == "None"):
            if values[0] == "All":
                listt = ["Close", "Volume", "RSI", "SMA", "EMA", "WMA", "SMMA"]
            elif values[0] != "None":
                listt = ["Close", "Volume", values[0]]
            if values[1] != "None" and values[0] == "All":
                listt = ["Close", "Volume", "RSI", "SMA", "EMA", "WMA", "SMMA", values[1] + " Close"]
            elif values[1] != "None":
                listt = ["Close", "Volume", values[1] + " Close"]
        obs = self.env.df.loc[:, listt].to_numpy()[date_idx - self.env.window_size + 1:date_idx + 1]

        # Add the batch dimension
        obs = obs.reshape((1,) + obs.shape)

        action, _states = self.model.predict(obs, deterministic=True)
        if action == 0:
            print(f"On {self.date}, the agent would Sell.")
        else:
            print(f"On {self.date}, the agent would Buy.")
        return action

class MyCustomEnv(StocksEnv):
    _process_data = signals


class ToolGUI(Frame):

    # Constructor of the class :
    def __init__(self, **kw):
        # Givinf the title and size of the frame :
        super().__init__(**kw)
        root.title("Trading Algorithms Tool")
        root.geometry('1300x700')

        canvasVerticalLine = Canvas(root, width=1, height=700)
        canvasVerticalLine.place(x=275, y=0)

        self.bgColor = "#d2dada"
        root.config(bg=self.bgColor)

        # Creating and placing labels, buttons and entries :

        self.lblInfo = Label(root, text="Please enter data on the left", font=('Times New Roman', 25, 'bold'))
        self.lblInfo.config(bg=self.bgColor)
        self.lblInfo.place(x=600, y=300)

        self.lblInput = Label(root, text="Input test data", font=('Times New Roman', 25, 'bold'))
        self.lblInput.config(bg=self.bgColor)
        self.lblInput.place(x=35, y=15)

        startY = 80
        self.font = ('Times New Roman', 20, 'bold')
        self.fontSmaller = ('Times New Roman', 20)
        self.fontSmall = ('Times New Roman', 17)

        self.lblMax = Label(root, text="", font=self.fontSmall)
        self.lblMax.config(bg=self.bgColor)
        self.lblMax.place(x=170, y=startY + 352)

        self.lblMax2 = Label(root, text="", font=self.fontSmall)
        self.lblMax2.config(bg=self.bgColor)
        self.lblMax2.place(x=170, y=startY + 482)

        # Ticker:
        self.lblTicker = Label(root, text="Ticker:", font=self.font, borderwidth=2, relief="groove")
        self.lblTicker.config(bg=self.bgColor)
        self.lblTicker.place(x=85, y=startY)
        self.tickerChoices = tickers
        self.tickerChoiceVar = StringVar()
        self.tickerChoiceVar.trace("w", self.callback)
        self.tickerChoiceVar.set('AAPL')
        self.tickerOptionMenu = OptionMenu(root, self.tickerChoiceVar, *self.tickerChoices)
        self.tickerOptionMenu.config(bg=self.bgColor, font=self.fontSmaller)
        self.tickerOptionMenu.place(x=78, y=startY + 45)

        # Algorithm:
        self.lblAlgorithm = Label(root, text="Choose your algorithm:", font=self.font, borderwidth=2, relief="groove")
        self.lblAlgorithm.config(bg=self.bgColor)
        self.lblAlgorithm.place(x=35, y=startY + 90)
        self.algorithmChoices = ['PPO', 'A2C', 'DQN']
        self.algorithmChoiceVar = StringVar()
        self.algorithmChoiceVar.set('PPO')
        self.algorithmOptionMenu = OptionMenu(root, self.algorithmChoiceVar, *self.algorithmChoices)
        self.algorithmOptionMenu.config(bg=self.bgColor, font=self.fontSmaller)
        self.algorithmOptionMenu.place(x=92, y=startY + 135)

        # Timeseps
        self.lblTimeseps = Label(root, text="Timesteps:", font=self.font, borderwidth=2, relief="groove")
        self.lblTimeseps.config(bg=self.bgColor)
        self.lblTimeseps.place(x=75, y=startY + 180)
        self.txtTimesteps = Entry(root, width=7)
        self.txtTimesteps.insert(0, "1000")
        self.txtTimesteps.config(highlightbackground=self.bgColor)
        self.txtTimesteps.place(x=60, y=startY + 225)
        self.lblMin3 = Label(root, text="min: 1000", font=self.fontSmall)
        self.lblMin3.config(bg=self.bgColor)
        self.lblMin3.place(x=140, y=startY + 215)
        self.lblMax3 = Label(root, text="max: 1 mil", font=self.fontSmall)
        self.lblMax3.config(bg=self.bgColor)
        self.lblMax3.place(x=140, y=startY + 235)

        # Test data range:
        self.lblTestRange = Label(root, text="Test data range:", font=self.font, borderwidth=2, relief="groove")
        self.lblTestRange.config(bg=self.bgColor)
        self.lblTestRange.place(x=60, y=startY + 280)

        self.lblFrom = Label(root, text="From", font=self.fontSmaller)
        self.lblFrom.config(bg=self.bgColor)
        self.lblFrom.place(x=10, y=startY + 325)
        self.txtFrom = Entry(root, width=7)
        self.txtFrom.insert(0, "100")
        self.txtFrom.config(highlightbackground=self.bgColor)
        self.txtFrom.place(x=60, y=startY + 325)

        self.lblMin = Label(root, text="(min: 10)", font=self.fontSmall)
        self.lblMin.config(bg=self.bgColor)
        self.lblMin.place(x=68, y=startY + 352)

        self.lblTo = Label(root, text="To", font=self.fontSmaller)
        self.lblTo.config(bg=self.bgColor)
        self.lblTo.place(x=145, y=startY + 325)
        self.txtTo = Entry(root, width=8)
        self.txtTo.insert(0, "2000")
        self.txtTo.config(highlightbackground=self.bgColor)
        self.txtTo.place(x=175, y=startY + 325)

        self.previous1 = -1
        self.previous2 = -1
        self.previous3 = "None"
        self.previous4 = "None"

        # Agent's performance dates:
        self.lblAgentDates = Label(root, text="Agent's performance dates:", font=self.font, borderwidth=2,
                                   relief="groove")
        self.lblAgentDates.config(bg=self.bgColor)
        self.lblAgentDates.place(x=20, y=startY + 410)

        self.lblFrom2 = Label(root, text="From", font=self.fontSmaller)
        self.lblFrom2.config(bg=self.bgColor)
        self.lblFrom2.place(x=10, y=startY + 455)
        self.txtFrom2 = Entry(root, width=7)
        self.txtFrom2.insert(0, "2100")
        self.txtFrom2.config(highlightbackground=self.bgColor)
        self.txtFrom2.place(x=60, y=startY + 455)

        self.lblMin2 = Label(root, text="(min: 10)", font=self.fontSmall)
        self.lblMin2.config(bg=self.bgColor)
        self.lblMin2.place(x=68, y=startY + 482)

        self.lblTo2 = Label(root, text="To", font=self.fontSmaller)
        self.lblTo2.config(bg=self.bgColor)
        self.lblTo2.place(x=145, y=startY + 455)
        self.txtTo2 = Entry(root, width=8)
        self.txtTo2.insert(0, "2200")
        self.txtTo2.config(highlightbackground=self.bgColor)
        self.txtTo2.place(x=175, y=startY + 455)

        # Start button:
        self.btnStart = Button(root, text="Start", fg="blue", command=self.__btnStartClicked, height=2, width=10,
                               font=self.font)
        self.btnStart.place(x=80, y=startY + 560)

        self.indicatorChoiceVar = StringVar()
        self.indicatorChoiceVar.set('None')
        self.tickerChoiceVar2 = StringVar()
        self.tickerChoiceVar2.set('None')

        root.mainloop()

    # A function for when the "Sale" button is clicked :
    def __btnStartClicked(self):
        try:
            if not (1000 <= int(self.txtTimesteps.get()) <= 1000000):
                messagebox.showerror('Invalid Timesteps!', 'Timesteps has to be a positive number\n'
                                                           'between 1 thousand and 1 million.')
            elif not (10 <= int(self.txtFrom.get()) <= 2520
                      and 10 <= int(self.txtTo.get()) <= 2520
                      and int(self.txtFrom.get()) < int(self.txtTo.get())):
                messagebox.showerror('Invalid Test data range!', 'Test data range has to be between 2 and 2520, \n'
                                                                 'where the \"From\" value is less than the \"To\".')
            elif not (10 <= int(self.txtFrom2.get()) <= 2520
                      and 10 <= int(self.txtTo2.get()) <= 2520
                      and int(self.txtFrom2.get()) < int(self.txtTo2.get())):
                messagebox.showerror('Invalid Agent\'s performance dates!', 'Agent\'s performance dates have to be '
                                                                            'between 2 and 2520, \n '
                                                                            'where the \"From\" value is less than '
                                                                            'the \"To\".')
            else:
                values[0] = self.indicatorChoiceVar.get()
                values[1] = self.tickerChoiceVar2.get()
                self.__drawPlot()

                canvasHorizontalLine = Canvas(root, width=1025, height=1)
                canvasHorizontalLine.place(x=275, y=525)

                canvasAddSeparator = Canvas(root, width=1, height=175)
                canvasAddSeparator.place(x=700, y=525)

                canvasPredictionSeparator = Canvas(root, width=1, height=175)
                canvasPredictionSeparator.place(x=1100, y=525)

                self.__drawAdd()
                self.__drawResults()
                self.__drawPredictions()

                self.previous = values
        except Exception as e:
            print(e)
            messagebox.showerror('Invalid Inputs!', 'All inputs have to be numbers')

    def __btnTestClicked(self):
        values[0] = self.indicatorChoiceVar.get()
        values[1] = self.tickerChoiceVar2.get()

        self.__drawPlot()

        self.lblProfit2.config(text="Total Profit = " + str(self.tool.getProfit()))

        percentage, earn = self.tool.getPercentage()
        if earn:
            self.lblEarnLoss.config(text="The agent has earned " + str(percentage) + "%", fg="green")
        else:
            self.lblEarnLoss.config(text="The agent has lost " + str(percentage) + "%", fg="red")

        print("Indicator: " + self.indicatorChoiceVar.get())
        print("Ticker2: " + self.tickerChoiceVar2.get())

        self.previous = values

    def __drawAdd(self):
        self.previous = ["None", "None"]

        self.lblAddData = Label(root, text="Add more testing data", font=('Times New Roman', 25, 'bold'))
        self.lblAddData.config(bg=self.bgColor)
        self.lblAddData.place(x=350, y=533)

        self.lblIndicator = Label(root, text="Technical Indicator:", font=self.fontSmaller, borderwidth=2,
                                  relief="groove")
        self.lblIndicator.config(bg=self.bgColor)
        self.lblIndicator.place(x=300, y=580)
        self.indicatorChoices = ["None", "RSI", "SMA", "EMA", "WMA", "SMMA", "All"]
        self.indicatorOptionMenu = OptionMenu(root, self.indicatorChoiceVar, *self.indicatorChoices)
        self.indicatorOptionMenu.config(bg=self.bgColor, font=self.fontSmaller)
        self.indicatorOptionMenu.place(x=350, y=615)

        self.lblTicker2 = Label(root, text="Another stock data:", font=self.fontSmaller, borderwidth=2, relief="groove")
        self.lblTicker2.config(bg=self.bgColor)
        self.lblTicker2.place(x=500, y=580)
        tickers2 = tickers.copy()
        tickers2.remove(self.tickerChoiceVar.get())
        self.tickerChoices2 = ["None"] + tickers2
        self.tickerOptionMenu2 = OptionMenu(root, self.tickerChoiceVar2, *self.tickerChoices2)
        self.tickerOptionMenu2.config(bg=self.bgColor, font=self.fontSmaller)
        self.tickerOptionMenu2.place(x=550, y=615)

        self.btnTest = Button(root, text="Test", fg="blue", command=self.__btnTestClicked, height=2, width=8,
                              font=self.font)
        self.btnTest.place(x=450, y=640)

    def __drawResults(self):
        self.lblResults = Label(root, text="Results", font=('Times New Roman', 25, 'bold'))
        self.lblResults.config(bg=self.bgColor)
        self.lblResults.place(x=840, y=533)

        self.lblProfit = Label(root, text="Total Profit = Final Money / Starting Money", font=self.fontSmaller,
                               borderwidth=2)
        self.lblProfit.config(bg=self.bgColor)
        self.lblProfit.place(x=720, y=575)

        self.lblProfit2 = Label(root, text="Total Profit = " + str(self.tool.getProfit()),
                                font=self.fontSmaller, borderwidth=2)
        self.lblProfit2.config(bg=self.bgColor)
        self.lblProfit2.place(x=720, y=605)

        percentage, earn = self.tool.getPercentage()
        if earn:
            self.lblEarnLoss = Label(root, text="The agent has earned " + str(percentage) + "%", fg="green",
                                     font=('Times New Roman', 25, 'bold'))
        else:
            self.lblEarnLoss = Label(root, text="The agent has lost " + str(percentage) + "%", fg="red",
                                     font=('Times New Roman', 25, 'bold'))
        self.lblEarnLoss.config(bg=self.bgColor)
        self.lblEarnLoss.place(x=770, y=640)

    def __drawPredictions(self):
        self.lblAgent = Label(root, text="Agent's", font=('Times New Roman', 25, 'bold'))
        self.lblAgent.config(bg=self.bgColor)
        self.lblAgent.place(x=1145, y=533)

        self.lblPredictions = Label(root, text="Predictions", font=('Times New Roman', 25, 'bold'))
        self.lblPredictions.config(bg=self.bgColor)
        self.lblPredictions.place(x=1130, y=580)

        self.btnOpen = Button(root, text="Open", fg="blue", command=self.__btnOpenClicked, height=2, width=8,
                              font=self.font)
        self.btnOpen.place(x=1155, y=640)

    def __btnOpenClicked(self):
        self.newWindow = Toplevel(root)
        self.newWindow.title("Agent's Predictions")
        self.newWindow.geometry("400x350")
        self.newWindow.config(bg=self.bgColor)

        lblBounds = Label(self.newWindow, text="The agent's current date frame bounds are:", font=self.fontSmaller,
                          borderwidth=2,
                          relief="groove")
        lblBounds.config(bg=self.bgColor)
        lblBounds.place(x=20, y=40)

        self.tool.getBounds()
        bounds = "From:  " + str(self.tool.bound1) + "        To:  " + str(self.tool.bound2)
        lblBounds2 = Label(self.newWindow, text=bounds, font=self.font)
        lblBounds2.config(bg=self.bgColor)
        lblBounds2.place(x=20, y=75)

        lblEnter = Label(self.newWindow, text="Enter a date in these frame bounds:", font=self.fontSmaller,
                         borderwidth=2,
                         relief="groove")
        lblEnter.config(bg=self.bgColor)
        lblEnter.place(x=40, y=130)

        self.txtDate = Entry(self.newWindow, width=15)
        self.txtDate.config(highlightbackground=self.bgColor)
        self.txtDate.place(x=120, y=175)

        self.btnPredict = Button(self.newWindow, text="Predict", fg="blue", command=self.__btnPredictClicked, height=2,
                                 width=8,
                                 font=self.font)
        self.btnPredict.place(x=150, y=220)

    def __btnPredictClicked(self):
        if self.tool.isValidDate(self.txtDate.get()):
            action = self.tool.predict_action_on_date()
            if action == 1:
                self.lblPrediction = Label(self.newWindow, text="The agent would buy!", fg="green",
                                           font=('Times New Roman', 25, 'bold'))
            else:
                self.lblPrediction = Label(self.newWindow, text="The agent would sell!", fg="red",
                                           font=('Times New Roman', 25, 'bold'))
            self.lblPrediction.config(bg=self.bgColor)
            self.lblPrediction.place(x=80, y=290)
        else:
            messagebox.showerror('Invalid date!', 'Please enter a date within the given frame bounds.')

    def __drawPlot(self):
        if not (int(self.txtFrom.get()) == self.previous1 and int(self.txtTo.get()) == self.previous2 and
                self.indicatorChoiceVar.get() == self.previous3 and self.tickerChoiceVar2.get() == self.previous4):
            self.tool = Tool(self.tickerChoiceVar.get(), self.algorithmChoiceVar.get(), self.txtFrom.get(),
                         self.txtTo.get(),
                         self.txtFrom2.get(), self.txtTo2.get(), self.txtTimesteps.get())
            self.previous1 = int(self.txtFrom.get())
            self.previous2 = int(self.txtTo.get())
            self.previous3 = self.indicatorChoiceVar.get()
            self.previous4 = self.tickerChoiceVar2.get()
        self.tool.plot(int(self.txtFrom2.get()), int(self.txtTo2.get()))

        i = Image.open("temp.jpg")
        img = i.resize((1000, 500))
        image = ImageTk.PhotoImage(img)
        panel = Label(root, image=image)
        panel.photo = image
        panel.place(x=300, y=0)

    def callback(self, *args):
        txt = "(max: " + str(2520) + ")"
        self.lblMax.config(text=txt)
        self.lblMax2.config(text=txt)
        pass

# Creating a tkinter variable :
root = Tk()
# Calling the class :
gui = ToolGUI()
