import pandas as pd
import numpy as np

df = pd.read_csv("data/2026.csv")
data = df.set_index('Player')
means = {}
std = {}

for col in ["PTS","AST","TRB","STL","BLK","GS","eFG%","TOV","PF"]:
    means[col] = np.mean(df[col])
    std[col] = np.std(df[col])

impact_vals = (df["PTS"] + df["AST"] + df["STL"] - df["TOV"]) / df["MP"].replace(0, 1)
means["impact"] = np.mean(impact_vals)
std["impact"] = np.std(impact_vals)


class Player:
    #this is used to convert the strs of the header of the dataset to self.(varnames)
    statMap = {
        "PTS": "pts",
        "AST": "ast",
        "TRB": "reb",
        "STL": "stl",
        "BLK": "blk",
        "GS": "gs",
        "eFG%": "efg",
        "TOV": "tov",
        "PF": "pf",
        "impact": "impact",
    }

    #these coef come from the logisitic regression that I trained
    #link to that github: https://github.com/AdeelSyed897/NBA-All-Star-ML
    modelCoef = {
        "BLK": 0.1923102818414375,
        "AST": 0.30958064380196687,
        "TOV": 0.3124880363606635,
        "PF": -0.34849204008917944,
        "impact": 0.3964833979980206,
        "STL": 0.42064100175520003,
        "TRB": 0.5520399517470511,
        "eFG%": 0.7553335232531223,
        "GS": 1.1437457442579049,
        "PTS": 1.7623577007708657,
    }
    intercept = -8.822259004092109


    def __init__(self, name):
        self.name = name
        row = data.loc[name]
        # Incase someone is traded like luka :(
        if isinstance(row, pd.DataFrame):
            row = row[row["Team"].str.endswith("TM")].iloc[0]

        self.pts = row['PTS']
        self.reb = row['TRB']
        self.ast = row['AST']
        self.stl = row['STL']
        self.blk = row['BLK']
        self.efg = row['eFG%']
        self.tov = row['TOV']
        self.pf = row['PF']
        self.gs = row['GS']
        self.mp = row['MP']

        box = (row['PTS'] + row['AST'] + row['STL'] - row['TOV'])
        if self.mp == 0:
            self.impact = box / 1
        else:
            self.impact = box / self.mp

    #This is helper functions standarize the data
    def standardizeStat(self, cat):
        attr = Player.statMap[cat]
        val = getattr(self, attr)
        return (val - means[cat]) / std[cat]
    

    #This is the math for the logisitic regression model
    def AllStarProb(self):
        val = Player.intercept

        for cat, coef in Player.modelCoef.items():
            val += coef * self.standardizeStat(cat)

        prob = 1 / (1 + np.exp(-val))
        return prob