import pandas as pd
import numpy as np
from math import inf
import seaborn as sns
from datetime import datetime
from dateutil import parser
import os
import requests
import json
import yfinance as yf
import matplotlib.pyplot as plt


def basic():
    data = pd.Series([0, 0.25, 0.5, 0.75, 1.0], index=["a", "b", "c", "d", "e"])
    print(data.values)
    print(data.index)
    print(data["a"])

    states_population = {
        "Cordoba": 1245844,
        "Antioquia": 384848848,
        "Caldas": 12443434,
        "Atlantico": 14434444,
        "Sucre": 48858583,
        "Cundinamarcar": 88558588883,
    }

    populationSerie = pd.Series(states_population)
    print(populationSerie)
    print(populationSerie["Caldas":"Sucre"])

    states_area = {
        "Cordoba": 1245844,
        "Antioquia": 384848848,
        "Caldas": 12443434,
        "Atlantico": 14434444,
        "Sucre": 48858583,
        "Cundinamarcar": 88558588883,
    }

    df_states_area = pd.Series(states_area)
    print(df_states_area)

    states = pd.DataFrame({"population": populationSerie, "area": df_states_area})

    print(states)
    print(states.columns)
    print(
        pd.DataFrame(
            np.random.rand(3, 2), columns=["foo", "bar"], index=["a", "b", "c"]
        )
    )

    indA = pd.Index([1, 3, 5, 7, 9])
    indB = pd.Index([2, 3, 5, 7, 11])

    print(indA.intersection(indB))
    print(indA.union(indB))
    print(indA.symmetric_difference(indB))


def indexingSelection():
    data = pd.Series([0.25, 0.5, 0.75, 1.0], index=["a", "b", "c", "d"])
    print("a" in data)
    print(data.keys())
    print(list(data.items()))
    print(data[(data > 0.3) & (data < 0.8)])

    data = pd.Series(["a", "b", "c"], index=[1, 3, 5])
    print(data)

    area = pd.Series(
        {
            "California": 423967,
            "Texas": 695662,
            "Florida": 170312,
            "New York": 141297,
            "Pennsylvania": 119280,
        }
    )
    pop = pd.Series(
        {
            "California": 39538223,
            "Texas": 29145505,
            "Florida": 21538187,
            "New York": 20201249,
            "Pennsylvania": 13002700,
        }
    )

    data = pd.DataFrame({"area": area, "pop": pop})
    print(data)

    data["density"] = data["pop"] / data["area"]
    print(data)

    print(data.values)

    ##* To transport rows/cols
    print(data.T)

    print(data.iloc[:3, :2])

    print(data.loc[:"Florida", :"pop"])

    print(data.loc[data["density"] > 120, ["density", "pop"]])


# indexingSelection()


def indexAlign():
    area = pd.Series(
        {"Alaska": 1723337, "Texas": 695662, "California": 423967}, name="area"
    )
    population = pd.Series(
        {"California": 39538223, "Texas": 29145505, "Florida": 21538187},
        name="population",
    )

    print(population.div(area, fill_value=0).replace(inf, 0))

    print(area.index.union(population.index))


# indexAlign()


def nullValues():
    vals1 = np.array([1, None, 2, 3])
    print(vals1)
    # ! print(vals1.sum()) error!!
    vals1 = np.array([1, np.nan, 2, 3])
    print(vals1.sum(), vals1.min(), vals1.max())
    print(np.nansum(vals1))

    x = pd.Series(range(2), dtype=int)
    print(x)
    x[0] = None
    print(x)

    x = pd.Series([1, np.nan, 2, None, pd.NA], dtype="Int32")
    print(x)

    data = pd.Series([1, np.nan, "Hello", None])
    print(data.isnull())
    print(data[data.notnull()])
    print(data.dropna())

    df = pd.DataFrame([[1, np.nan, 2], [2, 3, 5], [np.nan, 4, 6]])
    print(df)
    print(df.dropna())

    print("-----------------------")
    data = pd.Series([1, np.nan, 2, None, 3], index=list("abcde"), dtype="Int32")
    print(data)
    print(data.fillna(0))
    print(data.ffill())
    print(data.bfill())
    print(data)


# nullValues()


def hierarchicalIndexing():
    index = [
        ("California", 2010),
        ("California", 2020),
        ("New York", 2010),
        ("New York", 2020),
        ("Texas", 2010),
        ("Texas", 2020),
    ]
    populations = [37253956, 39538223, 19378102, 20201249, 25145561, 29145505]

    pop = pd.Series(populations, index=index)
    print(pop)
    print(pop[("California", 2020):("Texas", 2010)])
    print(pop[[i for i in pop.index if i[1] == 2010]])

    index = pd.MultiIndex.from_tuples(index)
    pop = pop.reindex(index)
    print(pop)
    print(pop[:, 2020])

    pop_df = pop.unstack()
    print(pop_df)
    print(pop_df.stack())

    pop_df = pd.DataFrame(
        {
            "total": pop,
            "under18": [9284094, 8898092, 4318033, 4181528, 6879014, 7432474],
        }
    )
    print(pop_df)

    f_u18 = pop_df["under18"] / pop_df["total"]
    print(f_u18.unstack())


# hierarchicalIndexing()


def mergeConcatDf():
    print(makeDf("ABV", range(3)))

    ser1 = pd.Series(["A", "B", "C"], range(1, 4))
    ser2 = pd.Series(["D", "E", "F"], range(4, 7))

    print(pd.concat([ser1, ser2]))

    x = makeDf("AB", [0, 1])
    y = makeDf("AB", [2, 3])
    y.index = x.index  # make indices match


def makeDf(cols, idx):
    data = {c: [str(c) + str(i) for i in idx] for c in cols}
    return pd.DataFrame(data, idx)


# mergeConcatDf()


def relationalAlgebra():
    df1 = pd.DataFrame(
        {
            "employee": ["Bob", "Jake", "Lisa", "Sue"],
            "group": ["Accounting", "Engineering", "Engineering", "HR"],
        }
    )
    df2 = pd.DataFrame(
        {
            "employee": ["Lisa", "Bob", "Jake", "Sue"],
            "hire_date": [2004, 2008, 2012, 2014],
        }
    )

    print(df1, df2)

    df3 = pd.merge(df1, df2)

    print(df3)

    df4 = pd.DataFrame(
        {
            "group": ["Accounting", "Engineering", "HR"],
            "supervisor": ["Carly", "Guido", "Steve"],
        }
    )

    df5 = pd.merge(df3, df4)
    print(df5)

    df6 = pd.DataFrame(
        {
            "group": [
                "Accounting",
                "Accounting",
                "Engineering",
                "Engineering",
                "HR",
                "HR",
            ],
            "skills": [
                "math",
                "spreadsheets",
                "software",
                "math",
                "spreadsheets",
                "organization",
            ],
        }
    )

    df7 = pd.merge(df5, df6)
    print(df7)

    df6 = pd.DataFrame(
        {"name": ["Peter", "Paul", "Mary"], "food": ["fish", "beans", "bread"]},
        columns=["name", "food"],
    )
    df7 = pd.DataFrame(
        {"name": ["Mary", "Joseph"], "drink": ["wine", "beer"]},
        columns=["name", "drink"],
    )

    print(pd.merge(df6, df7, how="inner"))
    print(pd.merge(df6, df7, how="outer"))

    df8 = pd.DataFrame({"name": ["Bob", "Jake", "Lisa", "Sue"], "rank": [1, 2, 3, 4]})
    df9 = pd.DataFrame({"name": ["Bob", "Jake", "Lisa", "Sue"], "rank": [3, 1, 4, 2]})
    print(pd.merge(df8, df9, on="name", suffixes=["_L", "_R"]))

    pop = pd.read_csv("data/state-population.csv")
    areas = pd.read_csv("data/state-areas.csv")
    abbrevs = pd.read_csv("data/state-abbrevs.csv")

    merged = pd.merge(
        pop, abbrevs, how="outer", left_on="state/region", right_on="abbreviation"
    )

    merged = merged.drop("abbreviation", axis=1)
    print(merged)
    print(merged.isnull().any())
    print(merged[merged["population"].isnull()].head())
    print(merged.loc[merged["state"].isnull(), "state/region"].unique())
    merged.loc[merged["state/region"] == "PR", "state"] = "Puerto Rico"
    merged.loc[merged["state/region"] == "USA", "state"] = "United States"

    print(merged.isnull().any())

    final = pd.merge(merged, areas, on="state", how="left")
    print(final)

    print(final.isnull().any())

    print(final["state"][final["area (sq. mi)"].isnull()].unique())

    final.dropna(inplace=True)
    print(final.head())

    data2010 = final.query("year == 2010 & ages == 'total'")
    print(data2010)

    data2010.set_index("state", inplace=True)
    print(data2010)
    density = data2010["population"] / data2010["area (sq. mi)"]
    density.sort_values(ascending=False, inplace=True)
    print(density.head())


# relationalAlgebra()
def filterFunc(x):
    return x["data2"].std() > 4


def center(x):
    return x - x.mean()


def norm_by_data2(x):
    x["data1"] /= x["data2"].sum()
    return x


def grouping():
    planets = sns.load_dataset("planets")
    print(planets.shape)
    print(planets.head())
    print(planets.dropna().describe())

    df = pd.DataFrame(
        {"key": ["A", "B", "C", "A", "B", "C"], "data": range(6)},
        columns=["key", "data"],
    )

    print(df)
    print(df.groupby("key"))
    print(df.groupby("key").sum())
    print(planets.groupby("method")["orbital_period"])

    for method, group in planets.groupby("method"):
        print("{0:30s} shape={1}".format(method, group.shape))

    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "key": ["A", "B", "C", "A", "B", "C"],
            "data1": range(6),
            "data2": rng.randint(0, 10, 6),
        },
        columns=["key", "data1", "data2"],
    )

    print(df.groupby("key").aggregate(["min", np.median, max]))
    print(df.groupby("key").aggregate({"data1": "min", "data2": "max"}))

    print(df.groupby("key").filter(filterFunc))

    print(df.groupby("key").transform(center))

    print(df.groupby("key").apply(norm_by_data2))

    L = [0, 1, 0, 1, 2, 0]
    print(df.groupby(L).sum())


# grouping()


def pivotTables():
    titanic = sns.load_dataset("titanic")

    print(titanic.head())

    print(titanic.groupby("sex")[["survived"]].mean())
    print(titanic.groupby(["sex", "class"])["survived"].aggregate("mean").unstack())

    print(titanic.pivot_table("survived", index="sex", columns="class", aggfunc="mean"))


# pivotTables()


def timeSeries():
    print(datetime(year=2021, month=7, day=4))
    date = parser.parse("4th of July, 2021")
    print(date)
    print(date.strftime("%A"))
    date = np.array("2024-07-12", dtype=np.datetime64)
    print(date)

    dateArray = date + np.arange(12)
    print(dateArray)

    print(np.datetime64("2021-12-07 12:30"))
    date = pd.to_datetime("4th of July, 2023")
    print(date)

    index = pd.DatetimeIndex(["2021-04-23", "2022-07-12", "2014-05-09", "2021-08-19"])

    data = pd.Series([0, 1, 2, 3], index=index)

    print(data)
    print(data["2021"])

    dates = pd.to_datetime(
        [
            datetime(2024, 12, 9),
            "4th of July, 2021",
            "2021-Jul-6",
            "07-07-2021",
            "20210708",
        ]
    )

    print(pd.DatetimeIndex(dates))
    print(dates.to_period("D"))
    print(dates - dates[0])
    print(pd.date_range("2021-07-09", "2021-07-23"))
    print(pd.date_range("2024-12-01", periods=8))
    print(pd.date_range("2024-12-01", periods=8, freq="h"))
    print(pd.date_range("2024-12-01", periods=8, freq="M"))
    print(pd.timedelta_range(0, periods=6, freq="2H30T"))

    dat = yf.Ticker("MSFT")
    
    data = dat.history(period="max")

    print(data.head())
    print(data.columns)
    closingPrices = data["Close"]
    closingPrices.plot(title="MSFT Closing Prices")
    plt.xlabel("Date")
    plt.ylabel("Closing Price")
    plt.savefig("proyecciones.png")

timeSeries()
