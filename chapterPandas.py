import pandas as pd
import numpy as np
from math import inf


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
    print(pd.merge(df8, df9))


relationalAlgebra()
