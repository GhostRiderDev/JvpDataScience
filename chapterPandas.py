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
        {"California": 39538223, "Texas": 29145505, "Florida": 21538187}, name="population"
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
    
    data = pd.Series([1, np.nan, 'Hello', None])
    print(data.isnull())
    print(data[data.notnull()])
    print(data.dropna())
    
nullValues()


