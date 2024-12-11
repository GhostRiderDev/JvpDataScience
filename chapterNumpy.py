import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import special


def nestListFn():
    nestedList = [range(i, i + 3) for i in [2, 4, 6]]

    for i in nestedList:
        for j in i:
            print(j, end=" ")
        print("")


def createZeros():
    zeros = np.zeros(10, dtype=int)
    print(zeros)


def createOnes():
    ones = np.ones((3, 5), dtype=float)
    print(ones)


# createOnes()


def createFull():
    full = np.full(shape=(5, 6), fill_value=3.14, dtype=float)
    print(full)


# createFull()


def createSeq():
    sequence = np.arange(0, 20, 2)
    print(sequence)


# createSeq()


def createAuto():
    auto = np.linspace(0, 1, 11)
    print(auto)


# createAuto()


def createRandom():
    randoms = np.random.random((3, 4))
    print(randoms)


# createRandom()


def createNormal():
    normals = np.random.normal(0, 1, (3, 3))
    print(normals)


# createNormal()


def createRandomInts():
    randomInts = np.random.randint(0, 10, (3, 3))
    print(randomInts)


# createRandomInts()


def createIdentity():
    indentity = np.eye(3)
    print(indentity)


# createIdentity()


def createEmpty():
    empty = np.empty(3)
    print(empty)


# createEmpty()

############?Data Types##########################


def zerosType():
    zerosTypes = np.zeros(10, dtype=np.int32)
    print(zerosTypes)


# zerosType()


def arrayAtt():
    rng = np.random.default_rng(seed=1707)

    x1 = rng.integers(10, size=6)

    x2 = rng.integers(10, size=(3, 4))

    x3 = rng.integers(10, size=(3, 4, 5))

    print(x1)
    print("-----------------------------")
    print(x2)
    print("-----------------------------")
    print(x3)


# arrayAtt()


def arrayPosIdx():
    array = np.array([1, 3, 5, 6, 8])
    print(array[0])
    print(array[4])
    print(array[-1])
    print(array[-2])

    print("-------------------------")

    array2d = np.array([[1, 3, 4], [2, 5, 7], [9, 2, 6]])

    print(array2d[0, 0])
    print(array2d[2, 0])
    print(array2d[2, -1])

    print("----------------------")

    array2d[0, 0] = 11
    print(array2d)

    array[0] = 3.14544
    print(array)


# arrayPosIdx()


def arraySlicing():
    x1 = np.array([1, 4, 5, 6, 7, 8])
    print(x1[:3])

    print("---------------")

    print(x1[3:])

    print("---------------")

    print(x1[1:4])

    print("---------------")

    print(x1[::2])

    print("---------------")

    print(x1[1::2])

    print("----------------")

    print(x1[::-1])

    print("----------------")
    print(x1[4::-2])

    print("********************")

    x2 = np.array([[2, 4, 7, 5], [7, 4, 2, 7], [3, 6, 8, 2]])
    print([x2[:2, :3]])

    print("------------------")

    print(x2[:3, :2])

    print("------------------")

    print(x2[::-1, ::-1])

    print("------------------")

    print(x2[:, 0])

    print("---------------------")

    print(x2[0, :])

    print("--------------------")
    print(x2[0])

    print("----------------------")
    x2_sub = x2[:2, :2]
    print(x2_sub)
    x2_sub[0, 0] = 28
    print(x2)

    print("------------------")

    x2_sub_cp = x2[:2, :2].copy()
    print(x2_sub)
    x2_sub_cp[0, 0] = 99
    print(x2_sub_cp)
    print(x2)


# arraySlicing()


def reshapingArrays():
    grid = np.arange(1, 10).reshape(3, 3)
    print(grid)

    x = np.array([1, 2, 3])
    print(x.reshape((1, 3)))
    print(x.reshape((3, 1)))


# reshapingArrays()


def concatSplit():
    x = np.array([1, 3, 7, 8])
    y = np.array([4, 7, 9, 11])

    print(np.concatenate([x, y]))

    z = np.array([13, 19, 23])

    print(np.concatenate([x, y, z]))

    grid = np.array([[1, 4, 5, 6], [8, 11, 9, 13]])

    print(np.concatenate([grid, grid]))

    print(np.concatenate([grid, grid], axis=1))

    print(np.vstack([x, grid]))

    print(np.hstack([x.reshape(2, 2), grid]))

    print("------------------------------------")

    x = [2, 4, 5, 6, 7, 8, 11, 13, 9, 23, 45, 1]
    x1, x2, x3 = np.split(x, [3, 7])
    print(x1, x2, x3)

    grid = np.arange(16).reshape((4, 4))

    print(grid)

    upper, lower = np.vsplit(grid, [2])
    print("**************")
    print(upper)
    print("----------")
    print(lower)

    left, right = np.hsplit(grid, [2])
    print(left)
    print(right)


# concatSplit()

##################? Universal Function################


def ufuncs():
    x = np.arange(9).reshape((3, 3))
    print(2**x)
    print(x // 2)
    print(-((0.5 * x + 1) ** 2))
    x = x * -1
    print(np.abs(x))

    thetas = np.linspace(0, np.pi, 3)

    print(thetas)
    print(np.sin(thetas))
    print(np.cos(thetas))
    print(np.tan(thetas))

    x = np.array([2, 4, 7, 11])

    print(np.exp(x))
    print(np.exp2(x))
    print(np.power(x, 5))

    print(np.log(x))
    print(np.log2(x))
    print(np.log10(x))

    print(np.expm1(x))
    print(np.log1p(x))

    print(special.gamma(x))
    print(special.gammaln(x))

    y = np.empty(4)

    np.multiply(x, 10, out=y)
    print(y)

    y = np.zeros(8)

    np.power(2, x, out=y[::2])
    print(y)


def agregates():
    x = np.arange(1, 6)
    print(np.add.reduce(x))
    print(np.multiply.reduce(x))
    print(np.add.accumulate(x))
    print(np.multiply.accumulate(x))
    print(np.multiply.outer(x, x))

    rng = np.random.default_rng()

    L = rng.random(100)

    print(np.sum(L))
    print(np.min(L))
    print(np.max(L))

    print("-----------------------")

    M = rng.integers(0, 10, (3, 4))

    print(M)
    print(np.sum(M))
    print(np.sum(M, axis=1))
    print(np.max(M, axis=1))


def usPresidentHeight():
    data = pd.read_csv("data/president_heights.csv")
    heights = np.array(data["height(cm)"])
    print(heights)
    print("Mean: ", heights.mean())
    print("Min: ", heights.min())
    print("Max: ", heights.max())
    print("Standart: ", heights.std())
    print("25th percentile: ", np.percentile(heights, 25))
    print("50th percentile: ", np.percentile(heights, 50))
    print("75th percentile: ", np.percentile(heights, 75))

    # plt.style.use("seaborn-whitegrid")
    plt.hist(heights)
    plt.title("Hight distribution of US presidents")
    plt.xlabel("height (cm)")
    plt.ylabel("number")
    plt.show()


# usPresidentHeight()


def broadcasting():
    a = np.array([0, 1, 2])
    b = np.array([5, 5, 5])
    print(a + b)
    print(a + 5)

    M = np.ones((3, 3))
    print(M)

    print(a + M)

    a = np.arange(3)
    b = np.arange(3)[:, np.newaxis]

    print(a)
    print(b)

    print(a + b)

    rng = np.random.default_rng(seed=1701)

    X = rng.random((10, 3))

    print(X)
    print(X.mean(0))

    X_mean = X.mean(0)
    X_Centered = X - X_mean

    print(X_Centered)


# broadcasting()


def plooting():
    x = np.linspace(0, 5, 50)
    y = np.linspace(0, 5, 50).reshape((50, 1))

    z = np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)

    plt.imshow(z, origin="lower", extent=[0, 5, 0, 5])
    plt.colorbar()
    plt.show()


def seatleRains():
    df = pd.read_csv("data/seattle-weather.csv")

    # Asegurarse de que la columna 'date' es de tipo datetime
    df["date"] = pd.to_datetime(df["date"])

    # Filtrar las filas correspondientes al año 2015
    df_2015 = df[df["date"].dt.year == 2015]

    # Extraer la columna de precipitaciones y convertirla en un array de NumPy
    rainfall_mm_2015 = np.array(df_2015.set_index("date")["precipitation"])

    # Imprimir la longitud del array para verificar
    print(len(rainfall_mm_2015))

    # Imprimir el array de precipitaciones del 2015
    print(rainfall_mm_2015)

    plt.hist(rainfall_mm_2015, 40)
    plt.show()


# seatleRains()


def comparetors():
    x = np.array([1, 2, 3, 4, 5])
    print(x < 3)

    rng = np.random.default_rng(seed=1701)
    x = rng.integers(0, 16, (3, 4))
    print(x)
    print(np.count_nonzero(x < 6))
    print(np.any(x > 8))
    print(np.all(x == 10))
    lower_than5 = x[x < 5]
    print(lower_than5)
    print(bin(42))


# comparetors()


def fancyIndex():
    rng = np.random.default_rng(seed=1701)
    x = rng.integers(100, size=10)
    print(x)
    idx = [3, 5, 7]
    print(x[idx])
    idx = np.array([[1, 4], [8, 9]])
    print(x[idx])


# fancyIndex()


def sortArray():
    X = np.random.rand(10, 2)

    dist_sq = np.sum((X[:, np.newaxis] - X[np.newaxis, :]) ** 2, axis=-1)
    print("------------------------")
    print(dist_sq)
    print("------------------------")

    K = 3
    if K + 1 > dist_sq.shape[1]:
        raise ValueError(f"K + 1 ({K + 1}) es mayor que el número de elementos ({dist_sq.shape[1]}) en dist_sq a lo largo del eje 1")

    nearest_partition = np.argpartition(dist_sq, K + 1, axis=1)
    plt.scatter(X[:, 0], X[:, 1], s=100)
    for i in range(X.shape[0]):
        for j in nearest_partition[i, : K + 1]:
            plt.plot(*zip(X[j], X[i]), color="black")
    plt.show()

sortArray()
