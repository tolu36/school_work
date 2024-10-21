# %%
# read in the packages
import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import time

# %%
# read in the dataset
df = pd.read_csv("Tetuan City power consumption.csv")
df["Total Power Consumption"] = df.iloc[:, -3:].sum(axis=1)
df.drop(columns=list(df.iloc[:, -4:-1].columns), inplace=True)
df.drop(columns=["DateTime"], inplace=True)
df_scaled = StandardScaler().fit_transform(df)


# %%
error_rate = []
np.random.seed(131)
df_scaled_x = df_scaled[:, 0 : df_scaled.shape[1] - 1]
X_train, X_test, y_train, y_test = train_test_split(
    df_scaled_x,
    df_scaled[:, df_scaled.shape[1] - 1],
    random_state=0,
    test_size=0.30,
)

for i in range(1, 20):
    knn = KNeighborsRegressor(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    error_rate.append(mean_squared_error(y_test, pred))

plt.plot(range(1, 20), error_rate, marker="o")
plt.show()

print(
    f"The minimum error of {min(error_rate)} is reach at k = {np.argmin(error_rate)}."
)

# %%
time_stamps = {}
train_mse = []
test_mse = []
pca_mse = []
df_scaled_x = df_scaled[:, 0 : df_scaled.shape[1] - 1]
for i in list(range(1, df_scaled_x.shape[1] + 1))[::-1]:
    # pca
    start_time = time.time()
    pca = PCA(n_components=i)
    pca.fit(df_scaled_x)
    X_Reduced = pca.transform(df_scaled_x)
    XReconstructed = pca.inverse_transform(X_Reduced)
    pca_mse.append(mean_squared_error(df_scaled_x, XReconstructed))
    stop_time = time.time()
    total = stop_time - start_time
    time_stamps[f"pca component {i}"] = {"pca fit": total}

    # partitioning the data
    np.random.seed(131)
    X_train, X_test, y_train, y_test = train_test_split(
        X_Reduced,
        df_scaled[:, df_scaled.shape[1] - 1],
        random_state=0,
        test_size=0.30,
    )
    # training
    start_time = time.time()
    knn = KNeighborsRegressor(n_neighbors=7)
    knn.fit(X_train, y_train)
    prediction = knn.predict(X_train)
    train_mse.append(mean_squared_error(y_train, prediction))
    stop_time = time.time()
    total = stop_time - start_time
    time_stamps[f"train w/ pca component {i}"] = {"train": total}
    print(f"training R^2 with {i} features is {r2_score(y_train, prediction)}.")

    # testing
    start_time = time.time()
    prediction = knn.predict(X_test)
    test_mse.append(mean_squared_error(y_test, prediction))
    stop_time = time.time()
    total = stop_time - start_time
    time_stamps[f"test w/ pca component {i}"] = {"test": total}
    print(f"test R^2 with {i} features is {r2_score(y_test, prediction)}.")
plt.plot(
    list(range(1, df_scaled_x.shape[1] + 1)[::-1]), pca_mse, linestyle="--", marker="o"
)
plt.ylabel("PCA MSE")
plt.xlabel("Number of Features")
plt.show()

plt.plot(
    list(range(1, df_scaled_x.shape[1] + 1)[::-1]),
    train_mse,
    linestyle="--",
    marker="o",
)
plt.ylabel("Train MSE")
plt.xlabel("Number of Features")
plt.show()

plt.plot(
    list(range(1, df_scaled_x.shape[1] + 1)[::-1]), test_mse, linestyle="--", marker="o"
)
plt.ylabel("Test MSE")
plt.xlabel("Number of Features")
plt.show()


# %%
time_lst = []
for i in list(time_stamps.keys()):
    time_lst.append(
        [i, list(time_stamps[i].keys())[0], list(time_stamps[i].values())[0]]
    )

df = pd.DataFrame(time_lst, columns=["group", "column", "val"])

ax = df.pivot("column", "group", "val").plot(kind="bar", figsize=(8, 8))
ax.set(xlabel="model process", ylabel="Time (seconds)")
plt.tight_layout()
plt.show()

# %%
time_stamps = {}
train_mse = []
test_mse = []

df_scaled_x = df_scaled[:, 0 : df_scaled.shape[1] - 1]
df_scaled_y = df_scaled[:, df_scaled.shape[1] - 1]

# partitioning the data
np.random.seed(131)
X_train, X_test, y_train, y_test = train_test_split(
    df_scaled_x,
    df_scaled_y,
    random_state=0,
    test_size=0.30,
)
knn = KNeighborsRegressor(n_neighbors=7)

# training
start_time = time.time()
knn.fit(X_train, y_train)
prediction = knn.predict(X_train)
train_mse.append(mean_squared_error(y_train, prediction))
stop_time = time.time()
total = stop_time - start_time
time_stamps[f"train w/ sfs {5}"] = {"train": total}
print(f"training R^2 with {5} features is {r2_score(y_train, prediction)}.")

# testing
start_time = time.time()
prediction = knn.predict(X_test)
test_mse.append(mean_squared_error(y_test, prediction))
stop_time = time.time()
total = stop_time - start_time
time_stamps[f"test w/ sfs {5}"] = {"test": total}
print(f"test R^2 with {5} features is {r2_score(y_test, prediction)}.")

np.random.seed(131)
for i in [4, 3, 2, 1]:
    # backward selection
    start_time = time.time()
    knn = KNeighborsRegressor(n_neighbors=7)
    sfs = SequentialFeatureSelector(
        knn,
        n_features_to_select=i,
        direction="backward",
        scoring="neg_mean_squared_error",
    )
    sfs.fit(df_scaled_x, df_scaled_y)
    Xtest = sfs.transform(df_scaled_x)
    stop_time = time.time()
    total = stop_time - start_time
    time_stamps[f"train w/ sfs {i}"] = {"train": total}

    # partitioning the data
    np.random.seed(131)
    X_train, X_test, y_train, y_test = train_test_split(
        Xtest,
        df_scaled_y,
        random_state=0,
        test_size=0.30,
    )

    # training
    start_time = time.time()
    knn.fit(X_train, y_train)
    prediction = knn.predict(X_train)
    train_mse.append(mean_squared_error(y_train, prediction))
    stop_time = time.time()
    total = stop_time - start_time
    time_stamps[f"train w/ sfs {i}"] = {"train": total}
    print(f"training R^2 with {i} features is {r2_score(y_train, prediction)}.")

    # testing
    start_time = time.time()
    prediction = knn.predict(X_test)
    test_mse.append(mean_squared_error(y_test, prediction))
    stop_time = time.time()
    total = stop_time - start_time
    time_stamps[f"test w/ sfs {i}"] = {"test": total}
    print(f"test R^2 with {i} features is {r2_score(y_test, prediction)}.")

plt.plot(
    list(range(1, df_scaled_x.shape[1] + 1)[::-1]),
    train_mse,
    linestyle="--",
    marker="o",
)
plt.ylabel("Train MSE")
plt.xlabel("Number of Features")
plt.show()

plt.plot(
    list(range(1, df_scaled_x.shape[1] + 1)[::-1]), test_mse, linestyle="--", marker="o"
)
plt.ylabel("Test MSE")
plt.xlabel("Number of Features")
plt.show()

# %%
time_lst = []
for i in list(time_stamps.keys()):
    time_lst.append(
        [i, list(time_stamps[i].keys())[0], list(time_stamps[i].values())[0]]
    )

df = pd.DataFrame(time_lst, columns=["group", "column", "val"])

ax = df.pivot("column", "group", "val").plot(kind="bar", figsize=(8, 8))
ax.set(xlabel="model process", ylabel="Time (seconds)")
plt.tight_layout()
plt.show()

# %%
time_stamps = {}
train_mse = []
test_mse = []
pca_mse = []
df_scaled_x = df_scaled[:, 0 : df_scaled.shape[1] - 1]
for i in list(range(1, df_scaled_x.shape[1] + 1))[::-1]:
    # pca
    start_time = time.time()
    pca = PCA(n_components=i)
    pca.fit(df_scaled_x)
    X_Reduced = pca.transform(df_scaled_x)
    XReconstructed = pca.inverse_transform(X_Reduced)
    pca_mse.append(mean_squared_error(df_scaled_x, XReconstructed))
    stop_time = time.time()
    total = stop_time - start_time
    time_stamps[f"pca component {i}"] = {"pca fit": total}

    # partitioning the data
    np.random.seed(131)
    X_train, X_test, y_train, y_test = train_test_split(
        X_Reduced,
        df_scaled[:, df_scaled.shape[1] - 1],
        random_state=0,
        test_size=0.30,
    )
    # training
    start_time = time.time()
    lnreg = LinearRegression()
    lnreg.fit(X_train, y_train)
    prediction = lnreg.predict(X_train)
    train_mse.append(mean_squared_error(y_train, prediction))
    stop_time = time.time()
    total = stop_time - start_time
    time_stamps[f"train w/ pca component {i}"] = {"train": total}
    print(f"training R^2 with {i} features is {r2_score(y_train, prediction)}.")

    # testing
    start_time = time.time()
    prediction = lnreg.predict(X_test)
    test_mse.append(mean_squared_error(y_test, prediction))
    stop_time = time.time()
    total = stop_time - start_time
    time_stamps[f"test w/ pca component {i}"] = {"test": total}
    print(f"test R^2 with {i} features is {r2_score(y_test, prediction)}.")
plt.plot(
    list(range(1, df_scaled_x.shape[1] + 1)[::-1]), pca_mse, linestyle="--", marker="o"
)
plt.ylabel("PCA MSE")
plt.xlabel("Number of Features")
plt.show()

plt.plot(
    list(range(1, df_scaled_x.shape[1] + 1)[::-1]),
    train_mse,
    linestyle="--",
    marker="o",
)
plt.ylabel("Train MSE")
plt.xlabel("Number of Features")
plt.show()

plt.plot(
    list(range(1, df_scaled_x.shape[1] + 1)[::-1]), test_mse, linestyle="--", marker="o"
)
plt.ylabel("Test MSE")
plt.xlabel("Number of Features")
plt.show()

# %%
time_lst = []
for i in list(time_stamps.keys()):
    time_lst.append(
        [i, list(time_stamps[i].keys())[0], list(time_stamps[i].values())[0]]
    )

df = pd.DataFrame(time_lst, columns=["group", "column", "val"])

ax = df.pivot("column", "group", "val").plot(kind="bar", figsize=(8, 8))
ax.set(xlabel="model process", ylabel="Time (seconds)")
plt.tight_layout()
plt.show()

# %%
time_stamps = {}
train_mse = []
test_mse = []

df_scaled_x = df_scaled[:, 0 : df_scaled.shape[1] - 1]
df_scaled_y = df_scaled[:, df_scaled.shape[1] - 1]

# partitioning the data
np.random.seed(131)
X_train, X_test, y_train, y_test = train_test_split(
    df_scaled_x,
    df_scaled_y,
    random_state=0,
    test_size=0.30,
)
lnreg = LinearRegression()

# training
start_time = time.time()
lnreg.fit(X_train, y_train)
prediction = lnreg.predict(X_train)
train_mse.append(mean_squared_error(y_train, prediction))
stop_time = time.time()
total = stop_time - start_time
time_stamps[f"train w/ sfs {5}"] = {"train": total}
print(f"training R^2 with {5} features is {r2_score(y_train, prediction)}.")

# testing
start_time = time.time()
prediction = lnreg.predict(X_test)
test_mse.append(mean_squared_error(y_test, prediction))
stop_time = time.time()
total = stop_time - start_time
time_stamps[f"test w/ sfs {5}"] = {"test": total}
print(f"test R^2 with {5} features is {r2_score(y_test, prediction)}.")

np.random.seed(131)
for i in [4, 3, 2, 1]:
    # backward selection
    start_time = time.time()
    lnreg = LinearRegression()
    sfs = SequentialFeatureSelector(
        lnreg,
        n_features_to_select=i,
        direction="backward",
        scoring="neg_mean_squared_error",
    )
    sfs.fit(df_scaled_x, df_scaled_y)
    Xtest = sfs.transform(df_scaled_x)
    stop_time = time.time()
    total = stop_time - start_time
    time_stamps[f"train w/ sfs {i}"] = {"train": total}

    # partitioning the data
    np.random.seed(131)
    X_train, X_test, y_train, y_test = train_test_split(
        Xtest,
        df_scaled_y,
        random_state=0,
        test_size=0.30,
    )

    # training
    start_time = time.time()
    lnreg.fit(X_train, y_train)
    prediction = lnreg.predict(X_train)
    train_mse.append(mean_squared_error(y_train, prediction))
    stop_time = time.time()
    total = stop_time - start_time
    time_stamps[f"train w/ sfs {i}"] = {"train": total}
    print(f"training R^2 with {i} features is {r2_score(y_train, prediction)}.")

    # testing
    start_time = time.time()
    prediction = lnreg.predict(X_test)
    test_mse.append(mean_squared_error(y_test, prediction))
    stop_time = time.time()
    total = stop_time - start_time
    time_stamps[f"test w/ sfs {i}"] = {"test": total}
    print(f"test R^2 with {i} features is {r2_score(y_test, prediction)}.")

plt.plot(
    list(range(1, df_scaled_x.shape[1] + 1)[::-1]),
    train_mse,
    linestyle="--",
    marker="o",
)
plt.ylabel("Train MSE")
plt.xlabel("Number of Features")
plt.show()

plt.plot(
    list(range(1, df_scaled_x.shape[1] + 1)[::-1]), test_mse, linestyle="--", marker="o"
)
plt.ylabel("Test MSE")
plt.xlabel("Number of Features")
plt.show()

# %%
time_lst = []
for i in list(time_stamps.keys()):
    time_lst.append(
        [i, list(time_stamps[i].keys())[0], list(time_stamps[i].values())[0]]
    )

df = pd.DataFrame(time_lst, columns=["group", "column", "val"])

ax = df.pivot("column", "group", "val").plot(kind="bar", figsize=(8, 8))
ax.set(xlabel="model process", ylabel="Time (seconds)")
plt.tight_layout()
plt.show()

# %%
