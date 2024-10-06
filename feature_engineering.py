import numpy as np, pandas as pd
from sklearn.preprocessing import LabelEncoder

class OneHotEncoder:

    def __init__(self, categorical_features) -> None:
        self.categorical_features = categorical_features

    def __call__(self, data) -> list:

        if "venue" in self.categorical_features:
            data["venue"] = data["venue"].astype(str)

        if "trade" in self.categorical_features:
            data["trade"] = data["trade"].astype(str)

        one_hot_categ = pd.get_dummies(data[self.categorical_features])
        data = pd.concat([data, one_hot_categ], axis=1)

        return data, list(one_hot_categ.columns)


def drop_abnormal(train_data, target_data, th=1000):
    x_copy = train_data.copy()
    y_copy = target_data.copy()
    idxs_to_drop_price = np.where(x_copy["price"] > th)[0] # We remove the stocks where abnormal prices have been observed
    idxs_to_drop_bid_size = np.where(x_copy["bid_size"] < 0)[0]
    idxs_to_drop = np.concatenate((idxs_to_drop_price, idxs_to_drop_bid_size))
    stocks_to_drop = [x_copy.iloc[idx]["obs_id"] for idx in idxs_to_drop]
    x_copy = x_copy.query("obs_id != @stocks_to_drop")
    y_copy = y_copy.query("obs_id != @stocks_to_drop")

    print(f"Dropped {len(stocks_to_drop)} abnormal observations with price > {th}.")

    return x_copy, y_copy

def replace_abnormal(train_data, th=1000):
    x_copy = train_data.copy()

    idxs_to_drop_price = np.where(x_copy["price"] > th)[0] # We remove the stocks where abnormal prices have been observed
    idxs_to_drop_bid_size = np.where(x_copy["bid_size"] < 0)[0]
    idxs_to_drop = np.concatenate((idxs_to_drop_price, idxs_to_drop_bid_size))
    
    x_copy["price"] = x_copy["price"].apply(lambda x: x_copy["price"].mean() if x > th else x)
    x_copy["bid_size"] = x_copy["bid_size"].apply(lambda x: x_copy["bid_size"].mean() if x < 0 else x)


    print(f"Replaced {len(idxs_to_drop)} abnormal observations.")

    return x_copy


def symmetric_lorentzian(arr, n=2):
    "Returns the lorentzian(x) if positive, -lorentzian(x) if negative"
    res = arr.copy()
    res[np.where(res>0)], res[np.where(res<0)] = np.log(1+0.5*(res[np.where(res>0)]/15)**n), -np.log(1+0.5*(-res[np.where(res<0)]/15)**n)
    return res


def symmetric_log(arr):
    "Returns a sign-preserving log for large values, tanh for values near 0"
    old_settings = np.seterr(divide='raise', invalid='raise')
    try:
        # Temporarily set settings to ignore invalid values and divide by zero warnings
        with np.errstate(divide='ignore', invalid='ignore'):
            result_pos = np.maximum(np.log(arr*2), np.tanh(arr))
            result_neg = np.maximum(np.log(-arr*2), -np.tanh(arr))
    finally:
        # Restore the original error handling settings
        np.seterr(**old_settings)
    result_pos[np.isinf(result_pos) | np.isnan(result_pos)] = 0
    result_neg[np.isinf(result_neg) | np.isnan(result_neg)] = 0
    return result_pos - result_neg

def add_features(x, embeddings=False, rescaler=symmetric_log):
    
    features = ["price", "bid", "ask"]
    x_copy = x.copy()

    # Log and imb of sizes
    sizes = ["bid_size", "ask_size"]
    
    for size in sizes:
        x_copy[f"log_{size}"] = np.log(x_copy[size]+1)
        features.append(f"log_{size}")
    x_copy['sizes_imb'] = x.eval('(bid_size - ask_size) / (bid_size + ask_size + 1.0e-8)')
    features.append("sizes_imb")

    # Log of abs flux keeping the sign
        
    x_copy["sym_log_flux"] = rescaler(x_copy["flux"])
    features.append("sym_log_flux")

    # Imb of prices
    prices = ["price", "bid", "ask"]

    for i,a in enumerate(prices):
        for j,b in enumerate(prices):
            if i>j:
                # x_copy[f'{a}_{b}_imb'] = x.eval(f'({a} - {b}) / ({a} + {b} + 1.0e-8)')
                x_copy[f'{a}_{b}_imb'] = rescaler(x.eval(f'({a} - {b}) / ({a} + {b} + 1.0e-8)'))
                features.append(f'{a}_{b}_imb')

    for i,a in enumerate(prices):
        for j,b in enumerate(prices):
            for k,c in enumerate(prices):
                if i>j and j>k:
                    max_ = x[[a,b,c]].max(axis=1)
                    min_ = x[[a,b,c]].min(axis=1)
                    mid_ = x[[a,b,c]].sum(axis=1)-min_-max_

                    # x_copy[f'{a}_{b}_{c}_imb2'] = (max_-mid_)/(mid_-min_+1.0e-8)
                    x_copy[f'{a}_{b}_{c}_imb2'] = rescaler((max_-mid_)/(mid_-min_+1.0e-8))
                    features.append(f'{a}_{b}_{c}_imb2')

    # For categorical, either ordinal encoding (for embedding after) or one-hot
    # categorical = ["venue", "action", "side", "trade"]
    categorical = ["venue", "action", "trade"]

    if embeddings:
        enc = LabelEncoder()
        for categ in categorical:
            x_copy[categ] = enc.fit_transform(x_copy[categ])
        features += categorical
    else:
        enc = OneHotEncoder(categorical)
        x_copy, onehot = enc(x_copy)
        features += onehot
    print(f"Feature engineering completed: {len(features)} features.")
    return x_copy[features], features