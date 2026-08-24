import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from src.utils.metrics import evaluate

MODELS = {}


def _register(name):
    def deco(fn):
        MODELS[name] = fn
        return fn
    return deco


@_register("random_forest")
def make_random_forest(n_estimators=500, max_features=0.5, random_state=42, **kw):
    return RandomForestRegressor(
        n_estimators=n_estimators, max_features=max_features,
        n_jobs=-1, random_state=random_state, **kw,
    )


@_register("svr")
def make_svr(C=1.0, epsilon=0.1, **kw):
    return Pipeline([
        ("scale", StandardScaler()),
        ("svr", SVR(C=C, epsilon=epsilon, **kw)),
    ])


@_register("xgboost")
def make_xgboost(n_estimators=800, learning_rate=0.05, max_depth=6,
                 subsample=0.8, colsample_bytree=0.8, random_state=42, **kw):
    import xgboost as xgb
    return xgb.XGBRegressor(
        n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
        subsample=subsample, colsample_bytree=colsample_bytree,
        random_state=random_state, verbosity=0, **kw,
    )


@_register("lightgbm")
def make_lightgbm(n_estimators=800, learning_rate=0.05, num_leaves=63,
                  subsample=0.8, colsample_bytree=0.8, random_state=42, **kw):
    import lightgbm as lgb
    return lgb.LGBMRegressor(
        n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=num_leaves,
        subsample=subsample, colsample_bytree=colsample_bytree,
        random_state=random_state, verbose=-1, **kw,
    )


@_register("catboost")
def make_catboost(iterations=800, learning_rate=0.05, depth=6, random_state=42, **kw):
    from catboost import CatBoostRegressor
    return CatBoostRegressor(
        iterations=iterations, learning_rate=learning_rate, depth=depth,
        random_seed=random_state, verbose=0, **kw,
    )


@_register("mlp")
def make_mlp(hidden_layer_sizes=(256, 128), alpha=1e-3, random_state=42, **kw):
    return Pipeline([
        ("scale", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes, alpha=alpha,
            max_iter=400, early_stopping=True, random_state=random_state, **kw,
        )),
    ])


def get_model(name, **overrides):
    if name not in MODELS:
        raise ValueError(f"Unknown model: {name}. Available: {sorted(MODELS)}")
    return MODELS[name](**overrides)


def train_and_evaluate(name, X_train, y_train, X_val, y_val, X_test, y_test,
                       sample_weight=None, **overrides):
    """Train model, return {train,val,test} metric dicts and the fitted model."""
    model = get_model(name, **overrides)
    if sample_weight is not None:
        model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    return {
        "model": name,
        "train": evaluate(y_train, train_pred),
        "val": evaluate(y_val, val_pred),
        "test": evaluate(y_test, test_pred),
    }, model


def compute_zero_weights(y_train, metal_weight: float = 1.0, semi_weight: float = 3.0):
    """Upweight semiconducting/insulating (non-zero) samples to counter metallic bias."""
    y = np.asarray(y_train, dtype=float)
    w = np.ones(len(y))
    w[y != 0] = semi_weight / metal_weight
    w[y == 0] = 1.0
    return w

