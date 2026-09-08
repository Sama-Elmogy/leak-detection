
import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

import xgboost as xgb


RANDOM_STATE = 42
TARGET = 'Leak_No_Leak'


df = pd.read_csv(
    '/content/aqua_grid_pfv_vibration_v2.csv'
)


leakage_cols = [
    'Leak_Type',
    'Leak_Type_Eligible'
]

future_target_cols = [
    'Failure_Next_7_Days',
    'Failure_Next_30_Days',
    'Failure_Next_90_Days'
]

id_date_cols = [
    'Pressure_Sensor_ID',
    'Flow_Sensor_ID',
    'Timestamp',
    'Observation_Date',
    'Installation_Date'
]

constant_cols = [
    'Pressure_Unit',
    'Flow_Unit'
]


groups = df['Asset_ID']


drop_cols = (
    leakage_cols
    + future_target_cols
    + id_date_cols
    + constant_cols
    + ['Asset_ID']
)


X = df.drop(
    columns=drop_cols + [TARGET]
)

y = df[TARGET].astype(int)


bool_cols = X.select_dtypes(
    include='bool'
).columns.tolist()

X[bool_cols] = X[bool_cols].astype(int)


cat_cols = (
    X.select_dtypes(
        include='object'
    ).columns.tolist()
    + [
        c for c in X.columns
        if str(X[c].dtype) == 'str'
    ]
)

cat_cols = list(
    dict.fromkeys(cat_cols)
)


num_cols = [
    c for c in X.columns
    if c not in cat_cols
]


gss1 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=RANDOM_STATE
)

trainval_idx, test_idx = next(
    gss1.split(
        X,
        y,
        groups=groups
    )
)


X_trainval = X.iloc[trainval_idx]
X_test = X.iloc[test_idx]

y_trainval = y.iloc[trainval_idx]
y_test = y.iloc[test_idx]

groups_trainval = groups.iloc[
    trainval_idx
]


gss2 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=RANDOM_STATE
)

train_idx, val_idx = next(
    gss2.split(
        X_trainval,
        y_trainval,
        groups=groups_trainval
    )
)


X_train = X_trainval.iloc[train_idx]
X_val = X_trainval.iloc[val_idx]

y_train = y_trainval.iloc[train_idx]
y_val = y_trainval.iloc[val_idx]


preprocessor = ColumnTransformer(
    transformers=[
        (
            'num',
            StandardScaler(),
            num_cols
        ),
        (
            'cat',
            OneHotEncoder(
                handle_unknown='ignore',
                drop='if_binary'
            ),
            cat_cols
        )
    ]
)


X_train_proc = preprocessor.fit_transform(
    X_train
)

X_val_proc = preprocessor.transform(
    X_val
)

X_test_proc = preprocessor.transform(
    X_test
)


pos = y_train.sum()
neg = len(y_train) - pos

scale_pos_weight = neg / pos


model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss',
    random_state=RANDOM_STATE,
    n_jobs=-1
)


model.fit(
    X_train_proc,
    y_train
)
