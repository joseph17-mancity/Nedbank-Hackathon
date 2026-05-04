import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')

def build_037_breakthrough_features():
    print("🚀 Extracting Banking Domain 'Magic' Features...")
    train_label = pd.read_csv('Train.csv')
    test_label = pd.read_csv('Test.csv')
    txn = pd.read_parquet('transactions_features.parquet')
    demo = pd.read_parquet('demographics_clean.parquet')
    
    # --- DIAGNOSTIC CHECK ---
    # If the code fails later, check your console to see these names
    print(f"Demographics columns found: {demo.columns.tolist()}")
    
    test_label['is_test'] = 1
    train_label['is_test'] = 0
    
    txn['TransactionDate'] = pd.to_datetime(txn['TransactionDate'])
    txn['day'] = txn['TransactionDate'].dt.day
    max_dt = pd.Timestamp('2015-10-31')

    # 1. PAYDAY DETECTION
    txn['is_payday'] = txn['day'].isin([25, 26, 27, 28, 29, 30, 31, 1])
    payday_stats = txn.groupby('UniqueID')['is_payday'].agg(['mean', 'sum']).reset_index()
    payday_stats.columns = ['UniqueID', 'payday_ratio', 'payday_total_cnt']

    # 2. HOLIDAY SENSITIVITY
    dec_14_cnt = txn[(txn['TransactionDate'] >= '2014-12-01') & (txn['TransactionDate'] <= '2014-12-31')].groupby('UniqueID').size().reset_index(name='dec_14_vol')
    oct_15_cnt = txn[(txn['TransactionDate'] >= '2015-10-01') & (txn['TransactionDate'] <= '2015-10-31')].groupby('UniqueID').size().reset_index(name='oct_15_vol')

    # 3. TRANSACTION VELOCITY
    m3_date = max_dt - pd.Timedelta(days=90)
    velocity = txn[txn['TransactionDate'] >= m3_date].groupby(['UniqueID', txn['TransactionDate'].dt.to_period('M')]).size().unstack(fill_value=0)
    
    # Trend = last month / average of last 3 months
    velocity_stats = pd.DataFrame({
        'v_mean': velocity.mean(axis=1),
        'v_trend': velocity.iloc[:, -1] / (velocity.mean(axis=1) + 1)
    }).reset_index()

    # 4. AGE CALCULATION
    if 'BirthDate' in demo.columns:
        demo['BirthDate'] = pd.to_datetime(demo['BirthDate'], errors='coerce')
        demo['age'] = 2015 - demo['BirthDate'].dt.year
        demo['age'] = demo['age'].fillna(demo['age'].median()).clip(18, 90)

    # 5. MASTER MERGE
    full_df = pd.concat([train_label, test_label], axis=0).reset_index(drop=True)
    full_df = full_df.merge(demo, on='UniqueID', how='left')
    full_df = full_df.merge(payday_stats, on='UniqueID', how='left')
    full_df = full_df.merge(dec_14_cnt, on='UniqueID', how='left')
    full_df = full_df.merge(oct_15_cnt, on='UniqueID', how='left') # Fixed variable name
    full_df = full_df.merge(velocity_stats, on='UniqueID', how='left')

    base_agg = txn.groupby('UniqueID').size().reset_index(name='lifetime_cnt')
    full_df = full_df.merge(base_agg, on='UniqueID', how='left')

    # --- THE FIX: DYNAMIC CATEGORICAL ENCODING ---
    # Automatically find all text-based columns and factorize them
    cat_cols = full_df.select_dtypes(include=['object', 'string', 'category']).columns
    print(f"Encoding Categoricals: {[c for c in cat_cols if c != 'UniqueID']}")
    
    for col in cat_cols:
        if col not in ['UniqueID', 'BirthDate']:
            full_df[col], _ = pd.factorize(full_df[col].astype(str))

    features = [c for c in full_df.columns if c not in ['UniqueID', 'BirthDate', 'next_3m_txn_count', 'is_test', 'month']]
    full_df[features] = full_df[features].fillna(0)
    
    train_out = full_df[full_df['is_test'] == 0].copy()
    test_out = full_df[full_df['is_test'] == 1].copy()
    
    return train_out, test_out, features

# --- TRAINING PIPELINE ---
train, test, features = build_037_breakthrough_features()

X = train[features]
y = np.log1p(train['next_3m_txn_count'])
X_test = test[features]

# Using 10 Folds for the 0.37 precision
kf = KFold(n_splits=10, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.003,
    'num_leaves': 18,
    'max_depth': 5,
    'lambda_l2': 80, 
    'feature_fraction': 0.6,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'min_data_in_leaf': 150,
    'verbose': -1
}

print(f"\nTraining on {len(features)} features...")

for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y)):
    X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    model = lgb.LGBMRegressor(**params, n_estimators=8000)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(300)])
    
    oof_preds[val_idx] = model.predict(X_val)
    test_preds += model.predict(X_test) / 10

cv_score = np.sqrt(mean_squared_error(y, oof_preds))
print(f"\n✨ FINAL BREAKTHROUGH CV RMSLE: {cv_score:.5f}")

# Finalizing output
final_submit = np.expm1(test_preds).clip(0.5, 450)
submission = test[['UniqueID']].copy()
submission['next_3m_txn_count'] = final_submit
submission.to_csv('sub_037_breakthrough_v2.csv', index=False)
print("🏁 File saved: sub_037_breakthrough_v2.csv")