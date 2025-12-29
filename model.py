import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import joblib

def coerce_numeric_df(X: pd.DataFrame) -> pd.DataFrame:
    Xc = X.copy()
    for c in Xc.columns:
        if Xc[c].dtype == 'O':
            Xc[c] = Xc[c].apply(lambda v: float(np.mean(np.asarray(v, dtype=float))) if not np.isscalar(v) else float(v))
    Xc = Xc.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    return Xc

df = pd.read_csv('features_table_advanced.csv')
X = df.drop(columns=['seq_id','fault_type','line_id','segment_id'])
y_fault = df['fault_type']
y_line  = df['line_id']
X = coerce_numeric_df(X)

X_train_full, X_test, y_fault_train_full, y_fault_test, y_line_train_full, y_line_test = train_test_split(
    X, y_fault, y_line, test_size=0.15, random_state=42, stratify=y_fault)

X_train, X_val, y_fault_train, y_fault_val, y_line_train, y_line_val = train_test_split(
    X_train_full, y_fault_train_full, y_line_train_full, test_size=0.1765, random_state=42, stratify=y_fault_train_full)

# Fault model (CatBoost)
fault_params = dict(
    iterations=3000, depth=7, learning_rate=0.05, l2_leaf_reg=14.0,
    loss_function='MultiClass', random_seed=42, verbose=False,
    od_type='Iter', od_wait=120, use_best_model=True
    # task_type='GPU'
)
fault_model = CatBoostClassifier(**fault_params)
fault_model.fit(X_train, y_fault_train, eval_set=(X_val, y_fault_val))

# Line CatBoost
line_cat_params = dict(
    iterations=3000, depth=6, learning_rate=0.045, l2_leaf_reg=16.0,
    loss_function='MultiClass', random_seed=42, verbose=False,
    od_type='Iter', od_wait=120, use_best_model=True
)
line_cat = CatBoostClassifier(**line_cat_params)
line_cat.fit(X_train, y_line_train, eval_set=(X_val, y_line_val))

# Line XGBoost with LabelEncoder (fix class handling)
le_line = LabelEncoder()
y_line_train_enc = le_line.fit_transform(y_line_train)
y_line_val_enc   = le_line.transform(y_line_val)
y_line_test_enc  = le_line.transform(y_line_test)
n_classes = len(le_line.classes_)

line_xgb = Pipeline([
    ('scaler', StandardScaler(with_mean=True, with_std=True)),
    ('xgb', XGBClassifier(
        n_estimators=800, max_depth=6, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
        reg_lambda=3.0, reg_alpha=0.5, objective='multi:softprob', num_class=n_classes,
        random_state=42, tree_method='hist'
    ))
])
line_xgb.fit(X_train, y_line_train_enc)

def soft_vote_proba(X):
    p1 = line_cat.predict_proba(X)
    p2 = line_xgb.predict_proba(X)
    # reorder CatBoost proba to LabelEncoder class order
    cb_classes = list(line_cat.classes_)
    xgb_classes = list(le_line.classes_)
    reorder = [cb_classes.index(c) for c in xgb_classes]
    p1_re = p1[:, reorder]
    return 0.6*p1_re + 0.4*p2

def soft_vote_predict(X):
    proba = soft_vote_proba(X)
    idx = np.argmax(proba, axis=1)
    return le_line.inverse_transform(idx)

# Train acc
fault_train_acc = accuracy_score(y_fault_train, fault_model.predict(X_train))
line_train_pred = soft_vote_predict(X_train)
line_train_acc  = accuracy_score(y_line_train, line_train_pred)
print(f"[TRAIN] Fault acc: {fault_train_acc*100:.2f}%")
print(f"[TRAIN] Line  acc: {line_train_acc*100:.2f}%")

# Test acc
fault_pred = fault_model.predict(X_test).ravel()
line_pred  = soft_vote_predict(X_test)
fault_acc = accuracy_score(y_fault_test, fault_pred)
line_acc  = accuracy_score(y_line_test,  line_pred)
print("\n=== Fault Type Classification (Test) ===")
print(classification_report(y_fault_test, fault_pred))
print(f"[ACC] Fault classifier accuracy (test): {fault_acc*100:.2f}%")
print("\n=== Line Localization Classification (Test) ===")
print(classification_report(y_line_test, line_pred))
print(f"[ACC] Line classifier accuracy (test): {line_acc*100:.2f}%")

# 5-fold CV on line ensemble
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
for tr_idx, va_idx in skf.split(X_train_full, y_line_train_full):
    Xt, Xv = X_train_full.iloc[tr_idx], X_train_full.iloc[va_idx]
    yt, yv = y_line_train_full.iloc[tr_idx], y_line_train_full.iloc[va_idx]
    # CatBoost per-fold
    lc = CatBoostClassifier(**line_cat_params)
    lc.fit(Xt, yt, eval_set=(Xv, yv))
    # XGB per-fold with encoding
    le_cv = LabelEncoder()
    yt_enc = le_cv.fit_transform(yt)
    yv_enc = le_cv.transform(yv)
    n_cls_cv = len(le_cv.classes_)
    lx = Pipeline([
        ('scaler', StandardScaler()),
        ('xgb', XGBClassifier(
            n_estimators=600, max_depth=6, learning_rate=0.06, subsample=0.9, colsample_bytree=0.9,
            reg_lambda=3.0, reg_alpha=0.5, objective='multi:softprob', num_class=n_cls_cv,
            random_state=42, tree_method='hist'
        ))
    ])
    lx.fit(Xt, yt_enc)
    # vote with aligned class order
    p1 = lc.predict_proba(Xv)
    reorder = [list(lc.classes_).index(c) for c in le_cv.classes_]
    p1_re = p1[:, reorder]
    p2 = lx.predict_proba(Xv)
    p = 0.6*p1_re + 0.4*p2
    pred = le_cv.inverse_transform(np.argmax(p, axis=1))
    cv_scores.append(accuracy_score(yv, pred))
print(f"[CV] Line localization 5-fold mean: {np.mean(cv_scores)*100:.2f}% ± {np.std(cv_scores)*100:.2f}%")

# Shifted-set evaluation
try:
    df_shift = pd.read_csv('features_table_shifted.csv')
    Xs = df_shift.drop(columns=['seq_id','fault_type','line_id','segment_id'])
    ys_fault = df_shift['fault_type']; ys_line = df_shift['line_id']
    def coerce_numeric_df_local(Xin):
        Xin = Xin.copy()
        for c in Xin.columns:
            if Xin[c].dtype == 'O':
                Xin[c] = Xin[c].apply(lambda v: float(np.mean(np.asarray(v, dtype=float))) if not np.isscalar(v) else float(v))
        return Xin.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    Xs = coerce_numeric_df_local(Xs)
    pf = fault_model.predict(Xs).ravel()
    # soft vote on shifted
    p1s = line_cat.predict_proba(Xs)
    reorder = [list(line_cat.classes_).index(c) for c in le_line.classes_]
    p1s_re = p1s[:, reorder]
    p2s = line_xgb.predict_proba(Xs)
    ps = 0.6*p1s_re + 0.4*p2s
    pls = le_line.inverse_transform(np.argmax(ps, axis=1))
    from sklearn.metrics import accuracy_score
    print(f"\n[SHIFTED] Fault acc: {accuracy_score(ys_fault, pf)*100:.2f}%")
    print(f"[SHIFTED] Line  acc: {accuracy_score(ys_line,  pls)*100:.2f}%")
except Exception as e:
    print("[WARN] Shifted set not available or failed]:", e)

joblib.dump(fault_model, 'cat_model_fault.pkl')
joblib.dump(line_cat,    'cat_model_line_cat.pkl')
joblib.dump(line_xgb,    'xgb_pipeline_line.pkl')
joblib.dump(le_line,     'label_encoder_line.pkl')
print("[OK] Saved models: cat_model_fault.pkl, cat_model_line_cat.pkl, xgb_pipeline_line.pkl, label_encoder_line.pkl")
