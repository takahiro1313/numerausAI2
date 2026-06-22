"""数値AI ハンズオン — 演習用（参加者に配布）。

「今月の活動 → 来月の成約数」を予測するAIを、アプリ内の編集セルに書いて動かす。
ポイント：AIに見せる「手がかり」を、`問い合わせ数`・`面談数` などの**活動の名前で**書く。

このアプリは **Streamlit Community Cloud にデプロイして使う**（計算はサーバ側）。
参加者は配布URLを開き、🟢の編集セルだけ触る。

▼ ローカル確認:
    pip install -r requirements.txt
    python make_dummy.py
    streamlit run app_ensyu.py
正解は app_seikai.py。
"""

import os
import streamlit as st
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

# =========================================================================
# 🔒 裏方（データ下ごしらえ・安全実行）。さわらない。
# =========================================================================
ACT_COLS = ["問い合わせ数", "架電数", "面談数", "案件提案数"]  # 手がかりの候補（活動）
TARGET = "翌月の成約数"
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "activities.csv")
PRED_PATH = os.path.join(os.path.dirname(__file__), "data", "予測結果.csv")  # まーぴーが読む
VAL_MONTHS = 3  # 終盤3ヶ月を「未来（検証）」に回す

SAFE_BUILTINS = {
    "True": True, "False": False, "None": None,
    "len": len, "range": range, "str": str, "int": int, "float": float,
    "list": list, "dict": dict, "print": print, "round": round,
}


def load_data():
    return pd.read_csv(DATA_PATH)


def build_features(df):
    """各担当の「今月の活動」に、答え＝翌月の成約数（成約数を1ヶ月先にズラす）を付ける。"""
    df = df.sort_values(["営業担当", "月"]).copy()
    df[TARGET] = df.groupby("営業担当", group_keys=False)["成約数"].shift(-1)
    return df.dropna(subset=[TARGET]).reset_index(drop=True)


def split_data(feat):
    """過去の月で学習し、未来（終盤の月）で答え合わせ（時間で分ける）。"""
    cutoff = feat["月"].max() - VAL_MONTHS
    return feat[feat["月"] <= cutoff], feat[feat["月"] > cutoff], cutoff


def latest_row(df, rep):
    """選んだ担当の“最新月”の活動（1行）と、直近3ヶ月の表示用。"""
    r = df[df["営業担当"] == rep].sort_values("月")
    return r.tail(1), r.tail(3)


def score(model, X, y):
    return float(r2_score(y, model.predict(X)))


def save_all_predictions(model, df, feature_cols):
    """学習済みモデルで全担当の来月を予測し、まーぴー用CSVに保存（🔒 連結）。"""
    rows = []
    for rep in sorted(df["営業担当"].unique()):
        last, _ = latest_row(df, rep)
        rows.append({"営業担当": rep,
                     "予測_翌月成約数": round(float(model.predict(last[feature_cols])[0]), 1)})
    pd.DataFrame(rows).to_csv(PRED_PATH, index=False, encoding="utf-8-sig")


def _exec_user(code, ns):
    """編集セルのコードを限定名前空間で実行。成功=None／失敗=例外。"""
    try:
        exec(code, {"__builtins__": SAFE_BUILTINS}, ns)
        return None
    except Exception as e:  # noqa: BLE001  事故防止のため全例外を画面表示
        return e


# --- コールバック（学習・推論） ------------------------------------------
def _do_learn():
    """Step2 の編集セル（手がかり＋モデル＋fit）を実行して学習する。"""
    feat = build_features(load_data())
    train_df, val_df, _ = split_data(feat)
    ns = {"XGBRegressor": XGBRegressor, "LinearRegression": LinearRegression,
          "訓練データ": train_df}
    err = _exec_user(st.session_state.learn_code, ns)
    cols = ns.get("手がかり")
    if err is None and ns.get("model") is None:
        err = NameError("model が作られていません。`model = ...` を書いてね。")
    if err is None and not cols:
        err = NameError("手がかり が空です。活動の列名（例: \"案件提案数\"）を書いてね。")
    if err is not None:
        st.session_state.fit_err = str(err)
        st.session_state.model = None
        return
    model, cols = ns["model"], list(cols)
    st.session_state.fit_err = None
    st.session_state.model = model
    st.session_state.feature_cols = cols
    st.session_state.train_score = score(model, train_df[cols], train_df[TARGET])
    st.session_state.val_score = score(model, val_df[cols], val_df[TARGET])
    st.session_state.prediction = None
    save_all_predictions(model, load_data(), cols)  # 🔗 まーぴー連結用


def _do_predict():
    if st.session_state.get("model") is None:
        st.session_state.pred_err = "先に「学習」してください。"
        st.session_state.prediction = None
        return
    cols = st.session_state.feature_cols
    last, _ = latest_row(load_data(), st.session_state.rep_select)
    ns = {"model": st.session_state.model, "来月予測用": last[cols]}
    err = _exec_user(st.session_state.predict_code, ns)
    if err is not None:
        st.session_state.pred_err = str(err)
        st.session_state.prediction = None
        return
    pred = ns.get("pred")
    if pred is None:
        st.session_state.pred_err = "pred に予測結果が入っていません。書き方を見直してください。"
        st.session_state.prediction = None
        return
    st.session_state.pred_err = None
    st.session_state.prediction = float(pred[0])


# =========================================================================
# 画面（🔒 描画。🟢 は編集セル）
# =========================================================================
def render_app(learn_snippet: str, predict_snippet: str, mode_label: str):
    st.set_page_config(page_title="数値AI｜来月の成約予測", page_icon="📈", layout="wide")
    st.title("📈 来月の成約数を予測するAI")
    st.caption(f"今月の活動から来月の成約を読む（モード：{mode_label}）")

    st.session_state.setdefault("learn_code", learn_snippet)
    st.session_state.setdefault("predict_code", predict_snippet)
    st.session_state.setdefault("model", None)
    st.session_state.setdefault("prediction", None)

    st.markdown(
        """
### このハンズオンでやること
IT編では「**人がルール(if)を書いて**」案内を出しました。今回は「**AIがデータからルールを学んで**」予測します。

| | ステップ | あなたの操作 |
|---|---|---|
| 🔒 | Step1 データを見る | 見るだけ（手がかり＝活動／答え＝翌月の成約） |
| 🟢 | Step2 手がかりを書いて学習 | ✍️ 見せる活動を書く＋モデル → 🎓 学習 |
| 🟢 | Step3 推論する | 担当を選ぶ → ✍️ predict → 🔮 推論 |

> 🟢＝あなたが書くゾーン／🔒＝裏方（さわらない）。
"""
    )

    df = load_data()
    feat = build_features(df)

    # --- 🔒 Step1: データを見る（全件・スクロール）---
    st.divider()
    st.subheader("🔒 Step1：データを見る（営業担当 × 月・全件）")
    st.dataframe(df, width="stretch", height=360, hide_index=True)
    _, _, cutoff = split_data(feat)
    st.caption(
        f"**手がかり(X)＝今月の活動**（{', '.join(ACT_COLS)}）。**答え(y)＝翌月の成約数**。"
        f" 全{len(df)}行（{df['営業担当'].nunique()}担当×{int(df['月'].max())}ヶ月）。"
        f" 学習用 {len(feat)} 行（過去{int(cutoff)}ヶ月で学習→残りで検証）。"
    )

    # --- 🟢 Step2: 手がかりを書いて学習（モデル作成＋fit を1セル）---
    st.divider()
    st.subheader("🟢 Step2：手がかりを書いて学習する")
    st.text_area("学習コード（手がかり → モデル → 学習）", key="learn_code", height=200)
    st.button("🎓 学習", type="primary", on_click=_do_learn)
    _cols_disp = "、".join(f"「{c}」" for c in ACT_COLS)
    st.caption(
        f"使える活動の列名：{_cols_disp}。 学習用データは `訓練データ`、答えの列は「翌月の成約数」。"
        " 木の数字を大きく（例 300, 6）すると過学習しやすい。"
    )

    if st.session_state.get("fit_err"):
        st.error(f"⚠️ エラー: {st.session_state.fit_err}\n\nStep2 の書き方を見直してください。")
    elif st.session_state.get("model") is not None:
        st.success(f"学習できました！（手がかり：{', '.join(st.session_state.feature_cols)}）")
        s1, s2 = st.columns(2)
        s1.metric("訓練スコア（train R²）", f"{st.session_state.train_score:.3f}")
        s2.metric("検証スコア（val R²）", f"{st.session_state.val_score:.3f}")
        if st.session_state.train_score - st.session_state.val_score > 0.2:
            st.warning("訓練◎なのに検証✕＝**過学習**ぎみ。モデルの数字を下げてみましょう。")
        else:
            st.info("訓練と検証のスコアが近い＝バランス良好です。")
        if hasattr(st.session_state.model, "feature_importances_"):
            imp = pd.Series(
                st.session_state.model.feature_importances_, index=st.session_state.feature_cols
            ).sort_values(ascending=False)
            st.write("**どの活動が翌月の成約に効いた？（重要度）**")
            st.bar_chart(imp)

    # --- 🟢 Step3: 推論（predict）---
    st.divider()
    st.subheader("🟢 Step3：推論する（predict を書く）")
    st.selectbox("予測する営業担当を選ぶ", sorted(df["営業担当"].unique()), key="rep_select")
    _, last3 = latest_row(df, st.session_state.rep_select)
    st.caption(f"{st.session_state.rep_select} の直近の活動（最新月＝来月予測用）:")
    st.dataframe(last3[["月", *ACT_COLS]], width="stretch", hide_index=True)
    st.text_area("推論コード", key="predict_code", height=100)
    st.button("🔮 推論", type="primary", on_click=_do_predict)

    if st.session_state.get("pred_err"):
        st.error(f"⚠️ {st.session_state.pred_err}")
    elif st.session_state.get("prediction") is not None:
        st.metric(f"🔮 {st.session_state.rep_select} の来月の予測成約数",
                  f"{st.session_state.prediction:.1f} 件")
        st.caption("担当や手がかり・モデルの数字を変えて、予測がどう動くか試してみましょう。")


# 🟢 Step2 学習（手がかり＋モデル＋fit）の編集セル初期値（演習用）
LEARN_SNIPPET = '''# 🟢 ===== ここを編集 =====
# ① AIに見せる「手がかり」を書く（どの活動を見せる？ 列名を並べる）
手がかり = ["___"]   # 例: "問い合わせ数", "架電数", "面談数", "案件提案数"

# ② モデルを作って、手がかりを見せて「翌月の成約数」を当てるよう学習させる
model = XGBRegressor(n_estimators=40, max_depth=2, learning_rate=0.1, random_state=42)
model.fit(訓練データ[手がかり], 訓練データ["翌月の成約数"])
# 🟢 ===== 編集はここまで =====
'''

# 🟢 Step3 推論（predict）の編集セル初期値（演習用・空欄）
PREDICT_SNIPPET = '''# 🟢 ===== ここを編集 =====
pred = model.predict(___)   # ← ___ に「来月予測用」を入れる
# 🟢 ===== 編集はここまで =====
'''

render_app(LEARN_SNIPPET, PREDICT_SNIPPET, mode_label="演習用")
