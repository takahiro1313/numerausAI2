"""数値AIハンズオン用のダミー時系列CSVを生成する（SFDCイメージ）。

営業担当 × 月 のパネルデータ。各月の「先行活動」から、AIが
「翌月の成約数」を予測できるように、活動→成約にリードタイム（ラグ）を持たせる。
結果系（契約数）は入れない（自明回避）。実データは一切使わない（要件 §4）。

担当者名はデモで読み上げやすい実名にする（要件・ユーザー指定）:
  - 固定名（成約 多い）: 濵田隼人 / 大林ふみ / 佐藤周治 / 兵頭大志 / 川端須開
  - 固定名（成約 少ない）: 斎藤貴大 / 黒須ひろとし / 谷本りゅうき
  - 残りはランダムな氏名
多寡は「面談・案件提案のベース水準」を上下させて作る（架電は据え置き＝架電は効かない指標のまま）。

生データ列: 営業担当, 月, 問い合わせ数, 架電数, 面談数, 案件提案数, 成約数

実行:  python make_dummy.py  → data/activities.csv
"""

import os
import numpy as np
import pandas as pd

N_REPS = 75          # 営業担当の人数（固定8名 ＋ ランダム67名）
N_MONTHS = 12        # 月数（1〜12月。暦の月に合わせる）
SEED = 42

ACT_COLS = ["問い合わせ数", "架電数", "面談数", "案件提案数"]
TARGET_COL = "成約数"

# 固定名（デモで使う）。成約が「多い人」「少ない人」を仕込む。
HIGH_NAMES = ["濵田隼人", "大林ふみ", "佐藤周治", "兵頭大志", "川端須開"]
LOW_NAMES = ["斎藤貴大", "黒須ひろとし", "谷本りゅうき"]

# ランダム氏名の素（残り67名分。固定名と重複しないよう後で除外）
SURNAMES = [
    "鈴木", "高橋", "田中", "伊藤", "渡辺", "山本", "中村", "小林", "加藤", "吉田",
    "山田", "佐々木", "山口", "松本", "井上", "木村", "清水", "山崎", "森", "池田",
    "橋本", "阿部", "石川", "前田", "藤田", "後藤", "岡田", "村上", "近藤", "坂本",
    "遠藤", "青木", "藤井", "西村", "福田", "太田", "三浦", "岡本", "松田", "中島",
]
GIVEN = [
    "翔太", "蓮", "陽斗", "大和", "悠真", "湊", "拓海", "健太", "駿", "直樹",
    "和也", "涼介", "亮", "美咲", "さくら", "陽菜", "葵", "結衣", "彩", "遥",
    "奈々", "裕子", "真由", "彩花", "千尋", "理恵", "麻衣", "智也", "京子", "桃子",
]

# 各活動の平均・ばらつき（担当ごとに独立に baseline を引く）
ACT_MEAN_SD = {
    "問い合わせ数": (45, 10),
    "架電数": (85, 20),
    "面談数": (28, 7),
    "案件提案数": (9, 3),
}
# 成約の多寡をつくる「面談・案件提案」のベース倍率（架電・問い合わせは据え置き）
DRIVER_COLS = {"面談数", "案件提案数"}
PERF_MULT = {**{n: 1.4 for n in HIGH_NAMES}, **{n: 0.6 for n in LOW_NAMES}}  # 既定 1.0


def build_rep_names(rng) -> list:
    """固定8名 ＋ ランダム67名（重複なし）の担当者名リストを作る。"""
    fixed = HIGH_NAMES + LOW_NAMES
    pool = [s + g for s in SURNAMES for g in GIVEN if (s + g) not in fixed]
    rng.shuffle(pool)
    randoms = list(dict.fromkeys(pool))[: N_REPS - len(fixed)]
    return fixed + randoms  # 固定名を先頭に（プレビューで実名が見える）


def main():
    rng = np.random.default_rng(SEED)
    rep_names = build_rep_names(rng)
    rows = []

    for name in rep_names:
        mult = PERF_MULT.get(name, 1.0)
        # 面談・案件提案だけ倍率を掛ける（架電は据え置き＝効かない指標のまま）
        base = {
            c: float(max(1.0, rng.normal(m * (mult if c in DRIVER_COLS else 1.0), s)))
            for c, (m, s) in ACT_MEAN_SD.items()
        }
        history = []

        for month in range(1, N_MONTHS + 1):
            act = {c: int(max(0, rng.poisson(base[c]))) for c in ACT_COLS}
            history.append(act)

            # 成約数(今月) = 「過去の」先行活動だけから決まる（リードタイムあり）。
            def get(idx, col):
                return history[idx][col] if -len(history) <= idx < len(history) else 0

            # 「今月の活動 → 来月の成約」（1ヶ月リード）。get(-2)=先月の活動。
            lead = (
                0.40 * get(-2, "案件提案数")      # 先月の提案が最重要
                + 0.09 * get(-2, "面談数")        # 先月の面談
                + 0.012 * get(-2, "問い合わせ数")  # 先月の問い合わせは弱く効く
                # 架電数は意図的に係数ゼロ（効かない指標）
            )
            noise = rng.normal(0, 1.0)
            seiyaku = int(np.clip(round(lead + noise), 0, None))

            rows.append({"営業担当": name, "月": month, **act, TARGET_COL: seiyaku})

    df = pd.DataFrame(rows, columns=["営業担当", "月", *ACT_COLS, TARGET_COL])

    os.makedirs("data", exist_ok=True)
    out = os.path.join("data", "activities.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"生成しました: {out}（{len(df)}行＝{N_REPS}担当 × {N_MONTHS}ヶ月）")
    avg = df.groupby("営業担当")[TARGET_COL].mean()
    print("成約 多い固定名:", {n: round(avg[n], 1) for n in HIGH_NAMES})
    print("成約 少ない固定名:", {n: round(avg[n], 1) for n in LOW_NAMES})


if __name__ == "__main__":
    main()
