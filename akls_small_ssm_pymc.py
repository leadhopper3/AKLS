#!/usr/bin/env python3
# このファイルを直接実行する際に、環境中の Python 3 を使うことを示す。
"""
PyMC implementation of the AKLS small state-space model.

This script expects the normalized AKLS master table created earlier, preferably
`master_data.csv` or the `Master_Data` sheet from `AKLS_master_dataset.xlsx`.

Model:
  F(t) is a monotone latent membrane-fouling state.
  TMP is observed as a sum of run baseline, known back-pressure control effect,
  and a smooth fouling burden.
  Transmission is observed on the logit scale and decreases with fouling burden.

Example:
  python akls_small_ssm_pymc.py \
    --input AKLS_master_dataset.xlsx \
    --output-dir akls_pymc_results \
    --draws 1000 --tune 1000 --chains 4
"""

# Python 3 の型注釈を、実行時ではなく遅延評価できるようにする。
from __future__ import annotations

# コマンドライン引数を定義・解析する標準ライブラリ。
import argparse
# OS に依存しないファイルパス操作を行う標準ライブラリ。
from pathlib import Path

# 数値配列・ベクトル演算を扱うライブラリ。
import numpy as np
# 表形式データの読み込み・整形を扱うライブラリ。
import pandas as pd


def logit_np(x: np.ndarray) -> np.ndarray:
    # 確率を、実数全体に写像する logit 変換の前に有限範囲へ制限する。
    x = np.clip(x, 1e-6, 1 - 1e-6)
    # p/(1-p) の対数を返す。0 や 1 を避けることで無限大を防ぐ。
    return np.log(x / (1 - x))


def invlogit_np(x: np.ndarray) -> np.ndarray:
    # logit スケールの値を 0～1 の確率へ戻す逆ロジット変換。
    return 1 / (1 + np.exp(-x))


def read_master(path: Path) -> pd.DataFrame:
    # Excel と CSV で読み込み方法を切り替える。
    if path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        # Excel では、正規化済みデータが入った Master_Data シートを読む。
        return pd.read_excel(path, sheet_name="Master_Data")
    # それ以外の拡張子は CSV として読む。
    return pd.read_csv(path)


def prepare_model_data(df: pd.DataFrame) -> dict:
    # 時刻と TMP が存在する行だけを、モデルの基本観測行として残す。
    keep = df["elapsed_h"].notna() & df["tmp_kpa"].notna()
    # 元データを変更しないようにコピーして以降の加工対象にする。
    df = df.loc[keep].copy()
    # run ごとに時刻順へ並べ、後続処理で使う整数インデックスを振り直す。
    df = df.sort_values(["run_id", "elapsed_h", "row_number"]).reset_index(drop=True)

    # 累積ろ過量を run 内で線形補間し、補間不能な値は 0 とする。
    df["cumulative_filtrate_l_m2_interp"] = (
        # run ごとに独立した時系列として補間する。
        df.groupby("run_id")["cumulative_filtrate_l_m2"]
        # 前後の観測を使って欠測値を埋める。
        .transform(lambda s: s.interpolate(limit_direction="both"))
        # run 内に観測が全くない場合の既定値。
        .fillna(0.0)
    )
    # 背圧も run 内で補間し、欠測値は 0 とする。
    df["back_pressure_kpa_interp"] = (
        df.groupby("run_id")["back_pressure_kpa"]
        .transform(lambda s: s.interpolate(limit_direction="both"))
        .fillna(0.0)
    )
    # 負の背圧を、TMP を増加させる正の負荷量へ変換する。
    df["back_pressure_load"] = np.maximum(-df["back_pressure_kpa_interp"].to_numpy(dtype=float), 0.0)
    # 供給液抗体濃度を run 内で補間する。これは filtrate 予測に使う。
    df["feed_antibody_g_l_interp"] = (
        df.groupby("run_id")["feed_antibody_g_l"]
        .transform(lambda s: s.interpolate(limit_direction="both"))
    )

    # Transmission は logit スケールでモデル化するため、まず観測値を 0～1 に変換する。
    trans_obs_mask = df["transmission_pct"].notna().to_numpy()
    # 欠測でない Transmission(%) だけを取り出し、端点を避けて確率にする。
    transmission_frac = np.clip(df.loc[trans_obs_mask, "transmission_pct"].to_numpy(dtype=float) / 100.0, 0.005, 0.995)
    # Transmission の観測値を logit スケールへ変換する。
    transmission_logit_obs = logit_np(transmission_frac)

    # 条件順を固定した run の一覧を作る。LMH と shear rate の順で並べる。
    runs = sorted(
        df["run_id"].unique(),
        key=lambda r: (
            int(df.loc[df["run_id"] == r, "lmh_group"].iloc[0]),
            int(df.loc[df["run_id"] == r, "shear_rate_lmh"].iloc[0]),
        ),
    )
    # 文字列などの run ID を、PyMC の配列添字に使える 0 始まり整数へ対応付ける。
    run_index = {r: i for i, r in enumerate(runs)}
    # 各観測行へ対応する run の整数インデックスを追加する。
    df["run_idx"] = df["run_id"].map(run_index).astype(int)

    # 遷移行を作り、PyMC が scan なしで F = A @ transition_increments を計算できるようにする。
    transitions = []
    # 旧実装との互換・メモ用の空配列。現在の計算では使用しない。
    obs_to_transition = np.zeros((len(df), 0), dtype=float)
    # これまでに生成した遷移数を数える。
    transition_count = 0
    # 全 run を一つずつ処理する。
    for run in runs:
        # 現在の run に属する観測行の整数位置を取得する。
        idx = df.index[df["run_id"] == run].to_numpy()
        # 経過時間を NumPy の浮動小数配列として取得する。
        t = df.loc[idx, "elapsed_h"].to_numpy(dtype=float)
        # 累積ろ過量を 500 でスケーリングして数値の大きさを抑える。
        cum = df.loc[idx, "cumulative_filtrate_l_m2_interp"].to_numpy(dtype=float) / 500.0
        # run の整数インデックスを取得する。
        run_idx = run_index[run]
        # run における shear rate を取得する。
        shear = float(df.loc[idx, "shear_rate_lmh"].iloc[0])
        # LMH group が 10 なら高負荷フラグを 1、それ以外を 0 とする。
        high_load = 1.0 if int(df.loc[idx, "lmh_group"].iloc[0]) == 10 else 0.0
        # run 内で隣り合う観測間の遷移を作る。
        for j in range(1, len(idx)):
            # 時間差を日単位に直し、時刻が逆転している場合は 0 にする。
            dt_days = max(t[j] - t[j - 1], 0.0) / 24.0
            # 区間中央の累積ろ過量を遷移の代表値とする。
            cum_mid = 0.5 * (cum[j] + cum[j - 1])
            # 1 つの遷移に必要な説明変数と観測位置を記録する。
            transitions.append(
                {
                    # 遷移が属する run の識別子。
                    "run": run,
                    # run の整数インデックス。
                    "run_idx": run_idx,
                    # 遷移の始点となる観測位置。
                    "from_obs": int(idx[j - 1]),
                    # 遷移の終点となる観測位置。
                    "to_obs": int(idx[j]),
                    # 遷移時間（日）。
                    "dt_days": dt_days,
                    # 基準 shear rate に対する対数比。
                    "log_shear": np.log(shear / 500.0),
                    # 高負荷条件フラグ。
                    "high_load": high_load,
                    # 区間中央のスケーリング済み累積ろ過量。
                    "cum_mid": cum_mid,
                }
            )
            # 遷移数を 1 増やす。
            transition_count += 1

    # 行が観測、列が遷移となる累積行列 A を初期化する。
    A = np.zeros((len(df), transition_count), dtype=float)
    # 全遷移について、影響する観測行へ 1 を設定する。
    for k, tr in enumerate(transitions):
        # 遷移の run と終点観測位置を取り出す。
        run = tr["run"]
        to_pos = tr["to_obs"]
        # この遷移増分は、同じ run の終点以降のすべての観測に寄与する。
        later = df.index[(df["run_id"] == run) & (df.index >= to_pos)].to_numpy()
        # 該当観測行・遷移列に 1 を設定する。
        A[later, k] = 1.0

    # 遷移辞書の一覧を表形式へ変換する。
    trans_df = pd.DataFrame(transitions)
    # Transmission が観測されている行の位置を取得する。
    trans_obs_idx = np.flatnonzero(trans_obs_mask)
    # Filtrate 抗体と供給液抗体の両方が利用可能な行を選ぶ。
    filtrate_obs_mask = df["filtrate_antibody_g_l"].notna() & df["feed_antibody_g_l_interp"].notna()

    # モデル構築に必要な配列を、名前付き辞書として返す。
    return {
        # 整形後の観測データ。
        "df": df,
        # 条件順に並んだ run の一覧。
        "runs": runs,
        # 累積状態を作る観測×遷移行列。
        "A": A,
        "dt_days": trans_df["dt_days"].to_numpy(dtype=float),
        "log_shear_transition": trans_df["log_shear"].to_numpy(dtype=float),
        "high_load_transition": trans_df["high_load"].to_numpy(dtype=float),
        "cum_mid_transition": trans_df["cum_mid"].to_numpy(dtype=float),
        "run_idx_obs": df["run_idx"].to_numpy(dtype=int),
        "tmp_obs": df["tmp_kpa"].to_numpy(dtype=float),
        "back_pressure_load": df["back_pressure_load"].to_numpy(dtype=float),
        "trans_obs_idx": trans_obs_idx,
        "transmission_logit_obs": transmission_logit_obs,
        "filtrate_obs_idx": np.flatnonzero(filtrate_obs_mask.to_numpy()),
        "filtrate_antibody_obs": df.loc[filtrate_obs_mask, "filtrate_antibody_g_l"].to_numpy(dtype=float),
        "feed_antibody_interp": df["feed_antibody_g_l_interp"].to_numpy(dtype=float),
    }


def build_model(data: dict):
    # PyMC とその計算グラフ用テンソルライブラリは、モデル構築時だけ遅延 import する。
    import pymc as pm
    import pytensor.tensor as pt

    # run ごとのベースラインパラメータ数を決める。
    n_runs = len(data["runs"])

    # ここから PyMC の確率モデルコンテキストに入る。
    with pm.Model() as model:
        # 観測データを PyMC の共有入力として登録する。
        A = pm.Data("A", data["A"])
        dt_days = pm.Data("dt_days", data["dt_days"])
        log_shear = pm.Data("log_shear_transition", data["log_shear_transition"])
        high_load = pm.Data("high_load_transition", data["high_load_transition"])
        cum_mid = pm.Data("cum_mid_transition", data["cum_mid_transition"])
        run_idx_obs = pm.Data("run_idx_obs", data["run_idx_obs"])
        back_pressure_load = pm.Data("back_pressure_load", data["back_pressure_load"])

        # fouling の対数増加率を構成する事前分布付きパラメータ。
        theta0 = pm.Normal("theta0_log_rate", mu=-3.0, sigma=1.5)
        theta_shear = pm.Normal("theta_shear", mu=0.0, sigma=0.8)
        theta_high = pm.Normal("theta_high_load", mu=0.0, sigma=1.0)
        theta_cum = pm.Normal("theta_cumulative_dose", mu=0.0, sigma=1.0)

        # 時間・shear・高負荷・累積ろ過量から遷移ごとの対数増加率を計算する。
        log_rate = theta0 + theta_shear * log_shear + theta_high * high_load + theta_cum * cum_mid
        # 正の増加率へ戻し、遷移時間を掛けて fouling 増分にする。
        increments = pm.Deterministic("fouling_increments", pt.exp(log_rate) * dt_days)
        # 累積行列との積で、各観測時点の単調増加 fouling 状態 F を得る。
        F = pm.Deterministic("fouling_state", pt.dot(A, increments))

        beta_tmp = pm.HalfNormal("beta_tmp", sigma=5.0)
        beta_transmission = pm.HalfNormal("beta_transmission_logit", sigma=3.0)
        beta_back_pressure = pm.LogNormal("beta_back_pressure", mu=0.0, sigma=0.15)
        tau_tmp = pm.HalfNormal("tau_tmp", sigma=1.0)
        tau_transmission = pm.HalfNormal("tau_transmission", sigma=1.0)

        # x が負ならほぼ 0、正ならほぼ x になる滑らかな正値化関数。
        def smooth_positive(x, k=4.0):
            return pt.log1p(pt.exp(k * x)) / k

        # TMP と Transmission が fouling の影響を受け始める量を計算する。
        tmp_burden = smooth_positive(F - tau_tmp)
        transmission_burden = smooth_positive(F - tau_transmission)

        # run 固有の TMP 切片と Transmission の初期 logit を定義する。
        alpha_tmp = pm.Normal("alpha_tmp_baseline", mu=0.0, sigma=5.0, shape=n_runs)
        gamma_transmission = pm.Normal("gamma_transmission_baseline", mu=4.0, sigma=1.5, shape=n_runs)

        # TMP の条件付き平均: run 切片 + 背圧効果 + fouling 効果。
        mu_tmp = pm.Deterministic(
            "tmp_pred_kpa",
            alpha_tmp[run_idx_obs] + beta_back_pressure * back_pressure_load + beta_tmp * tmp_burden,
        )
        # TMP 観測ノイズの標準偏差。
        sigma_tmp = pm.HalfNormal("sigma_tmp_kpa", sigma=2.0)
        # 外れ値に比較的頑健な Student-t 分布で TMP 観測を説明する。
        pm.StudentT("tmp_obs", nu=4, mu=mu_tmp, sigma=sigma_tmp, observed=data["tmp_obs"])

        # Transmission が観測されている行だけを選ぶ。
        trans_idx = data["trans_obs_idx"]
        # Transmission の予測平均を logit スケールで作る。
        mu_trans_logit = pm.Deterministic(
            "transmission_logit_pred",
            gamma_transmission[run_idx_obs[trans_idx]] - beta_transmission * transmission_burden[trans_idx],
        )
        # Transmission の logit スケールの観測ノイズ。
        sigma_trans = pm.HalfNormal("sigma_transmission_logit", sigma=0.7)
        # logit 変換済み Transmission 観測を Student-t 尤度へ入れる。
        pm.StudentT(
            "transmission_logit_obs",
            nu=4,
            mu=mu_trans_logit,
            sigma=sigma_trans,
            observed=data["transmission_logit_obs"],
        )

        # 供給液抗体濃度と Transmission から Filtrate 抗体濃度を診断的に予測する。
        # この値は既定では尤度に入れず、モデル出力としてのみ保存する。
        transmission_pred_pct = pm.Deterministic("transmission_pred_pct", 100.0 * pm.math.sigmoid(gamma_transmission[run_idx_obs] - beta_transmission * transmission_burden))
        # 補間済み供給液抗体濃度をモデル入力として登録する。
        feed = pm.Data("feed_antibody_interp", data["feed_antibody_interp"])
        # Filtrate 抗体濃度 = 供給液抗体濃度 × Transmission 比率。
        pm.Deterministic("filtrate_antibody_pred_g_l", feed * transmission_pred_pct / 100.0)

    # 構築済みの PyMC モデルを返す。
    return model


def posterior_mean_predictions(idata, data: dict, hdi_prob: float = 0.90) -> pd.DataFrame:
    # 元データをコピーし、推定後の平均・区間列を追加する。
    df = data["df"].copy()
    # CSV に出力する deterministic 変数を順に処理する。
    for var in ["fouling_state", "tmp_pred_kpa", "transmission_pred_pct", "filtrate_antibody_pred_g_l"]:
        # 事後サンプルから平均値と指定区間を計算する。
        mean, lo, hi = posterior_interval(idata, var, hdi_prob=hdi_prob)
        # 予測平均列を追加する。
        df[f"{var}_mean"] = mean
        # 区間下限列を追加する。
        df[f"{var}_lower"] = lo
        # 区間上限列を追加する。
        df[f"{var}_upper"] = hi
    # 観測データと予測列を結合した表を返す。
    return df


def posterior_array(idata, var_name: str) -> np.ndarray:
    """Return posterior draws as an array shaped (n_observations, n_samples)."""
    # chain と draw を一つの sample 軸に結合する。
    da = idata.posterior[var_name].stack(sample=("chain", "draw"))
    # sample 以外の次元を抽出する。
    non_sample_dims = [d for d in da.dims if d != "sample"]
    # 観測次元を先、サンプル次元を最後に並べて形状を統一する。
    if non_sample_dims:
        da = da.transpose(*non_sample_dims, "sample")
    # xarray の値を NumPy 配列へ変換する。
    arr = np.asarray(da.values)
    # スカラー系列の場合も、観測数×サンプル数の2次元にそろえる。
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    # 形状統一済みの事後サンプルを返す。
    return arr


def posterior_interval(idata, var_name: str, hdi_prob: float = 0.90) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Equal-tail credible interval for a vector deterministic variable."""
    # 対象変数の事後サンプルを2次元配列として取得する。
    arr = posterior_array(idata, var_name)
    # 両側区間の下側分位点を計算する。
    q_lo = (1.0 - hdi_prob) / 2.0
    # 両側区間の上側分位点を計算する。
    q_hi = 1.0 - q_lo
    # サンプル軸方向の事後平均を計算する。
    mean = arr.mean(axis=-1)
    # サンプル軸方向の下側分位点を計算する。
    lo = np.quantile(arr, q_lo, axis=-1)
    # サンプル軸方向の上側分位点を計算する。
    hi = np.quantile(arr, q_hi, axis=-1)
    # 平均、下限、上限を返す。
    return mean, lo, hi


def configure_matplotlib_for_japanese() -> bool:
    """Enable Japanese text when japanize_matplotlib is installed."""
    # 日本語フォント設定用パッケージが利用できるかを確認する。
    try:
        # import 自体で日本語表示設定が有効になるパッケージを読み込む。
        import japanize_matplotlib  # noqa: F401

        # 日本語表示が可能であることを呼び出し側へ返す。
        return True
    except ImportError:
        # パッケージ未導入でも英語ラベルで処理を継続する。
        return False


def make_diagnostic_plots(idata, data: dict, out_dir: Path, hdi_prob: float = 0.90) -> Path:
    # グラフ描画と複数ページ PDF 出力を遅延 import する。
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    # 日本語フォントが使えるか確認する。
    japanese = configure_matplotlib_for_japanese()
    # 描画用に整形済みデータと時間軸を取り出す。
    df = data["df"].copy()
    x = df["elapsed_h"].to_numpy(dtype=float)
    # 各 deterministic 変数の事後平均・区間を一度だけ計算して使い回す。
    interval = {
        var: posterior_interval(idata, var, hdi_prob)
        for var in ["fouling_state", "tmp_pred_kpa", "transmission_pred_pct", "filtrate_antibody_pred_g_l"]
    }

    # 診断 PDF の出力先を作る。
    pdf_path = out_dir / "akls_small_ssm_pymc_diagnostics.pdf"
    ci_label = f"{int(hdi_prob * 100)}% credible interval"
    if japanese:
        ci_label = f"{int(hdi_prob * 100)}% 信用区間"

    # PdfPages を使って run ごとの図を一つの PDF にまとめる。
    with PdfPages(pdf_path) as pdf:
        # run ごとに 4 段の診断図を作る。
        for run in data["runs"]:
            # 現在の run に属する観測行だけを選ぶ。
            mask = df["run_id"].eq(run).to_numpy()
            # 現在の run の時間軸と部分データを取り出す。
            xs = x[mask]
            sub = df.loc[mask]
            # F、TMP、Transmission、Filtrate 抗体を縦に並べた図を作る。
            fig, axes = plt.subplots(4, 1, figsize=(11.69, 8.27), sharex=True)
            # 図全体のタイトルに run ID と条件名を表示する。
            title = f"{run} / {sub['condition_label'].iloc[0]}"
            # 日本語フォントがある場合は日本語タイトルを使う。
            if japanese:
                fig.suptitle(f"{title} - PyMC小型SSM posterior診断", fontsize=14, fontweight="bold")
            # 日本語フォントがない場合は英語タイトルを使う。
            else:
                fig.suptitle(f"{title} - PyMC small SSM posterior diagnostics", fontsize=14, fontweight="bold")

            # 各系列の平均線と信用区間を描画する共通関数。
            def band(ax, var_name, label, color="#D62728"):
                # 指定変数の事後平均と区間を取得する。
                mean, lo, hi = interval[var_name]
                # 事後平均線を描く。
                ax.plot(xs, mean[mask], color=color, linewidth=2, label=label)
                # 区間を半透明の帯として描く。
                ax.fill_between(xs, lo[mask], hi[mask], color=color, alpha=0.18, label=ci_label)

            # 最上段に潜在 fouling 状態 F の平均と区間を描く。
            band(axes[0], "fouling_state", "Fouling state F" if not japanese else "潜在fouling状態 F", "#2F5597")
            # F 軸のラベル、グリッド、凡例を設定する。
            axes[0].set_ylabel("F" if not japanese else "Fouling状態")
            axes[0].grid(alpha=0.3)
            axes[0].legend(loc="best", fontsize=8)

            # TMP の観測値が存在する行を示すマスクを作る。
            obs_tmp = sub["tmp_kpa"].notna().to_numpy()
            # 2段目に TMP 観測値を散布図として描く。
            axes[1].scatter(xs[obs_tmp], sub.loc[obs_tmp, "tmp_kpa"], s=16, color="#1F77B4", label="Observed TMP" if not japanese else "観測TMP")
            # 同じ軸へ TMP の予測平均と区間を重ねる。
            band(axes[1], "tmp_pred_kpa", "Predicted TMP" if not japanese else "予測TMP")
            # TMP 軸の体裁を整える。
            axes[1].set_ylabel("TMP (kPa)")
            axes[1].grid(alpha=0.3)
            axes[1].legend(loc="best", fontsize=8)

            # Transmission の観測値が存在する行を示すマスクを作る。
            obs_tr = sub["transmission_pct"].notna().to_numpy()
            # 3段目に Transmission の観測値を描く。
            axes[2].scatter(
                xs[obs_tr],
                sub.loc[obs_tr, "transmission_pct"],
                s=16,
                color="#1F77B4",
                label="Observed transmission" if not japanese else "観測Transmission",
            )
            # Transmission の予測平均と区間を描く。
            band(axes[2], "transmission_pred_pct", "Predicted transmission" if not japanese else "予測Transmission")
            # Transmission 軸の体裁を整える。
            axes[2].set_ylabel("Transmission (%)")
            axes[2].grid(alpha=0.3)
            axes[2].legend(loc="best", fontsize=8)

            # Filtrate 抗体の観測値が存在する行を示すマスクを作る。
            obs_ab = sub["filtrate_antibody_g_l"].notna().to_numpy()
            # 4段目に Filtrate 抗体の観測値を描く。
            axes[3].scatter(
                xs[obs_ab],
                sub.loc[obs_ab, "filtrate_antibody_g_l"],
                s=16,
                color="#1F77B4",
                label="Observed filtrate Ab" if not japanese else "観測Filtrate抗体",
            )
            # Filtrate 抗体の予測平均と区間を描く。
            band(axes[3], "filtrate_antibody_pred_g_l", "Predicted filtrate Ab" if not japanese else "予測Filtrate抗体")
            # 最下段の軸ラベルと共通表示を設定する。
            axes[3].set_ylabel("g/L")
            axes[3].set_xlabel("Elapsed time (h)" if not japanese else "ろ過開始後時間 (h)")
            axes[3].grid(alpha=0.3)
            axes[3].legend(loc="best", fontsize=8)

            # タイトルと重ならないように図全体の余白を調整する。
            fig.tight_layout(rect=[0, 0.02, 1, 0.95])
            # 現在の run の図を PDF の1ページとして保存する。
            pdf.savefig(fig)
            # メモリ上の図を閉じ、run 数が多い場合のメモリ消費を抑える。
            plt.close(fig)

            # 作成した PDF のパスを呼び出し側へ返す。
    return pdf_path


def main() -> None:
    # コマンドライン引数を定義するパーサーを作成する。
    parser = argparse.ArgumentParser()
    # 入力 CSV または Excel のパスを必須引数として受け取る。
    parser.add_argument("--input", required=True, help="Path to master_data.csv or AKLS_master_dataset.xlsx")
    # 推定結果の出力ディレクトリを受け取る。
    parser.add_argument("--output-dir", default="akls_pymc_results")
    # MCMC の本サンプル数を受け取る。
    parser.add_argument("--draws", type=int, default=1000)
    # MCMC のウォームアップ・調整サンプル数を受け取る。
    parser.add_argument("--tune", type=int, default=1000)
    # 並列に実行する chain 数を受け取る。
    parser.add_argument("--chains", type=int, default=4)
    # NUTS の目標受入率を受け取る。
    parser.add_argument("--target-accept", type=float, default=0.9)
    # 乱数系列を再現可能にする seed を受け取る。
    parser.add_argument("--random-seed", type=int, default=42)
    # 予測 CSV と図に使う信用区間の確率を受け取る。
    parser.add_argument("--hdi-prob", type=float, default=0.90, help="Credible interval probability for plots and prediction CSV")
    # 指定時は診断 PDF の生成を省略するフラグ。
    parser.add_argument("--no-plots", action="store_true", help="Skip posterior diagnostic PDF generation")
    # コマンドライン引数を実際の値へ変換する。
    args = parser.parse_args()

    # 推定結果の要約と MCMC サンプリングに使うライブラリを読み込む。
    import arviz as az
    import pymc as pm

    # 出力ディレクトリを Path オブジェクトとして扱う。
    out_dir = Path(args.output_dir)
    # 出力先がなければ親ディレクトリも含めて作成する。
    out_dir.mkdir(parents=True, exist_ok=True)

    # 入力ファイルを読み込み、モデル用の配列へ整形する。
    df = read_master(Path(args.input))
    data = prepare_model_data(df)
    # 整形済みデータから PyMC モデルを構築する。
    model = build_model(data)

    # モデルコンテキスト内で MCMC サンプリングを実行する。
    with model:
        idata = pm.sample(
            # 事後分布から保存する本サンプル数。
            draws=args.draws,
            # サンプラー調整用のサンプル数。
            tune=args.tune,
            # 実行する独立 chain 数。
            chains=args.chains,
            # NUTS の目標受入率。
            target_accept=args.target_accept,
            # 推定を再現可能にする乱数 seed。
            random_seed=args.random_seed,
            # ArviZ InferenceData 形式で結果を返す。
            return_inferencedata=True,
        )

    # chain・draw を含む全推定結果を NetCDF に保存する。
    az.to_netcdf(idata, out_dir / "akls_small_ssm_pymc_idata.nc")
    # 主要パラメータの事後統計量を計算する。
    summary = az.summary(
        idata,
        var_names=[
            "theta0_log_rate",
            "theta_shear",
            "theta_high_load",
            "theta_cumulative_dose",
            "beta_tmp",
            "beta_transmission_logit",
            "beta_back_pressure",
            "tau_tmp",
            "tau_transmission",
            "sigma_tmp_kpa",
            "sigma_transmission_logit",
        ],
    )
    # パラメータ要約を BOM 付き UTF-8 CSV として保存する。
    summary.to_csv(out_dir / "akls_small_ssm_pymc_parameter_summary.csv", encoding="utf-8-sig")

    # 各観測行の予測平均と信用区間を計算する。
    pred = posterior_mean_predictions(idata, data, hdi_prob=args.hdi_prob)
    # 観測値と予測値を BOM 付き UTF-8 CSV として保存する。
    pred.to_csv(out_dir / "akls_small_ssm_pymc_predictions.csv", index=False, encoding="utf-8-sig")
    # --no-plots が指定されていない場合だけ診断 PDF を作成する。
    if not args.no_plots:
        # run ごとの診断図を作成する。
        plot_path = make_diagnostic_plots(idata, data, out_dir, hdi_prob=args.hdi_prob)
        # 作成した PDF の場所を標準出力へ表示する。
        print(f"Saved diagnostic plots to: {plot_path}")

    # 主要な出力ディレクトリを表示する。
    print(f"Saved outputs to: {out_dir}")
    # パラメータ要約をコンソールにも表示する。
    print(summary)


if __name__ == "__main__":
    # このファイルをスクリプトとして直接実行した場合だけ main を呼ぶ。
    main()
