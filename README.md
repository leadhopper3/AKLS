# AKLS: 膜ろ過実験の状態空間モデル解析

AKLSは、膜ろ過・パーフュージョン培養模擬実験の時系列データを対象に、PyMCによるベイズ状態空間モデル（State Space Model; SSM）を用いて、膜ファウリング、TMP、Transmission、filtrate中の抗体濃度、および累積回収量の関係を解析するリポジトリです。

## 目的

- 実験時系列をrun単位に整形し、観測値を可視化する
- 膜の目詰まり（fouling）を潜在状態として推定する
- fouling、背圧、せん断条件、累積ろ過量からTMPとTransmissionを推定する
- feed抗体濃度とTransmissionから、filtrate抗体濃度を診断的に予測する
- 前段モデルの予測を使って、Historical modelを二段階目のモデルとして評価する
- 事後分布、信用区間、診断図を保存し、条件ごとの不確実性を確認する

本リポジトリの予測は、観測された条件とデータ範囲を対象とした確率的な推定です。未検証条件への外挿や、モデルから因果関係を証明することを目的としていません。

## モデルの全体像

推奨する主な解析経路は次のとおりです。

```text
入力データ
  |
  v
AKLS_interactive_preliminary_analysis_v2.ipynb
  |  データ確認・整形・探索的可視化
  v
akls_small_ssm_pymc.py または akls_small_ssm_pymc_interactive.ipynb
  |  fouling / TMP / Transmission のベイズSSM
  v
akls_pymc_interactive_results/akls_small_ssm_pymc_predictions.csv
  |
  v
akls_historical_ssm_auxiliary_A_B.ipynb
  |  案A: effective sieving
  |  案B: cumulative recovery / quality
  v
akls_historical_ssm_auxiliary_results/
```

前段モデルの主な状態方程式は、各run内の遷移を累積する形で表します。

```text
F_t = F_{t-1} + fouling_increment_t
TMP_t = run_baseline + back_pressure_effect + fouling_effect
Transmission_t = sigmoid(run_baseline - fouling_effect)
filtrate_antibody_t = feed_antibody_t * Transmission_t
```

Historical SSMでは、前段の予測値とその信用区間を説明変数として受け取り、effective sievingまたは累積回収量を推定します。

## ファイル構成

### 推奨している実行ファイル

| ファイル | 役割 |
| --- | --- |
| `akls_small_ssm_pymc.py` | コマンドラインから実行できる小型SSM。再現実行に推奨 |
| `akls_small_ssm_pymc_interactive.ipynb` | 前段Propagation/SSMの対話的な実行・可視化 |
| `akls_historical_ssm_auxiliary_A_B.ipynb` | 前段出力を使うHistorical SSMの案A/B |
| `AKLS_interactive_preliminary_analysis_v2.ipynb` | 入力データの確認、整形、探索的分析 |
| `akls_hybrid_propagation_model_v1.ipynb` | ハイブリッドPropagationモデルの検討版 |

### 比較・旧版のNotebook

以下はモデル比較や過去実装の確認に利用できます。新規の再現実行では、まず上記の推奨経路を使用してください。

- `AKLS_interactive_preliminary_analysis.ipynb`
- `akls_historical_model_connected.ipynb`
- `akls_historical_model_connected_v2.ipynb`

### 入力データと出力

- `AKLS_cleaned_records.csv`: 整形済み観測データ
- `AKLS_experiment_summary.csv`: 実験・runの概要
- `user_files/`: 元データや追加入力を置く場所
- `akls_pymc_interactive_results/`: 前段SSMの結果
- `akls_historical_ssm_auxiliary_results/`: Historical SSMの結果
- `akls_historical_model_results/`: 接続型Historical modelの結果
- `akls_hybrid_propagation_v1_results/`: Hybrid propagation modelの結果

GitHubへ公開する場合、実験データ、元Excel、ZIP、NetCDF、診断PDFなどに機密情報やサイズ制限がないか確認してください。大容量ファイルや非公開データは、Git LFS、GitHub Releases、または外部ストレージを利用し、READMEには取得方法だけを記載する運用を推奨します。

## 環境構築

Python 3.10以上を推奨します。PyMCの対応範囲に合わせて、実行環境ではPythonと主要パッケージのバージョンを固定してください。

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r akls_pymc_requirements.txt
```

Jupyter Notebookを使う場合は、環境を選択した後にJupyter関連パッケージを追加してください。

```powershell
python -m pip install jupyterlab ipykernel
python -m ipykernel install --user --name akls-pymc --display-name "Python (AKLS PyMC)"
```

PyMCのインストール後、次のコマンドで主要パッケージを確認できます。

```powershell
python -c "import pymc, arviz, pytensor, numpy, pandas; print('PyMC', pymc.__version__)"
```

## 実行方法

### 1. 入力データを確認する

`AKLS_interactive_preliminary_analysis_v2.ipynb`を開き、上から順に実行します。入力データの列名、単位、runの分割、欠測値、観測範囲を確認してください。

前段SSMで最低限必要になる代表的な列は次のとおりです。

- `run_id`
- `elapsed_h`
- `tmp_kpa`
- `cumulative_filtrate_l_m2`
- `back_pressure_kpa`
- `transmission_pct`
- `lmh_group`
- `shear_rate_lmh`
- `feed_antibody_g_l`
- `filtrate_antibody_g_l`

Excelを使う場合、`akls_small_ssm_pymc.py`は`Master_Data`シートを読み込みます。CSVを使う場合は、同等のcanonical列名を用意してください。

### 2. 前段の小型SSMを実行する

再現性を重視する場合は、まずスクリプトを小さい設定で実行し、入出力を確認します。

```powershell
python akls_small_ssm_pymc.py `
  --input AKLS_cleaned_records.csv `
  --output-dir akls_pymc_interactive_results `
  --draws 100 `
  --tune 100 `
  --chains 2
```

入力がExcelの場合は、次のように指定します。

```powershell
python akls_small_ssm_pymc.py `
  --input user_files/AKLS_master_dataset.xlsx `
  --output-dir akls_pymc_interactive_results `
  --draws 1000 `
  --tune 1000 `
  --chains 4 `
  --target-accept 0.9 `
  --random-seed 42
```

診断PDFが不要な場合は `--no-plots` を追加できます。サンプリング後は、少なくとも次のファイルを確認してください。

- `akls_small_ssm_pymc_idata.nc`: ArviZ InferenceData形式の全事後結果
- `akls_small_ssm_pymc_parameter_summary.csv`: パラメータ要約
- `akls_small_ssm_pymc_predictions.csv`: 各時点の予測平均と信用区間
- `akls_small_ssm_pymc_diagnostics.pdf`: run別の診断図

### 3. Historical SSMを実行する

前段の `akls_small_ssm_pymc_predictions.csv` が作成された状態で、`akls_historical_ssm_auxiliary_A_B.ipynb`を開きます。

1. カーネルを再起動する
2. Notebookを上から順に実行する
3. 必要に応じて `RUN_MODEL_A`、`RUN_MODEL_B`、`RUN_SAMPLING` を変更する
4. まず `RUN_SAMPLING = False` でデータ整形とモデル構築を確認する
5. 問題がなければ、draws/tune/chainsを増やしてサンプリングする

案Aはfiltrate抗体濃度とeffective sievingを中心に扱います。案Bはそれに加えて、累積ろ過量から累積回収量proxyを構成します。案Bの `observed_cum_recovery_g_m2_proxy` は独立した実測累積回収量ではないため、結果の解釈には注意してください。

## 出力の読み方

予測CSVの `_mean`、`_lower`、`_upper` は、それぞれ事後平均、信用区間の下限、信用区間の上限です。既定の区間確率は90%です。

結果を採用する前に、次を確認してください。

- MCMCの発散や収束警告がないか
- `r_hat`、ESS、trace plotが妥当か
- 観測値と予測平均の系統的なずれがないか
- run間で信用区間の幅が不自然に狭くなっていないか
- 欠測補間やTransmissionの端点クリップが結論に影響していないか
- 前段の予測不確実性がHistorical SSMへ伝播されているか

## 重要な仮定と制約

- 欠測値は主にrun内の時系列補間で扱います。run内に値がない場合、一部の変数は0などの既定値になります。
- Transmissionの0%と100%は、logit変換のため計算上0.5%から99.5%へクリップされます。
- filtrate抗体濃度は、前段SSMでは主に診断的な予測値として扱われます。Historical SSMでは補助観測として尤度に含めます。
- Transmission観測とfiltrate抗体濃度を同時に尤度へ入れる場合、同じ情報を二重に使う可能性があります。
- PyMCのMCMC結果は乱数seedを固定しても、OS、BLAS、PyMC、サンプラー設定の違いにより完全一致しない場合があります。
- 前段の `_lower` / `_upper` 列が存在する場合、Historical SSMはそれらから近似した不確実性を利用します。

## 再現性のための記録

解析結果を共有する際は、次の情報を併記してください。

- Python、PyMC、ArviZ、PyTensorのバージョン
- 入力ファイル名とデータ版
- `draws`、`tune`、`chains`、`target_accept`
- 使用したNotebookまたはスクリプト
- 乱数seed
- 欠測値・補間・外れ値処理の変更内容
- 結果CSVと診断図を生成した日時

## ライセンスとデータ公開

このREADMEではライセンスを指定していません。GitHubで一般公開する前に、コードのライセンス、実験データの公開可否、第三者データや装置ログの取り扱いをプロジェクト責任者と確認してください。