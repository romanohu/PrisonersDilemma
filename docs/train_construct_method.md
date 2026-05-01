# N-Agent Pair PD: 外部ラッパー運用メモ

このドキュメントは、`PairedPopulationPrisonersDilemmaEnv` を**外部ラッパー経由で学習に接続する際の注意点**をまとめたものです。  
環境本体はフレームワーク非依存で、学習基盤固有の制御はラッパー側で行います。

## 1. 設計方針の分離

- 環境側:
  - ゲーム進行と報酬計算の正しさを保証する。
  - 非参加エージェントを `can_act=False` でハードマスクし、遷移・報酬に関与させない。
  - 制御メタデータは `infos[i]` に返す。
- ラッパー側:
  - `infos[i]` を見て、どのモデルを通すか・どのサンプルを学習に使うかを決める。
  - 学習基盤固有のマスク/バッファ投入制御を実装する。

## 2. 環境の固定API（外部から見える契約）

- `reset() -> (observations, infos)`
- `step(actions) -> (observations, rewards, terminations, truncations, infos)`
- 終了後に `step()` すると `RuntimeError`。**必ず `reset()` を呼ぶ**。
- `reset_population_stats()` でEMA統計のみ明示リセット（`reset()` ではEMAは消えない）。

## 3. action の扱い

- 各agent actionは常に `(partner_choice_rel, pd_action)`

補足:
- `partner_choice_rel` は相対ID（`Discrete(N-1)`）。
- `infos` に返る `selected_partner` / `active_opponent_id` は絶対ID。
- `C/D` は `core.py` 準拠で `0:C, 1:D`。

## 4. observation の扱い

- `Dict` で固定:
  - `selection_obs`: `(N-1, 2)`
  - `pd_obs`: `(2,)`
- 未使用フェーズ側はゼロ埋め。
- `pd_obs` エンコード:
  - `C -> [1,0]`
  - `D -> [0,1]`
  - 未観測初期値 -> `[0,0]`

`phase` は観測ベクトルに入れず、`infos[i]["phase"]` を使って外部でモデル切替する前提。

## 5. infos に載る制御メタデータ（ラッパー必読）

各agent `i` の `infos[i]`:

- `phase`: `0=matching`, `1=pd`
- `can_act`: このstepで環境遷移に有効な行動を出すべきか
- `new_match`: 新しいローカル対戦の開始stepか
- `active_opponent_id`: 対戦相手の絶対ID（非参加は `-1`）
- `selected_partner`: matchingで選んだ相手の絶対ID
- `true_objective`: エピソード累積報酬（`np.float32`）

## 6. 非参加エージェントの厳密ルール

環境実装上の保証:
- `can_act=False` agentの行動は、遷移・報酬計算に使わない。
- 非参加agentは学習上意味のある内部更新を行わない（local PD state更新対象外）。

ラッパー実装上の推奨:
- `can_act=False` のサンプルは学習更新対象から外す（または loss mask で無効化）。
- ただし `step()` 呼び出しの都合上、action配列は全agent分渡す必要があるため、非参加agentにはダミーactionを埋める。

## 7. エピソード進行（外から回すときの前提）

- 1エピソードは以下で固定:
  1. matching phase を1step（報酬は全員0）
  2. directed match `i -> partner[i]` を全agent分構築
  3. greedy packingで重複なしスロット列に分割
  4. 各スロットを `pd_horizon` step ずつ実行
  5. 全スロット完了でエピソード終了

重要:
- `i -> j` と `j -> i` は**独立な別試合**。
- 同一slot内で同一agentが重複しないようにpackされる。
- したがって、1step内で複数試合を同時処理しても、各agentはそのstepで高々1回だけ行動する。

## 8. 各ローカル試合の初期 `pd_obs`

- 各ローカル試合の1step目 `pd_obs` は常に `[0,0]`。
- 1step目には直前行動の履歴は存在しないため、環境側で固定初期観測を返す。
- 報酬、PD統計、EMA更新に追加の side channel は使わない。

## 9. EMA統計の意味と更新

selectionで使う候補特徴（初版）:
- `moving_avg_own_reward[j]`
- `moving_avg_partner_reward_when_j_is_present[j]`

意味:
- `own_reward`: agent `j` のエピソード総報酬のEMA
- `partner_reward_when_j_is_present`: `j` が参加した試合群における相手側報酬平均のEMA

更新規則:
- `ema <- (1 - alpha) * ema + alpha * episode_value`
- `reset()` では保持、`reset_population_stats()` でのみ初期化。

## 10. 外部ラッパー実装の最小チェックリスト

- `infos.phase` を見て selectionモデル / PDモデルを切替する。
- `infos.can_act` を見て行動生成と学習投入を制御する。
- `infos.new_match` を見て、必要ならRNN stateやローカル対戦バッファ境界を切る。
- 非参加agentのactionは必ずダミーで埋める（未定義値を渡さない）。
- 終了step受領後は次の `step()` 前に必ず `reset()`。
- EMAをエピソード間で保持したくない実験では `reset_population_stats()` を明示呼び出しする。

## 11. 学習基盤との関係（制約）

- この環境は学習基盤非依存で実装する。
- ただし `infos` ベースの制御メタデータを返すため、Sample Factoryを含む外部基盤でラッパー実装は可能。
- 「環境内に基盤依存ロジックを埋めない」方針を維持する。
