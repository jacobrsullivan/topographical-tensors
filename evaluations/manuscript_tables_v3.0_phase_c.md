# Phase C Tier C-Full (`v3.0.0`) — Full-Scale Lichess & Unmodified Stockfish 19 UCI Empirical Tables

- **Execution Host:** `apple-m4-workstation` (`mps`)
- **Total Wall-Clock Runtime:** `1228.21 s`
- **All 5 Phase C Stage-Gates Passed (`SG1-C`..`SG5-C`):** `True`

## Table 1: `WP1-C` Lichess Corpus Scale & Zobrist Deduplication Audit (`SG1-C`)

| Metric | Value | Gate Threshold | Status |
| :--- | :---: | :---: | :---: |
| Total Zobrist-Deduplicated $8 \times 8$ Lichess States | `1,269,820` | $\ge 1,000,000$ | **PASS** |
| Train / Val / Test Split States | `1,164,199` / `64,987` / `40,634` | `90% / 5% / 5%` | **PASS** |
| Cross-Split Zobrist & Mirror-Zobrist Overlap | `0` | `0` | **PASS** |
| 6-Ply Lichess Principal Variation Trajectories | `250,000` | $\ge 250,000$ | **PASS** |
| External Held-Out Lichess Benchmark Puzzles | `10,000` | `10,000` | **PASS** |
| Compact State Shard Footprint (`ALT-008`) | `264.24 MB` | `< 500 MB` | **PASS** |

## Table 2: `WP2-C` Full `CF-6M` (`5.87M` Param) Motif-Stratified Action Ranking (`SG2-C`)

| Configuration | Params | Spatial FLOPS | Overall $\tau$ | Pin $\tau$ | X-Ray $\tau$ | Discovered $\tau$ | Fork $\tau$ | Quiet $\tau$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Config_A** | `5,619,205` | `981.73M` | `0.5641` | `0.5559` | `0.5641` | `0.6193` | `0.5663` | `0.5853` |
| **Config_B** | `5,647,925` | `998.51M` | `0.6425` | `0.6318` | `0.6425` | `0.6430` | `0.6481` | `0.6503` |
| **Config_C** | `5,868,959` | `727.53M` | `0.8567` | `0.8365` | `0.8567` | `0.8597` | `0.8572` | `0.8689` |

- **Sparse IPA FLOPS Reduction (`Config_C` vs `Config_B`):** `27.14%`
- **Paired $t$-Test on Ray/Pin/X-Ray Motifs (`Config_C` vs `Config_B`):** `p = 5.866222e-12` (`< 0.01`)

## Table 3: `WP3-C` D3PM Intrinsic (Unmasked) Legality Scaling Curve (`ALT-009`, `SG3-C`)

| Training Trajectories ($N$) | Unmasked Ply-1 Legality | Unmasked Per-Ply Legality ($k=1..6$) | Unmasked Full 6-Ply Trajectory Legality | Post-Projection Legality |
| :---: | :---: | :---: | :---: | :---: |
| `1,000` | `8.98%` | `5.73%` | `0.39%` | `100.0%` |
| `10,000` | `4.30%` | `11.20%` | `0.00%` | `100.0%` |
| `100,000` | `100.00%` | `99.35%` | `97.27%` | `100.0%` |
| `250,000` | `100.00%` | `100.00%` | `100.00%` | `100.0%` |

## Table 4: `WP3-C` 10,000-Puzzle External Lichess Benchmark Zero-Shot Evaluation (`SG3-C`)

| Configuration | Overall Solve Rate | `<1400` Band | `1400-1800` Band | `1800-2200` Band | `2200+` Band | Ply-1 Acc | Ply-6 Acc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Config_A** | `24.68%` | `30.33%` | `21.99%` | `18.39%` | `15.04%` | `32.62%` | `8.98%` |
| **Config_B** | `24.65%` | `30.32%` | `22.14%` | `18.86%` | `13.84%` | `32.61%` | `8.96%` |
| **Config_C** | `25.97%` | `31.82%` | `23.96%` | `19.24%` | `14.75%` | `33.22%` | `8.24%` |
| **Config_D** | `47.17%` | `59.11%` | `40.29%` | `33.25%` | `30.06%` | `59.22%` | `43.59%` |

## Table 5: `WP4-C` Full 120-Ply BayesElo Leaderboard Anchored to Unmodified `Stockfish_UCI_Elo_1800 = 1800.0` (`SG4-C`)

| Rank | Participant | Calibrated BayesElo | 95% CI | Games Played | Score Rate |
| :---: | :--- | :---: | :---: | :---: | :---: |
| `1` | **Stockfish_UCI_Elo_2400** | `1995.6` | `[1967.6, 2023.7]` | `1,000` | `52.1%` |
| `2` | **Config_D** | `1980.7` | `[1962.7, 1998.7]` | `9,000` | `68.2%` |
| `3` | **Stockfish_10000_Nodes** | `1976.2` | `[1948.1, 2004.2]` | `1,000` | `49.4%` |
| `4` | **Stockfish_UCI_Elo_2100** | `1926.7` | `[1898.5, 1955.0]` | `1,000` | `42.3%` |
| `5` | **Stockfish_1000_Nodes** | `1822.4` | `[1801.2, 1843.6]` | `3,000` | `47.1%` |
| `6` | **Config_C** | `1809.5` | `[1792.2, 1826.7]` | `5,000` | `49.8%` |
| `7` | **Stockfish_UCI_Elo_1800** | `1800.0` | `[1783.8, 1816.2]` | `2,000` | `37.4%` |
| `8` | **Stockfish_100_Nodes** | `1765.9` | `[1734.4, 1797.3]` | `1,000` | `22.5%` |
| `9` | **Config_B** | `1747.1` | `[1726.5, 1767.7]` | `5,000` | `41.6%` |
| `10` | **Stockfish_UCI_Elo_1500** | `1733.2` | `[1703.4, 1763.0]` | `1,000` | `48.0%` |
| `11` | **Config_A** | `1714.1` | `[1692.3, 1735.8]` | `3,000` | `33.2%` |

- **Full 120-Ply $\Delta\text{Elo}(\text{Config\_D vs Config\_C})$:** `+171.24 Elo` (`p = 2.786286e-06`)
- **Full 120-Ply $\Delta\text{Elo}(\text{Config\_C vs Config\_B})$:** `+62.38 Elo` (`p = 2.800086e-03`)
- **Clean-Room Checkpoint Reload Max Deviation (`SG5-C`):** `0.0000%` (`<= 0.50%`)
