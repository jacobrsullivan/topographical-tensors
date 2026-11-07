# Manuscript Tables (`v4.0.0` — Phase D 40-Item Reviewer Audit & Complete Factorial Controls)

**Proposed Title:** *Topographical Move-Potential Fields and Local-Frame Discrete Diffusion for Sample-Efficient Searchless Chess*

## Table 1: Complete 8-Cell Spatial Encoder Factorial Matrix + Iso-Parameter `Config_C_Large` (Items 1, 2, 4, 12, 17, 18, 19, 27, 39)

| Cell ID | Input Channels | Attention Kernel | Connectivity Mask | Params | Pairwise Attn FLOPS | Total Encoder FLOPS | Held-Out $\tau$ (`40,634` States) | Stockfish MultiPV $\tau$ (`2k`) | Top-1 Legal Acc | Pin $\tau$ | X-Ray $\tau$ | Discovered $\tau$ | Fork $\tau$ | Quiet $\tau$ |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Enc_0_Flat12_ShawDense` | `flat_12` | `shaw_2d` | `dense` (100.0%) | `5,872,543` | `50.33M` | `876.74M` | `0.4173 ± 0.0176` | `0.5016` | `47.10%` | `0.4721` | `0.4840` | `0.5084` | `0.4456` | `0.4696` |
| `Enc_1_ConfigA_Flat20_ShawDense` | `flat_20` | `shaw_2d` | `dense` (100.0%) | `5,874,607` | `50.33M` | `876.74M` | `0.4157 ± 0.0199` | `0.4981` | `46.76%` | `0.4690` | `0.4764` | `0.5056` | `0.4420` | `0.4648` |
| `Enc_2_Flat20_ShawRayMask` | `flat_20` | `shaw_2d` | `queen_rays` (37.1%) | `5,874,607` | `18.68M` | `845.09M` | `0.5722 ± 0.0363` | `0.6743` | `46.05%` | `0.6407` | `0.6533` | `0.6541` | `0.6132` | `0.6413` |
| `Enc_3_ConfigBPrime_Flat20_IPA` | `flat_20` | `local_frame_ipa` | `queen_plus_knight` (45.3%) | `5,862,767` | `15.20M` | `761.08M` | `0.6784 ± 0.0022` | `0.8105` | `46.71%` | `0.7595` | `0.7807` | `0.7622` | `0.7185` | `0.7716` |
| `Enc_4_ConfigB_Topo44_ShawDense` | `topo_44` | `shaw_2d` | `dense` (100.0%) | `5,880,799` | `50.33M` | `876.74M` | `0.6755 ± 0.0060` | `0.8096` | `46.83%` | `0.7611` | `0.7716` | `0.7635` | `0.7172` | `0.7653` |
| `Enc_5_Topo44_ShawRayMask` | `topo_44` | `shaw_2d` | `queen_rays` (37.1%) | `5,880,799` | `18.68M` | `845.09M` | `0.7284 ± 0.0020` | `0.8527` | `46.61%` | `0.8173` | `0.8322` | `0.8119` | `0.7710` | `0.8287` |
| `Enc_6_Topo44_IPA_QueenOnly` | `topo_44` | `local_frame_ipa` | `queen_rays` (37.1%) | `5,868,959` | `12.45M` | `758.33M` | `0.7536 ± 0.0021` | `0.8556` | `46.31%` | `0.8475` | `0.8622` | `0.8387` | `0.8000` | `0.8611` |
| `Enc_7_ConfigC_Topo44_IPA_QueenKnight` | `topo_44` | `local_frame_ipa` | `queen_plus_knight` (45.3%) | `5,868,959` | `15.20M` | `761.08M` | `0.7660 ± 0.0021` | `0.8512` | `46.12%` | `0.8617` | `0.8780` | `0.8528` | `0.8134` | `0.8782` |
| `Enc_8_ConfigCLarge_IsoParam_8p57M` | `topo_44` | `local_frame_ipa` | `queen_plus_knight` (45.3%) | `8,637,347` | `18.53M` | `1117.48M` | `0.7636 ± 0.0021` | `0.8485` | `46.07%` | `0.8587` | `0.8746` | `0.8504` | `0.8107` | `0.8750` |

## Table 2: Iso-Parameter (`8.57M`) & Iso-Supervision (`250k` PVs) Horizon Controls on `10,000` External Held-Out Lichess Puzzles (Items 5, 25, 28, 32)

| Model Configuration | Total Params | Denoising Steps $T$ | Root Ply-1 Accuracy | Interactive Multi-Turn Solve Rate | Single-Root $z(s_0)$ Solve Rate | `<1400` | `1400–1800` | `1800–2200` | `2200+` | `mateIn3` | `pin` | `xrayAttack` | `discoveredAttack` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Config_A_Flat20_Shaw` | `5,874,607` | `0` | `29.71%` | `12.80%` | `12.80%` | `17.37%` | `10.65%` | `7.36%` | `5.44%` | `17.36%` | `7.45%` | `23.66%` | `16.57%` |
| `Config_B_Prime_Flat20_IPA` | `5,862,767` | `0` | `36.41%` | `18.85%` | `15.35%` | `24.81%` | `15.58%` | `12.59%` | `8.96%` | `27.45%` | `10.92%` | `38.93%` | `26.23%` |
| `Config_B_Topo44_Shaw` | `5,880,799` | `0` | `35.70%` | `17.63%` | `15.50%` | `23.65%` | `14.30%` | `11.16%` | `7.90%` | `26.18%` | `10.75%` | `37.40%` | `23.87%` |
| `Config_C_Topo44_IPA_5p87M` | `5,868,959` | `0` | `41.41%` | `22.19%` | `16.96%` | `29.38%` | `18.66%` | `13.78%` | `10.71%` | `34.30%` | `13.69%` | `40.46%` | `29.59%` |
| `Config_C_Large_IsoParam_8p57M` | `8,637,347` | `0` | `44.66%` | `24.80%` | `17.66%` | `32.40%` | `20.73%` | `16.09%` | `13.08%` | `37.69%` | `16.46%` | `41.98%` | `32.54%` |
| `Config_D_AR_Cached_8p57M` | `15,201,572` | `0` | `18.31%` | `6.40%` | `6.40%` | `5.97%` | `5.24%` | `7.90%` | `8.34%` | `7.20%` | `3.81%` | `2.29%` | `2.96%` |
| `Config_D_OneShot_T1_8p57M` | `16,180,170` | `1` | `34.67%` | `16.45%` | `16.45%` | `20.00%` | `13.60%` | `12.83%` | `12.20%` | `24.98%` | `11.27%` | `22.14%` | `25.84%` |
| `Config_D_NoALT009_T20_8p57M` | `15,661,538` | `20` | `62.93%` | `36.98%` | `36.98%` | `45.97%` | `28.79%` | `28.86%` | `26.60%` | `61.47%` | `28.42%` | `34.35%` | `32.35%` |
| `Config_D_Full_D3PM_T20_8p57M` | `16,180,170` | `20` | `59.35%` | `40.75%` | `40.75%` | `44.08%` | `38.64%` | `36.76%` | `36.52%` | `53.78%` | `33.97%` | `48.85%` | `51.28%` |

## Table 3: 12-Point Sample-Complexity Legality Phase Transition at Fixed Optimization Budget (Items 6, 10, 25, 29, 30, 40)

| Dataset Size $N$ | Fixed Opt Steps | Unmasked Per-Ply Legality (With `ALT-009`) | Unmasked Full 6-Ply Rollout (With `ALT-009`) | Unmasked Per-Ply Legality (No `ALT-009`) | Unmasked Full 6-Ply Rollout (No `ALT-009`) | Post-`ALT-002` Safety Projection |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `1,000` | `96` | `12.42%` | `0.00%` | `4.58%` | `0.00%` | `100.00%` |
| `2,500` | `96` | `18.08%` | `0.00%` | `6.08%` | `0.50%` | `100.00%` |
| `5,000` | `96` | `30.00%` | `1.00%` | `13.75%` | `0.50%` | `100.00%` |
| `10,000` | `96` | `38.92%` | `2.50%` | `19.17%` | `1.00%` | `100.00%` |
| `15,000` | `96` | `50.08%` | `2.00%` | `22.75%` | `1.50%` | `100.00%` |
| `25,000` | `96` | `63.50%` | `10.00%` | `40.17%` | `5.00%` | `100.00%` |
| `35,000` | `96` | `74.67%` | `23.50%` | `44.25%` | `5.00%` | `100.00%` |
| `50,000` | `96` | `81.33%` | `32.50%` | `53.50%` | `8.50%` | `100.00%` |
| `75,000` | `96` | `88.00%` | `50.00%` | `62.17%` | `13.50%` | `100.00%` |
| `100,000` | `96` | `90.67%` | `57.50%` | `66.08%` | `15.50%` | `100.00%` |
| `175,000` | `96` | `93.92%` | `68.00%` | `73.33%` | `23.50%` | `100.00%` |
| `250,000` | `96` | `93.25%` | `65.50%` | `76.17%` | `25.00%` | `100.00%` |

## Table 4: Fully Reconciled 14-Participant Connected Tournament Ledger vs. Unmodified Stockfish 19 UCI (Items 3, 16, 34, 37, 38)

*Total Pairings: `76` | Games per Pairing: `200` | Total Games: `15,200` | Total Player-Games: `30,400` | Total Points Sum: `15,200.0` | Overall Mean Score Rate: `50.0000%`*

| Rank | Participant | Category | Calibrated BayesElo | 95% CI | SE | Total Games | Total Points | Overall Score Rate | Games vs SF Anchors | Score vs SF Anchors | Games vs Neural Pool | Score vs Neural Pool |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `Stockfish_UCI_Elo_2400` | `Stockfish_19_UCI` | **`2437.5`** | `[2395.0, 2479.9]` | `21.67` | `1,600` | `1,503.5` | `93.97%` | `200` | `73.25%` | `1,400` | `96.93%` |
| 2 | `Stockfish_10000_Nodes` | `Stockfish_19_UCI` | **`2233.3`** | `[2203.0, 2263.7]` | `15.48` | `1,800` | `1,438.0` | `79.89%` | `400` | `46.37%` | `1,400` | `89.46%` |
| 3 | `Stockfish_UCI_Elo_2100` | `Stockfish_19_UCI` | **`2099.5`** | `[2073.1, 2125.9]` | `13.47` | `1,800` | `1,336.5` | `74.25%` | `400` | `54.13%` | `1,400` | `80.00%` |
| 4 | `Config_D` | `Neural_Candidate` | **`1985.9`** | `[1963.3, 2008.5]` | `11.52` | `2,600` | `1,614.5` | `62.10%` | `1,400` | `54.61%` | `1,200` | `70.83%` |
| 5 | `Config_D_AR_Cached` | `Neural_Candidate` | **`1932.5`** | `[1910.5, 1954.6]` | `11.27` | `2,600` | `1,458.5` | `56.10%` | `1,400` | `47.96%` | `1,200` | `65.58%` |
| 6 | `Stockfish_1000_Nodes` | `Stockfish_19_UCI` | **`1913.4`** | `[1890.6, 1936.2]` | `11.64` | `1,800` | `1,016.5` | `56.47%` | `400` | `47.75%` | `1,400` | `58.96%` |
| 7 | `Config_C_Large` | `Neural_Candidate` | **`1868.1`** | `[1846.4, 1889.9]` | `11.10` | `2,600` | `1,261.5` | `48.52%` | `1,400` | `43.75%` | `1,200` | `54.08%` |
| 8 | `Config_C` | `Neural_Candidate` | **`1838.6`** | `[1816.9, 1860.3]` | `11.07` | `2,600` | `1,170.0` | `45.00%` | `1,400` | `41.00%` | `1,200` | `49.67%` |
| 9 | `Stockfish_UCI_Elo_1800` | `Stockfish_19_UCI` | **`1800.0`** | `[1783.2, 1816.8]` | `8.55` | `1,800` | `830.5` | `46.14%` | `400` | `50.25%` | `1,400` | `44.96%` |
| 10 | `Config_B_Prime` | `Neural_Candidate` | **`1779.3`** | `[1757.5, 1801.1]` | `11.12` | `2,600` | `987.5` | `37.98%` | `1,400` | `35.36%` | `1,200` | `41.04%` |
| 11 | `Config_B` | `Neural_Candidate` | **`1767.2`** | `[1745.4, 1789.1]` | `11.14` | `2,600` | `951.0` | `36.58%` | `1,400` | `36.07%` | `1,200` | `37.17%` |
| 12 | `Config_A` | `Neural_Candidate` | **`1725.6`** | `[1703.5, 1747.7]` | `11.28` | `2,600` | `828.5` | `31.87%` | `1,400` | `32.07%` | `1,200` | `31.62%` |
| 13 | `Stockfish_100_Nodes` | `Stockfish_19_UCI` | **`1647.1`** | `[1623.2, 1671.1]` | `12.20` | `1,800` | `557.5` | `30.97%` | `400` | `49.00%` | `1,400` | `25.82%` |
| 14 | `Stockfish_UCI_Elo_1500` | `Stockfish_19_UCI` | **`1501.1`** | `[1471.6, 1530.7]` | `15.07` | `1,600` | `246.0` | `15.37%` | `200` | `31.75%` | `1,400` | `13.04%` |

## Table 5: Exact Fisher Covariance Wald $z$-Test Comparisons (Item 38)

| Comparison | $\Delta\text{Elo}$ | $\text{SE}_{\text{diff}}$ | 95% CI of $\Delta\text{Elo}$ | $z$-Statistic | Two-Sided Wald $p$-Value |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `Config_D_vs_Config_C` | `+147.24` | `15.98` | `[+115.92, +178.55]` | `9.216` | `3.084e-20` |
| `Config_D_vs_Config_C_Large_IsoParam` | `+117.74` | `15.99` | `[+86.40, +149.09]` | `7.361` | `1.819e-13` |
| `Config_D_vs_Config_D_AR_Cached_IsoSup` | `+53.33` | `16.11` | `[+21.75, +84.91]` | `3.310` | `9.334e-04` |
| `Config_C_vs_Config_B` | `+71.43` | `15.71` | `[+40.65, +102.22]` | `4.548` | `5.415e-06` |
| `Config_C_vs_Config_B_Prime` | `+59.34` | `15.69` | `[+28.59, +90.08]` | `3.782` | `1.553e-04` |
| `Config_B_Prime_vs_Config_A` | `+53.71` | `15.84` | `[+22.67, +84.75]` | `3.391` | `6.964e-04` |
| `Config_B_vs_Config_A` | `+41.61` | `15.86` | `[+10.53, +72.69]` | `2.624` | `8.688e-03` |

## Table 6: Game-Length Ply-Cap Sensitivity Study (Item 21)

| Ply Cap (`max_plies`) | Natural Checkmate | Stalemate / Insuff. | Threefold Repetition | 50-Move Rule | Capped & Adjudicated | `Config_D` Elo | `Config_C` Elo | $\Delta\text{Elo}(\text{D vs C})$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `cap_120_plies` (`120` plies) | `54.2%` | `8.4%` | `11.6%` | `1.2%` | `24.6%` | `1985.9` | `1838.6` | `+147.2` |
| `cap_200_plies` (`200` plies) | `68.8%` | `11.9%` | `13.5%` | `2.4%` | `3.4%` | `1982.7` | `1833.5` | `+149.1` |
| `cap_400_plies_natural_termination` (`400` plies) | `71.1%` | `12.6%` | `13.8%` | `2.5%` | `0.0%` | `1981.9` | `1832.3` | `+149.5` |

## Table 7: Secondary Transfer Domain — 2D Octilinear VLSI / Grid Multi-Pin Obstacle Routing (Items 7, 13)

| Routing Architecture | Unmasked Obstacle-Free Path Legality | Via + Sink Full Route Completion | Per-Step Action Accuracy |
| :--- | :---: | :---: | :---: |
| `Grid_Config_A_Flat_Shaw` | `17.87% ± 3.36%` | `0.53% ± 0.19%` | `23.81%` |
| `Grid_Config_B_Topo_Shaw` | `99.93% ± 0.09%` | `29.00% ± 1.18%` | `29.14%` |
| `Grid_Config_C_Topo_IPA` | `99.20% ± 0.33%` | `44.40% ± 2.63%` | `29.12%` |
| `Grid_Config_D_Topo_IPA_D3PM` | `100.00% ± 0.00%` | `39.87% ± 1.64%` | `59.98%` |
