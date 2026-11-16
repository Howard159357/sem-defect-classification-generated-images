# Test statistics (public AnomalyGen v1.0.0): test set, run once

Table view of `v2_stats.json`, written by `scripts/stats_table.py`. Class-3 recall is measured on the 300 new class-3 test images; the guardrails use 150 class-4 and 76 good test images. Δ is a difference in class-3 recall. Intervals are 95% paired-bootstrap intervals (2,000 resamples of chains and images). Every interval rests on 3 chains; the pre-registered plan notes for the secondary analysis that such intervals under-cover. The t interval treats the 3 chains as the sample (2 degrees of freedom).

## Primary analysis H1 (DINOv2-L + LR, G* − R, setting chosen per chain on validation data)

n = 4 is the single pre-registered hypothesis test (passed = p < 0.05); n = 10 is described only, as set in advance.

| n | R | G* | Δ | 95% interval | p | hypothesis test | per-chain Δ | t interval | guardrail | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 0.872 | 0.911 | +0.039 | [+0.019, +0.067] | < 0.001 | passed | +0.053 / +0.037 / +0.027 | [+0.005, +0.072] | class 4 −0.002 (0 chains, 0 of 3 runs beyond −0.02), good +0.000 (0, 0 of 3) → pass | **supported** |
| 10 | 0.923 | 0.951 | +0.028 | [+0.000, +0.056] | 0.059 | not tested | +0.040 / +0.000 / +0.043 | [−0.032, +0.088] | class 4 +0.000 (0 chains, 0 of 3 runs beyond −0.02), good +0.000 (0, 0 of 3) → pass | **described only** |

## H2 (DINOv2-L + LR, G* − P*; n = 4 is a single hypothesis test, n = 10 described only)

| n | P* | G* | Δ | 95% interval | p | hypothesis test | per-chain Δ | t interval | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 0.868 | 0.911 | +0.043 | [+0.020, +0.072] | < 0.001 | passed | +0.057 / +0.050 / +0.023 | [−0.000, +0.087] | supported |
| 10 | 0.929 | 0.951 | +0.022 | [+0.006, +0.046] | 0.008 | not tested | +0.040 / +0.007 / +0.020 | [−0.019, +0.064] | described only |

## Main table: n = 4, fixed setting (200 synthetic images, weight 1 : 1; mean over chains and seeds)

| classifier | training set | class 3 | class 4 | good | balanced accuracy | runs |
|---|---|---|---|---|---|---|
| DINOv2-L + LR | R | 0.872 | 1.000 | 1.000 | 0.957 | 3 |
| DINOv2-L + LR | P | 0.860 | 1.000 | 1.000 | 0.953 | 3 |
| DINOv2-L + LR | P_box | 0.872 | 1.000 | 1.000 | 0.957 | 3 |
| DINOv2-L + LR | G | 0.912 | 0.998 | 1.000 | 0.970 | 3 |
| ResNet-18 | R | 0.683 | 0.986 | 1.000 | 0.890 | 18 |
| ResNet-18 | P | 0.746 | 0.981 | 1.000 | 0.909 | 18 |
| ResNet-18 | P_box | 0.740 | 0.984 | 1.000 | 0.908 | 18 |
| ResNet-18 | G | 0.809 | 0.968 | 1.000 | 0.926 | 18 |
| ConvNeXt-T | R | 0.693 | 0.983 | 1.000 | 0.892 | 18 |
| ConvNeXt-T | P | 0.732 | 0.969 | 1.000 | 0.900 | 18 |
| ConvNeXt-T | P_box | 0.703 | 0.975 | 1.000 | 0.893 | 18 |
| ConvNeXt-T | G | 0.830 | 0.943 | 1.000 | 0.924 | 18 |

## Secondary analysis (added after the validation results were seen; n = 4, fixed setting)

Holm: correction over the 3 S1 tests, and separately over the 6 S2 and S3 tests. SD between seeds: spread of the class-3 recall of G between training seeds (CNNs only).

| classifier | hypothesis | Δ | bootstrap interval | per-chain Δ | t interval | SD between seeds | Holm | guardrail (G vs R) | class-3 judgement | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| DINOv2-L + LR | S1 (G − R) | +0.040 | [+0.017, +0.073] | +0.063 / +0.033 / +0.023 | [−0.012, +0.092] | n/a | passed | class 4 −0.002 (0 chains, 0 of 3 runs beyond −0.02), good +0.000 (0, 0 of 3) → pass | supported | **supported** |
| ResNet-18 | S1 (G − R) | +0.127 | [+0.071, +0.189] | +0.117 / +0.192 / +0.071 | [−0.026, +0.279] | 0.047 | passed | class 4 −0.018 (1 chain, 7 of 18 runs beyond −0.02), good +0.000 (0, 0 of 18) → pass | supported | **supported** |
| ConvNeXt-T | S1 (G − R) | +0.136 | [+0.061, +0.212] | +0.136 / +0.217 / +0.057 | [−0.062, +0.335] | 0.052 | passed | class 4 −0.041 (3 chains, 13 of 18 runs beyond −0.02), good +0.000 (0, 0 of 18) → **fail** | supported | **not supported (guardrail failed)** |
| DINOv2-L + LR | S2 (G − P) | +0.052 | [+0.027, +0.083] | +0.070 / +0.053 / +0.033 | [+0.007, +0.098] | n/a | passed | class 4 −0.002 (0 chains, 0 of 3 runs beyond −0.02), good +0.000 (0, 0 of 3) → pass | 3 chains agree in direction | **3 chains agree in direction** |
| DINOv2-L + LR | S3 (G − P_box) | +0.040 | [+0.017, +0.073] | +0.063 / +0.033 / +0.023 | [−0.012, +0.092] | n/a | passed | class 4 −0.002 (0 chains, 0 of 3 runs beyond −0.02), good +0.000 (0, 0 of 3) → pass | 3 chains agree in direction | **3 chains agree in direction** |
| ResNet-18 | S2 (G − P) | +0.064 | [+0.020, +0.124] | +0.034 / +0.128 / +0.029 | [−0.074, +0.201] | 0.047 | passed | class 4 −0.018 (1 chain, 7 of 18 runs beyond −0.02), good +0.000 (0, 0 of 18) → pass | 3 chains agree in direction | **3 chains agree in direction** |
| ResNet-18 | S3 (G − P_box) | +0.070 | [+0.021, +0.135] | +0.044 / +0.140 / +0.025 | [−0.084, +0.223] | 0.047 | passed | class 4 −0.018 (1 chain, 7 of 18 runs beyond −0.02), good +0.000 (0, 0 of 18) → pass | 3 chains agree in direction | **3 chains agree in direction** |
| ConvNeXt-T | S2 (G − P) | +0.098 | [−0.003, +0.258] | +0.031 / +0.263 / −0.001 | [−0.260, +0.456] | 0.052 | not passed | class 4 −0.041 (3 chains, 13 of 18 runs beyond −0.02), good +0.000 (0, 0 of 18) → **fail** | inconclusive (interval includes 0; chains differ in sign) | **not supported (guardrail failed)** |
| ConvNeXt-T | S3 (G − P_box) | +0.127 | [−0.006, +0.294] | +0.088 / +0.302 / −0.010 | [−0.270, +0.523] | 0.052 | not passed | class 4 −0.041 (3 chains, 13 of 18 runs beyond −0.02), good +0.000 (0, 0 of 18) → **fail** | inconclusive (interval includes 0; chains differ in sign) | **not supported (guardrail failed)** |

## Confusion matrices (image counts summed over chains and seeds)

Each cell lists how many images of the true class were called class 3, class 4 and good.

| model | true class 3 | true class 4 | true good |
|---|---|---|---|
| `dinov2\|R\|n4\|g200` | 785, 115, 0 | 0, 450, 0 | 0, 0, 228 |
| `dinov2\|P\|n4\|g200` | 774, 126, 0 | 0, 450, 0 | 0, 0, 228 |
| `dinov2\|P_box\|n4\|g200` | 785, 115, 0 | 0, 450, 0 | 0, 0, 228 |
| `dinov2\|G\|n4\|g200` | 821, 79, 0 | 1, 449, 0 | 0, 0, 228 |
| `resnet18\|R\|n4\|g200` | 3687, 1711, 2 | 14, 2663, 23 | 0, 0, 1368 |
| `resnet18\|P\|n4\|g200` | 4026, 1370, 4 | 33, 2648, 19 | 0, 0, 1368 |
| `resnet18\|P_box\|n4\|g200` | 3995, 1401, 4 | 22, 2657, 21 | 0, 0, 1368 |
| `resnet18\|G\|n4\|g200` | 4371, 1025, 4 | 69, 2614, 17 | 0, 0, 1368 |
| `convnext_tiny\|R\|n4\|g200` | 3744, 1642, 14 | 28, 2655, 17 | 0, 0, 1368 |
| `convnext_tiny\|P\|n4\|g200` | 3951, 1432, 17 | 72, 2617, 11 | 0, 0, 1368 |
| `convnext_tiny\|P_box\|n4\|g200` | 3796, 1603, 1 | 52, 2632, 16 | 0, 0, 1368 |
| `convnext_tiny\|G\|n4\|g200` | 4480, 917, 3 | 147, 2545, 8 | 0, 0, 1368 |
| `dinov2\|R\|n4\|primary` | 785, 115, 0 | 0, 450, 0 | 0, 0, 228 |
| `dinov2\|G*\|n4\|primary` | 820, 80, 0 | 1, 449, 0 | 0, 0, 228 |
| `dinov2\|P*\|n4\|primary` | 781, 119, 0 | 0, 450, 0 | 0, 0, 228 |
| `dinov2\|R\|n10\|primary` | 831, 69, 0 | 2, 448, 0 | 0, 0, 228 |
| `dinov2\|G*\|n10\|primary` | 856, 44, 0 | 2, 448, 0 | 0, 0, 228 |
| `dinov2\|P*\|n10\|primary` | 836, 64, 0 | 2, 448, 0 | 0, 0, 228 |

## Exploratory comparisons (described only, no verdict)

`*` = setting chosen on validation data; `g200`, `g50` = 200 or 50 synthetic images at weight 1 : 1; `c3_test_old` = the first-round class-3 test set (used repeatedly during design, not clean); `full224` = whole images at 224 px.

| comparison | Δ | interval | per-chain Δ |
|---|---|---|---|
| `dinov2\|P_box* - R\|n4` | +0.003 | [−0.008, +0.016] | +0.007 / +0.010 / −0.007 |
| `dinov2\|P_rot* - R\|n4` | +0.019 | [−0.003, +0.050] | +0.043 / +0.000 / +0.013 |
| `dinov2\|G* - P_rot*\|n4` | +0.020 | [+0.003, +0.043] | +0.010 / +0.037 / +0.013 |
| `dinov2\|P_box* - R\|n10` | −0.001 | [−0.008, +0.004] | −0.003 / +0.000 / +0.000 |
| `dinov2\|P_rot* - R\|n10` | +0.018 | [−0.007, +0.043] | +0.030 / −0.007 / +0.030 |
| `dinov2\|G* - P_rot*\|n10` | +0.010 | [+0.002, +0.020] | +0.010 / +0.007 / +0.013 |
| `dinov2\|G - R\|n10_g200` | +0.023 | [+0.000, +0.050] | +0.040 / +0.000 / +0.030 |
| `dinov2\|G - P\|n10_g200` | +0.019 | [+0.004, +0.038] | +0.030 / +0.007 / +0.020 |
| `dinov2\|G - P_box\|n10_g200` | +0.026 | [−0.003, +0.054] | +0.043 / −0.003 / +0.037 |
| `resnet18\|G - R\|n10_g200` | +0.044 | [+0.028, +0.060] | +0.048 / +0.043 / +0.040 |
| `resnet18\|G - P\|n10_g200` | +0.025 | [−0.001, +0.051] | +0.016 / +0.048 / +0.012 |
| `resnet18\|G - P_box\|n10_g200` | +0.034 | [+0.004, +0.062] | +0.053 / +0.047 / +0.003 |
| `convnext_tiny\|G - R\|n10_g200` | +0.051 | [+0.032, +0.073] | +0.059 / +0.041 / +0.054 |
| `convnext_tiny\|G - P\|n10_g200` | +0.037 | [+0.016, +0.059] | +0.032 / +0.039 / +0.040 |
| `convnext_tiny\|G - P_box\|n10_g200` | +0.041 | [+0.013, +0.069] | +0.018 / +0.041 / +0.063 |
| `dinov2\|G - R\|n4_g50` | +0.038 | [+0.017, +0.067] | +0.053 / +0.040 / +0.020 |
| `dinov2\|G - P\|n4_g50` | +0.043 | [+0.020, +0.071] | +0.050 / +0.057 / +0.023 |
| `dinov2\|G - P_box\|n4_g50` | +0.040 | [+0.019, +0.069] | +0.053 / +0.040 / +0.027 |
| `resnet18\|G - R\|n4_g50` | +0.084 | [+0.047, +0.117] | +0.094 / +0.108 / +0.049 |
| `resnet18\|G - P\|n4_g50` | +0.038 | [−0.010, +0.099] | +0.000 / +0.102 / +0.011 |
| `resnet18\|G - P_box\|n4_g50` | +0.005 | [−0.016, +0.026] | +0.012 / +0.007 / −0.004 |
| `convnext_tiny\|G - R\|n4_g50` | +0.070 | [−0.000, +0.159] | +0.003 / +0.163 / +0.044 |
| `convnext_tiny\|G - P\|n4_g50` | +0.038 | [−0.098, +0.248] | −0.100 / +0.258 / −0.043 |
| `convnext_tiny\|G - P_box\|n4_g50` | +0.141 | [+0.020, +0.342] | +0.031 / +0.347 / +0.046 |
| `c3_test_old\|dinov2\|G* - R\|n4` | +0.029 | [+0.007, +0.053] | +0.043 / +0.033 / +0.010 |
| `c3_test_old\|dinov2\|G* - R\|n10` | +0.017 | [−0.003, +0.040] | +0.017 / −0.003 / +0.037 |
| `c3_test_old\|dinov2\|G - R\|n4_g200` | +0.024 | [+0.007, +0.046] | +0.033 / +0.030 / +0.010 |
| `c3_test_old\|resnet18\|G - R\|n4_g200` | +0.136 | [+0.078, +0.213] | +0.111 / +0.219 / +0.079 |
| `c3_test_old\|convnext_tiny\|G - R\|n4_g200` | +0.126 | [+0.037, +0.222] | +0.123 / +0.224 / +0.029 |
| `full224\|dinov2\|G - R\|n4_g200` | +0.024 | [+0.008, +0.047] | +0.030 / +0.013 / +0.030 |
| `full224\|dinov2\|P - R\|n4_g200` | +0.008 | [−0.006, +0.021] | +0.010 / +0.007 / +0.007 |
| `full224\|dinov2\|P_box - R\|n4_g200` | +0.013 | [−0.003, +0.030] | +0.010 / +0.023 / +0.007 |
| `full224\|dinov2\|G - R\|n10_g200` | +0.012 | [+0.001, +0.027] | +0.010 / +0.020 / +0.007 |
| `full224\|dinov2\|P - R\|n10_g200` | +0.011 | [+0.000, +0.026] | +0.003 / +0.017 / +0.013 |
| `full224\|dinov2\|P_box - R\|n10_g200` | +0.012 | [+0.001, +0.026] | +0.010 / +0.007 / +0.020 |

## Gain by number of synthetic images (DINOv2-L + LR, Δ class-3 recall vs R, mean of 3 chains)

`r1`, `r2` = weight real : synthetic of 1 : 1 or 2 : 1. Columns: number of synthetic images.

| training set | 10 | 25 | 50 | 100 | 200 | 400 |
|---|---|---|---|---|---|---|
| `G\|n4\|r1` | +0.026 | +0.039 | +0.038 | +0.037 | +0.040 | +0.046 |
| `G\|n4\|r2` | +0.034 | +0.040 | +0.039 | +0.037 | +0.037 | +0.034 |
| `G\|n10\|r1` | +0.012 | +0.023 | +0.026 | +0.027 | +0.023 | +0.017 |
| `G\|n10\|r2` | +0.012 | +0.026 | +0.027 | +0.028 | +0.023 | +0.016 |
| `P\|n4\|r1` | −0.008 | −0.010 | −0.006 | −0.006 | −0.012 | −0.013 |
| `P\|n4\|r2` | +0.000 | −0.006 | −0.004 | −0.003 | −0.002 | −0.012 |
| `P\|n10\|r1` | −0.000 | +0.000 | +0.003 | +0.004 | +0.004 | +0.001 |
| `P\|n10\|r2` | −0.000 | +0.001 | +0.004 | +0.010 | +0.009 | +0.004 |
| `P_box\|n4\|r1` | −0.001 | +0.001 | −0.002 | +0.000 | +0.000 | −0.002 |
| `P_box\|n4\|r2` | +0.003 | +0.001 | +0.000 | +0.001 | +0.002 | −0.001 |
| `P_box\|n10\|r1` | −0.002 | −0.003 | −0.001 | −0.002 | −0.002 | −0.001 |
| `P_box\|n10\|r2` | +0.000 | +0.001 | −0.001 | −0.001 | −0.003 | −0.006 |
| `P_rot\|n4\|r1` | +0.006 | +0.006 | +0.009 | +0.010 | +0.013 | +0.006 |
| `P_rot\|n4\|r2` | +0.012 | +0.007 | +0.010 | +0.013 | +0.016 | +0.008 |
| `P_rot\|n10\|r1` | +0.012 | +0.011 | +0.014 | +0.012 | +0.011 | +0.011 |
| `P_rot\|n10\|r2` | +0.012 | +0.014 | +0.013 | +0.018 | +0.013 | +0.016 |

## Real images only (DINOv2-L + LR, class-3 recall, mean ± SD over sets of real images)

10 chains for n = 4 to 50; 5, 2 and 1 sets for n = 100, 200 and 500 (the ± 0.000 at 500 is a single set).

| real class-3 images | 4 | 6 | 8 | 10 | 15 | 25 | 50 | 100 | 200 | 500 |
|---|---|---|---|---|---|---|---|---|---|---|
| recall | 0.870 ± 0.036 | 0.898 ± 0.024 | 0.916 ± 0.036 | 0.926 ± 0.038 | 0.955 ± 0.035 | 0.981 ± 0.026 | 0.997 ± 0.000 | 0.996 ± 0.001 | 0.997 ± 0.000 | 0.997 ± 0.000 |

## Equivalent number of real images (point estimates, no interval)

| setting | equivalent real images |
|---|---|
| `G*\|n4` | 7.4 |
| `P*\|n4` | < 4 |
| `G*\|n10` | 14.2 |
| `P*\|n10` | 10.5 |
