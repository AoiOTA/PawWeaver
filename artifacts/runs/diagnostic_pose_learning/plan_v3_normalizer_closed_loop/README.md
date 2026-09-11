# Normalizer exchange: four CPU closed-loop episodes

New weights use amplitude25 actual STOP checkpoint 325 (326 completed iterations, 32,047,104 fresh transitions); old weights use neutral checkpoint 499. These are diagnostic derived deployment bundles, not trained checkpoints. Only normalizer `_mean`, `_var`, `_std`, `count` were exchanged. New/old source counts: {'old': 49152000, 'new': 81199104}.

Old normalization does not restore new-weight movement: forward tracking degrades and backward falls at 27.88 s. Old weights with new normalization retain most forward/backward response. This deployment intervention does not identify a unique training root cause. The actual final325 baseline differs from the earlier checkpoint250: its forward speed recovers but undesired turning remains.

Final325 backward completes60 s with nonfoot contact throughout the post20 window; it is not clean supported standing. Old weights with new normalization have zero nonfoot contact in both post20 windows.

## Command and end-effector evidence

Commands are forward +0.20 / backward −0.15 m/s, zero lateral/yaw. Velocity means use complete target-time 8.00–59.98 s / resulting state 8.02–60.00 s. EE RMSE uses complete target-time 20.00–59.98 s / state20.02–60.00 s. Fall rows show fragments only; no complete-window metric is inferred.

| W / N | Case | Actual seconds / fall | Mean vx, vy, yawdot | EE position / orientation RMSE |
|---|---|---|---|---|
| oldW_oldN | neutral_forward | 60.00 / False | 0.191407, 0.001550, 0.003082 | 0.007980 m / 0.010836 rad |
| oldW_oldN | neutral_backward | 60.00 / False | -0.149004, -0.003888, 0.014179 | 0.007571 m / 0.003117 rad |
| newW_newN | neutral_forward | 60.00 / False | 0.185231, -0.001078, 0.185930 | 0.013128 m / 0.012462 rad |
| newW_newN | neutral_backward | 60.00 / False | -0.001399, 0.000100, -0.000979 | 0.058754 m / 0.040204 rad |
| newW_oldN | neutral_forward | 60.00 / False | 0.043442, 0.053115, 0.244190 | 0.034672 m / 0.044320 rad |
| newW_oldN | neutral_backward | 27.88 / True | -0.010268, 0.019411, 0.010601 (fragment) | 0.107923 m / 0.114264 rad (fragment) |
| oldW_newN | neutral_forward | 60.00 / False | 0.156888, -0.000485, 0.039297 | 0.012013 m / 0.025050 rad |
| oldW_newN | neutral_backward | 60.00 / False | -0.135638, -0.005639, 0.015588 | 0.013736 m / 0.024850 rad |

## Actual support and FK

Existing E3 readout uses poststate20.00–60.00 s (2001 samples for complete runs), contact force >1 N per foot, nonfoot force >5 N, and actual nonfoot-ground contact pairs. FK applies existing MuJoCo hinge alignment offset −0.002 s. Fall row covers only post20.00–27.88 s. Full windows, per-foot contact fractions, foot heights and movement are retained in each support JSON.

| W / N | Case | Foot-support count fractions 0 / 1 / 2 / 3 / 4 | Nonfoot force / ground-pair fraction | FK xyz max m / orientation max rad |
|---|---|---|---|---|
| oldW_oldN | neutral_forward | 0.0000 / 0.0015 / 0.7431 / 0.2549 / 0.0005 | 0.000000 / 0.000000 | 9.43e-07 / 8.32e-08 |
| oldW_oldN | neutral_backward | 0.0000 / 0.0000 / 0.4918 / 0.1239 / 0.3843 | 0.000000 / 0.000000 | 4.67e-07 / 9.46e-08 |
| newW_newN | neutral_forward | 0.0000 / 0.0000 / 0.6242 / 0.2829 / 0.0930 | 0.000000 / 0.000000 | 1.87e-07 / 1.26e-07 |
| newW_newN | neutral_backward | 0.0000 / 0.0000 / 0.0000 / 0.0000 / 1.0000 | 1.000000 / 1.000000 | 4.43e-08 / 8.27e-08 |
| newW_oldN | neutral_forward | 0.0000 / 0.0065 / 0.1904 / 0.1589 / 0.6442 | 0.000000 / 0.000000 | 6.86e-08 / 1.19e-07 |
| newW_oldN | neutral_backward | 0.0253 / 0.0430 / 0.0278 / 0.8861 / 0.0177 | 0.936709 / 0.936709 | 4.33e-08 / 1.14e-07 |
| oldW_newN | neutral_forward | 0.0000 / 0.0000 / 0.6017 / 0.3218 / 0.0765 | 0.000000 / 0.000000 | 4.73e-07 / 1.35e-07 |
| oldW_newN | neutral_backward | 0.0000 / 0.0000 / 0.3173 / 0.2154 / 0.4673 | 0.000000 / 0.000000 | 4.83e-07 / 8.93e-08 |

## Execution and validation

`bash .../plan_v3_normalizer_closed_loop/run.sh prepare`, then `newW_oldN`, then `oldW_newN`; exact commands and raw logs have matching saved exitcodes, all 0. GPU hidden; CPU OMP/OpenBLAS/MKL single thread. Exactly four extra MuJoCo episodes, no PhysX, training, installs, commits or pushes. Baselines reuse amplitude25 initial/final MuJoCo dev11 reports and their saved full E3 readout/support; source hashes are retained.

Preparation followed actual ExportedPolicy / bundle loader paths. A loaded derived bundle matches the corresponding checkpoint with identical in-memory normalizer exchange on 60 preserved old observations (30 each forward/backward), maximum action difference0 for both. All deployed tensors equal their assigned weights/statistics source; 279-input observation and original action mapping remain unchanged. Bundle manifest/policy hashes are recomputed and checked by load_bundle. Original bundle manifests, policies and checkpoints remain byte-identical after execution (original_hashes_after.json). Source NPZ case metadata and hashes are unchanged; only relative paths in the two-case subset manifest differ.

`readout.py` reuses E3 summarize/support unchanged, binds support ROOT to this directory and loads the matching derived bundle. Readout exit0. `comparison.json` retains complete and fragment metrics separately, trace/report hashes and FK checks; no rerun was needed.

B route only: one 18-joint Actor; EE follows current base XY and yaw with ground-fixed Z. Velocity commands come from the preset simulation trajectory; deployment operator source remains unimplemented. These two neutral tasks do not show low/high/lateral whole-body capability, world-fixed EE-only success, formal acceptance or hardware validity. Stop boundary reached; no automatic expansion.
