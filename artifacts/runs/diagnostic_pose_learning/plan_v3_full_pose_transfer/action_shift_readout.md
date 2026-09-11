# Neutral-task action sensitivity after full-pose transfer

Read-only CPU comparison of neutral500 and transfer500 checkpoints, both index499. This diagnoses changes on stored neutral-policy states; it is not a new rollout or proof of a closed-loop cause or repair.

The actual WholeBodyActor / RSL-RL 5.0.1 path strictly loaded the279-dimensional models with CUDA hidden and one CPU thread. All10,400 observations from the preceding MuJoCo neutral stand/forward/left/right trajectories were used: target times8..59.98s,2,600 frames per case. Previous actions remain those of the original policy. Old weights and old normalization reproduce saved clipped actions to maximum absolute error4.47e-7.

Only `obs_normalizer.{_mean,_var,_std,count}` were exchanged in memory. Old/new counts are49,152,000/98,304,000. Original states were restored and checked tensor by tensor. Actor, environment and task source hashes match; the training entry hash changed, so this does not establish identical full training source.

RMS deviation from old weights plus old normalization, aggregated over all frames and the respective joint group:

| Weights / normalization | Raw action, legs / arm | Effective PD target rad, legs / arm |
|---|---:|---:|
| New / new, actual transfer | .1221 / .0950 | .1503 / .1228 |
| New / old | .1632 / .0901 | .2043 / .1339 |
| Old / new | .1000 / .0427 | .1078 / .0731 |

Effective targets use the existing JointPD action clipping, q0, scales and joint limits. For actual new/new weights, stand/forward/left/right target RMS deviations are respectively .0953/.1808/.1342/.1749rad for legs and .0821/.1220/.0662/.1853rad for the arm.

Holding old/new normalization fixed, changing weights gives raw leg/arm deviations .1632/.0901 and .1444/.0994. Holding old/new weights fixed, changing normalization gives .1000/.0427 and .1456/.0641. The interaction term is .1035/.0442. These are nonlinear sensitivities, not additive causal shares.

Both weights and normalization matter. Restoring old normalization does not restore old actions and increases leg target deviation on these stored states. Therefore freezing or replacing normalization is not an established fix; no checkpoint was changed or extra training launched. Any future normalization replacement would require a separate bounded closed-loop comparison before interpreting task effects.

Sources: `src/pawweaver/learning.py` WholeBodyActor; `src/pawweaver/control.py` JointPD; `src/pawweaver/mujoco_runtime.py` CommandedMujocoRunner; each experiment's `train500/checkpoint_000499.pt`; preceding `plan_v3_neutral_learning/final_mujoco_neutral7/neutral_{stand,forward,left,right}/trace.npz`.

This remains the provisional B route: one18-joint Actor, base-XY/yaw-following EE targets with ground-fixed Z and preset simulation velocity commands. It does not establish world-fixed EE-only performance or hardware validity.
