# Pose reward signal at the final transfer width

Read-only CPU evaluation of saved MuJoCo trajectories through the actual UmiPoseReward implementation. This does not reconstruct full training reward, training task transition proportions or PPO gradients.

The actual pose term is `4 * exp(-position_error_m**2 / position_width_m2) * exp(-orientation_error_rad / orientation_width_rad)`. Checkpoint499 stores widths .1m²/4rad and global error EMA .41615m/.84308rad. It did not remain at the minimum .005/.5. Widths can widen again at resets; the last100 iteration-end records contain59 instances of .05/2,40 of .1/4 and1 of .1/2, rather than a fixed narrow course.

Using the final .1/4 width, a position-only1cm improvement with orientation held constant gives the following finite reward change. This is not a policy gradient.

| Case / target-time window | Mean pose term, peak4 | Mean gain for1cm smaller position error |
|---|---:|---:|
| high, complete20..59.98s | .000882 | .000173 |
| low, complete20..59.98s | 1.14710 | .07608 |
| lateral_stand, executed8..11.56s fragment | 1.97670 | .08836 |
| lateral_arc, final2s before fall:5.98..7.96s | 3.30409 | .05891 |
| transferred neutral7, complete20..59.98s, range across cases | 3.69842..3.95533 | .01166.. .05025 |

High hold has .90208m/1.12779rad RMSE: mean position factor .000292, orientation factor .75431, with every frame below0.1% of peak pose reward. A .1rad orientation-only improvement yields merely .0000223 additional reward. The position term is the main suppressor. This may be a consequence of the failed posture/large error, not proof of the original training failure cause. Final high has foot support without recorded nonfoot support; its wrong body-height response alone does not prove physical collapse.

Neither lateral fragment has a frame below1% of peak pose reward. Their falls at11.58s/7.98s cannot be explained by the same observed reward disappearance. Neither has a full hold window.

Original neutral500 saved holds score3.98719..3.99675 under the same final width. The degraded neutral precision loses relatively little pose reward. Uniformly broadening all tasks further therefore lacks priority evidence.

Learned mean action-space std rises from .03085 to .09554 for legs and .02672 to .12611 for the arm. Multiplying each joint std by its action_scale gives group means .03898→.12087rad and .07561→.33203rad; final arm joint2/3 are about .669/.541rad. These are Gaussian std before clipping, not executed target variance. Exploration has not vanished.

The evidence supports considering a bounded task-amplitude course for high targets. It does not establish a single cause for high and lateral failures, justify blindly freezing normalization, or prove that further uniform reward widening will help. Task-range and reward-width interventions should be distinguished. Original full tasks and acceptance remain unchanged for evaluation.

The existing original endpoints were checked against the saved train trajectories. Relative to reset TCP [.208346, approximately0, .646208]m, a25% position displacement and SLERP angle gives:

| Endpoint | Original distance m / angle rad | Original reward at reset | 25% distance m / angle rad | 25% reward at reset |
|---|---:|---:|---:|---:|
| low | .63203 / 1.95183 | .04522 | .15801 / .48796 | 2.75838 |
| high | .66213 / 1.18037 | .03714 | .16553 / .29509 | 2.82500 |
| lateral | .65311 / 2.25109 | .03200 | .16328 / .56277 | 2.66182 |

All use the same .1m²/4rad final width. This increases the stationary reset reward from about0.8–1.1% of peak to66.5–70.6%, motivating one bounded25% course. It is only geometric error and reward mapping, not dynamic feasibility or learned response. The subsequent course preserves original evaluation amplitudes and thresholds.

Sources: `src/pawweaver/task.py` UmiPoseReward; neutral/full-transfer `train500/checkpoint_000499.pt`; full-transfer training metrics and `final_mujoco_dev8` / `final_mujoco_neutral7` traces. B-frame, single18-joint Actor and provisional-hardware boundaries apply.
