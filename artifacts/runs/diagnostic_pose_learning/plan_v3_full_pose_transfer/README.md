# Neutral500 to original full EE mix: completed, not adopted

Completed after root explicitly handed over the shared GPU/simulation resource. Fixed500 training, all seven planned evaluation processes and all support/FK readouts actually exited0; GPU is released. Root decided not to adopt this candidate, not to run independenttest8 and not to extend this recipe. The first incorrect evaluator launch exited1 before simulation and remains preserved below.

Question: can the single 18-joint B-route Actor retain neutral500's supported commanded movement while learning E3's original 28 EE trajectories? `config.json` copies neutral learning and changes only `demonstrations` to the exact E3 list. Existing paths retain the exact trajectories, rather than regenerating them. Learnable std, entropy .01, stand/move .30/.70, LR 1e-4 and floor 1e-7, PD, q0, rewards, action transforms and fall criteria are unchanged. Initial mean/std config values affect fresh creation only; checkpoint loading restores learned Actor/Critic/std/normalizers without resetting them.

Initialize from `../plan_v3_neutral_learning/train500/checkpoint_000499.pt` using `--initialize-from`, fresh optimizer, RNG and UMI reward-width course. No resume. Budget is exactly 500 new iterations × 4096 × 24 = 49,152,000 transitions / 10,000 PPO updates. Checkpoint index 250 contains 251 completed new iterations; final index 499 contains 500. No automatic extension or parameter sweep. Own stop-file path is `STOP` (absent unless a stop is requested). Preserve internal/nonfinite failures and return them to root; do not relabel partial episodes as completed tasks.

After root's handoff, run each phase with `bash artifacts/runs/diagnostic_pose_learning/plan_v3_full_pose_transfer/run.sh PHASE` from the repository. The script records the exact command, original console and actual subprocess exit code, and refuses to overwrite an earlier phase log. Existing environments and a cleared inherited PYTHONPATH are used.

1. `initial_mujoco_dev8`, `initial_physx_dev8`: unchanged neutral500 exported Actor on exact E3 dev8, full 60 seconds. Initial neutral7 is the preceding experiment's final neutral7 and is reused.
2. `train500`: continuous bounded transfer. While it runs, `export250` then `checkpoint250_mujoco_dev8` once checkpoint index 250 exists. No concurrent PhysX evaluation.
3. Following actual successful training exit: `final_mujoco_dev8`, `final_physx_dev8`, `final_mujoco_neutral7`, `final_physx_neutral7`.
4. Read each saved evaluation using `env -u PYTHONPATH PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' /home/lyb/miniconda3/envs/pawweaver-runtime/bin/python artifacts/runs/diagnostic_pose_learning/plan_v3_full_pose_transfer/readout.py PHASE`. This reuses E3 full-window metrics and support/FK, binding the suite root at this caller. It preserves complete windows versus executed failure fragments, full command and 20–60 second EE hold, real contacts, foot/thigh geometry, body height/tilt and nonfoot relation. Full 60 seconds alone is not task success.

Root decides whether final development improvement warrants the existing independent-source test8. It is not scheduled automatically; there is no full28 × neutral7 × test8 Cartesian product. No new stand/smoke or unchanged-path CPU testing.

B-route EE coordinates follow base XY/yaw with fixed ground Z. Velocity commands are preset simulation trajectories; intended deployment inputs come from an operator. This is not world-fixed EE-only A-route success or an independent far target test. Low/high/lateral/orientation tasks and purposeful crouching, leaning and support redistribution remain required capabilities; unchanged provisional hardware and `trained=false` prohibit formal/hardware acceptance. This is a curriculum/recipe experiment, not causal attribution to added velocity inputs alone. No commits, push or installs by this operator.

Execution correction: first initial MuJoCo attempt exited 1 before simulation with `Unsupported observation contract`: local runner incorrectly selected position/pose evaluator instead of existing commanded evaluator. Original command, log and exit code retained under `initial_mujoco_dev8_first_attempt_*`. Local runner now selects `evaluate_commanded_mujoco.py` and `evaluate_commanded_isaac.py`; no shared source changes.

PhysX dev8 uses the same8-environment batch before/after. Final PhysX neutral7 uses1 environment (seven sequential cases), preserving the reused neutral500 baseline layout; this keeps the known right-motion gap comparison on its original path. Checkpoint CPU export uses the existing pawweaver-train dependencies with CUDA hidden, matching the predecessor exporter invocation.

## Executed baseline and transfer launch — 2026-09-11

Both corrected commanded dev8 evaluators exited0; both support/FK readouts exited0. Each engine completed5/8 full60-second episodes. Four neutral cases survive with supported motion; complex low survives with persistent nonfoot net force, while high/lateral/lateral-arc fall. Initial neutral7 is reused from the preceding neutral500 final results, including the unresolved PhysX right-motion gap.

| Engine | Complex case | Actual seconds | Full20–60 position RMSE m | Full20–60 orientation RMSE rad | Post20 nonfoot net>5N fraction |
|---|---|---:|---:|---:|---:|
| mujoco | low_stand | 60.00 | 0.54774 | 1.46519 | 1.0000 |
| mujoco | high_stand | 20.04 | — | — | 0.6667 |
| mujoco | lateral_stand | 11.52 | — | — | — |
| mujoco | lateral_arc | 9.44 | — | — | — |
| physx | low_stand | 60.00 | 0.51907 | 1.30312 | 1.0000 |
| physx | high_stand | 17.08 | — | — | — |
| physx | lateral_stand | 11.08 | — | — | — |
| physx | lateral_arc | 9.16 | — | — | — |

Dashes denote no complete requested hold; high MuJoCo has only3 post20 samples before falling, which are preserved as a fragment in JSON and are not a sustained hold. Contact support windows use poststate20..60; pose errors use target20..59.98. Full FK alignment passed actual saved states.

Training launched after both baseline simulation processes actually exited. The final checkpoint499 of neutral learning initializes the new500 run; full task development failure, rather than neutral survival alone, is the paired baseline. Source and launch identity are recorded in train500/run.json and train500_command.txt.

## Actual checkpoint250 development readout

Checkpoint index250 contains251 new iterations; export, MuJoCo dev8 and saved-state support/FK all exited0. Still5/8 full60s: low falls14.88s (initial60), high lasts60s (initial20.04), lateral16.24s (initial11.52), lateral_arc15.24s (initial9.44). High survival is not task success: hold position/orientation RMSE .80079m/.84108rad; mean baseZ .16854m/max tilt .41975rad, exactly3 foot net forces>1N throughout post20, but persistent nonfoot-ground pairs and robot nonfoot net>5N throughout. No feet above their same-side thigh origins in that hold does not establish appropriate support. This is a low collapsed posture, not demonstrated purposeful crouching for the high target.

Neutral4 remain full60 without nonfoot contact, but forward/left/yaw response .15568/.08534/.17013 versus commands .20/.12/.30 degrades; forward hold errors .05632m/.09234rad and left .03154m/.06608rad exceed initial neutral errors. Full per-case and fragment metrics are preserved in checkpoint250_mujoco_dev8_readout.json and support.json, summarized in checkpoint250_summary.json. Continue the already fixed500 budget; no test8 or extension triggered by this readout.

## Completed fixed training budget

Training process actually exited0 after500 new iterations,49,152,000 transitions and10,000 optimizer updates. All recorded finite checks passed;12,513 training falls,463 in the final100 iterations. Checkpoint499 and final bundle are saved. No extension or independenttest8 was launched; final dev8 and neutral7 determine the behavioral result. Training summary preserves source/asset/initialization identity and trends.

Root-owned [saved-observation action-factor readout](action_shift_readout.md) separately examines weights and observation statistics on existing data. Both influence the action outputs; restoring the old normalizer is not a demonstrated closed-loop repair. This operator did not repeat that analysis or add simulation/learning probes.

Root-owned [pose-reward and exploration readout](pose_reward_readout.md) uses saved evidence to distinguish the high-target weak reward from lateral failures that retain reward signal. It does not establish one common cause or authorize a new recipe. No duplicate analysis or additional simulation was performed by this operator.

Root-owned [low/high saved-trajectory height and support figure](low_high_saved_trace.png) ([plot source](plot_height_cases.py)) reads existing MuJoCo traces only. Its foot-force display is interpreted alongside the saved nonfoot readouts, without new acceptance thresholds or simulation.

Root decision after completed final dual-engine dev8: **do not adopt this full500 candidate; do not run its independenttest8 or extend the same training recipe**. The already-running final PhysX neutral7 is being completed for retention evidence, not as a prerequisite to this decision. Any subsequent experiment is separately owned and outside this run.

## Final result and completed scope

**Not adopted.** Both final dev8 engines complete6/8, versus5/8 at the initial policy; this count increase does not establish task success. Low hold position/orientation RMSE improves from .54774m/1.4652rad to .32610m/.74283rad in MuJoCo and .51907m/1.3031rad to .31755m/.69154rad in PhysX, still far from task requirements. Both lateral cases still fall: MuJoCo11.58/7.98s and PhysX10.66/8.42s.

Final high survives60s with foot support, no post20 robot nonfoot net force>5N, and zero MuJoCo nonfoot-ground pairs, but mean body height moves down to .15497/.15622m and EE errors remain .90208/.90432m and1.12779/1.14822rad (MuJoCo/PhysX). Describe this as the **wrong body-height response for the high target and EE tracking failure**, not physical collapse inferred solely from low height. It differs from checkpoint250 high, which had persistent actual MuJoCo nonfoot-ground contacts. Final MuJoCo low/high each keep3 feet>1N throughout post20; low RL foot bottom mean.19974m and high FL.08095m are elevated, with max tilts.51426/.54922rad. Zero above-thigh fraction does not imply suitable task coordination.

Final neutral7 both engines actually complete7/7×60s without post20 robot nonfoot net force>5N; MuJoCo additionally records zero nonfoot-ground pairs. Nevertheless, every one of7 position **and** orientation hold errors worsens in each engine. MuJoCo forward/arc retain useful commanded motion, but backward/right become almost static and yaw weakens. PhysX forward/backward/right are nearly static, left/yaw weak, and arc retains motion with bias. The old rightward cross-engine gap is not repaired: MuJoCo also loses rightward motion.

| Engine | Forward vx (.20) | Backward vx (−.15) | Left vy (.12) | Right vy (−.12) | Yaw rate (.30) | Arc vx / yaw (.15/−.25) |
|---|---:|---:|---:|---:|---:|---|
| MuJoCo | .18892 | −.00010 | .05370 | −.00094 | .01704 | .14628 / −.24980 |
| PhysX | −.00038 | −.00134 | .04903 | −.00027 | .00230 | .17930 / −.26148 |

Final neutral hold errors span.01972–.07249m/.02937–.10336rad MuJoCo and.02734–.09669m/.03104–.14163rad PhysX. These are complete target20..59.98s windows. Velocity responses above use target8..59.98s. PhysX neutral7 retains the initial1-environment sequential layout; its values may differ from dev8's8-environment layout and are not silently substituted.

[Readable per-case stage and support tables](results.md), [full paired data with common executed windows](paired_results.json), [compact final summary](final_summary.json), [actual process/CPU exit status](execution_status.json), and [training evidence](training_summary.json) preserve the initial/250/final evidence. Checkpoint250 is MuJoCo only and contains251 completed new iterations. Short shared windows are explicitly labeled and cannot substitute for full holds. All seven planned evaluation processes, the checkpoint export, training, and saved-state readouts exited0. No further simulations, test8, training, commits, pushes or installs were performed by this operator; the assigned experiment is complete and the GPU was explicitly released.
