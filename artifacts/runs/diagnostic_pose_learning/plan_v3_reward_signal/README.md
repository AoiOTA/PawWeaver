# v3 execution: reward signal and first controlled repair

The user requested execution of the 2026-09-10 v3 plan. This resumes development and plan-scoped simulation after the previous stage's explicit stop. Root is the only GPU operator. World-fixed EE pose, one 18-joint Actor, causal observations, existing acceptance and provisional hardware boundaries remain unchanged. No remote publication is part of this run.

## Existing evidence reused

Actual `run.json`, `metrics.jsonl` and CPU-loaded checkpoints were inspected for `wbc_random_training/train2000`, `wbc_support_learning/train1000`, and `wbc_low_noise_learning/train1000`. All use the same 256 imported world-pose trajectories, demonstrations-only, uniform selection, 4096 environments, 24 steps, 60-second episodes, 5 epochs and 4 minibatches. Last-50 preclip means / clipped fractions were respectively 1.75644 / 11.1523%, -38.3263 / 99.8720%, and -7.89387 / 90.5817%. Latest checkpoint optimizer learning rates were 5.0625e-5, 1.70859375e-4 and 1e-5; initial LR does not describe the whole run. Support and low-noise checkpoints omit 35 and 23 later recorded iterations.

The actual GoalBank imported selection is uniform `rng.integers`; demonstrations-only bypasses adaptive family sampling. Explorer's CPU check using actual data produced identical complete position/orientation tensors for 256 resets with stage0/adaptive=false versus stage2/adaptive=true at the same seed. A separate 512-reset sample contained 132 far-extension trajectories (library 64/256). This is a missing curriculum consumer, not evidence that the earlier family-adaptive experiment failed to execute. Original data retain one reset-time fixed translation, and world goals are never reanchored to the moving base.

Existing UMI pose kernel, global EMA widths, force-distribution and feet-under-hips terms are reused. The two support terms were actually enabled at -1 in the later recipes. The no-positive-foot-load branch costs 100 before weighting, while pose reward peaks at 4; actual contribution still needs measurement. Action-rate and torque units differ from upstream; no blind coefficient transplant is selected. Existing upstream source audit remains at `artifacts/data/umi_on_legs_reference/` and prior `umi_recipe/` records.

## P0 implementation

- Both evaluators report requested and actual control steps; incomplete runs cannot pass position criteria. Missing specified cases cannot silently produce a summary. Post-transition metrics without any samples remain null; executed fragment RMSE is separate.
- `scripts/train.py --stop-file PATH` saves the last complete PPO update and exported bundle before normal early exit. SIGINT and internal failures still stop immediately. Actual completed iterations/transitions and termination reason are written to run/checkpoint metadata.
- `reward_diagnostics=true` reports actual weighted nonterminal components plus before/after-clipping means. Training logs actual optimizer LR and leg/arm action std. `umi_clip_nonnegative=false` is available for a controlled comparison; default behavior is unchanged, including 0.02 time scaling and fall event penalty.

Relevant CPU regressions and an independent read-only review passed. This does not establish physical execution or task improvement. Runtime-only tests that require tensordict remain skipped in that environment; the training environment exercised reward and training-loop tests.

## Frozen capture

```bash
env -u PYTHONPATH PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 /home/lyb/miniconda3/envs/pawweaver-train/bin/python artifacts/runs/diagnostic_pose_learning/plan_v3_reward_signal/capture.py --output artifacts/runs/diagnostic_pose_learning/plan_v3_reward_signal/low_noise251
```

This collects 128 environments for 60 global simulated seconds from the saved low-noise251 Actor and reward EMA, using stochastic actions with frozen model/normalizer and the natural training reset distribution. No optimization occurs. Existing generation thresholds label overlapping instantaneous low/high/lateral/far target bins; they are not source IDs, new acceptance conditions, or proof of task difficulty. The capture records true reward-call inputs before auto-reset, checks the actual reward sum, aggregates in memory, and writes a summary plus one bounded control-rate sample file for this diagnosis. It does not reconstruct missing old data or claim exact replay of historical training.

## KMA

The loaded stable Skill already supports authorized direct work, appropriate learning budgets, source/consumer checks and retaining both project goals. No new rule or role is needed. A real documentation inconsistency remained in dogfood's English/Chinese README: it still demanded user choice when delegation was unavailable. Only those two paragraphs were aligned to the existing rule, local commit `40cfc16`; no plugin reinstall or remote push. The subsequent PawWeaver work continues with existing roles and local fixes. This demonstrates the specific decision path in this task, not general speedup or a causal improvement in robot performance.

## Unfinished

Controlled reward-learning result, effective near/body/step curriculum, purposeful supported whole-body motion, selected-policy independent multi-seed evaluation, 60-second vision/collection, and real hardware parameter closure remain unfinished. Formal orientation limits await the user. A valid negative pilot must not be reported as full skill-learning impossibility, nor may a successful code check be reported as WBC success.

## E0 executed result and E1 decision

The first capture reached60 global seconds, then exited1 without a report; its script/log are preserved. A CPU reproduction confirmed that NumPy quantiles on boolean event arrays raise TypeError; conversion to float repaired that aggregation. The script now prints original exceptions before simulator shutdown and retains the bounded captured reward arrays. Retry `low_noise251_retry/` exited0:384,000 transitions, zero optimization,23 falls,130 resets,182.145s including aggregation. Actor and normalizer states remained exactly unchanged. Contact fields are50Hz net forces, not contact pairs.

90.5424% of nonterminal samples clipped; pose mean1.06626 versus preclip total−7.27081. Weighted force-distribution mean−5.99960, feet-under-hips−1.62306, joint margin−.37761, action-rate−.14011 and acceleration−.12914. No positive foot-Z load occurred in22,472 samples (5.8521%) and activated the−100 branch. Among361,528 positive-load samples,89.9546% still clipped. Low/high/lateral/far overlapping target bins clipped99.6022/91.8710/89.5928/87.5765%; these are mechanism samples, not completed tasks.

`counterfactual.json` recomputes only the no-load100→1 change on these same states: postclip changes in0.0549479% of all samples; overall clipping stays90.4875%. Hence reducing that raw outlier alone does not restore ordinary-state signal. E1 compares only `umi_clip_nonnegative=true/false`,100 iterations each, same low-noise initializer/fresh Adam1e-5/fresh UMI widths/fixed std/uniform256 paths/seed0/4096×24. Both configurations enable diagnostics. Each budget is9,830,400 transitions/2,000 optimizer updates. Default config and original2000 policy remain untouched.

Negative reward may incentivize earlier termination; assess task errors, full duration and contacts on fixed development workspace4 and test8, with both engines. No automatic extension follows merely because training is finite. The first two new-control metric rows exactly matched the historical low-noise numerical task/KL/clipping fields, providing a limited actual default-path check.

The fixed-group GoalBank consumer is also implemented and CPU-tested, with opt-in `demonstration_group_weights` and trajectory `training_group` labels. Unconfigured selection preserves the old RNG path; `demonstration_index` exposes the actual sampled ID. It is not enabled in E1, and its code checks do not establish a learned curriculum.
