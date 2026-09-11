# Neutral-only B-route learning prerequisite

Fresh single 18-joint Actor, 279 inputs. This experiment asks whether the B-route recipe can learn actual commanded motion with a neutral EE task, after E3 failed to move. EE targets use base XY/yaw and fixed ground Z; velocity commands come from preset trajectories now, planned operator input at deployment. This is not world-fixed EE-only success, UMI reproduction, full whole-body capability or hardware acceptance. A-route and low/high/lateral/distant/orientation capability goals remain unfinished. `trained=false`.

Only four config keys differ from E3: select its seven existing neutral training demonstrations; learn std; initial joint-space std .25 rad for 12 legs and .02 rad for six arm joints; entropy coefficient .01. Stand/move sampling remains .30/.70. This is a recipe comparison, not causal attribution to any one setting. q0, PD, rewards, UMI width course, unclipped reward, termination, action transform, task frame and commands are unchanged. Existing unchanged initialization/PD evidence is reused.

Budget: continuous fresh 500 iterations, 4096 environments × 24 steps = 49,152,000 transitions and 10,000 optimizer updates. Nominal neutral exposure is twice E3's full 1000 iterations (neutral probability .25 -> 1). Actual episode durations may affect transition exposure. No resume or weight transfer. No automatic extension. Read checkpoint index 250 (251 completed iterations) and final index 499 (500). MuJoCo CPU readouts may overlap training; PhysX runs only after training exits. neutral7 references exact original files and is explicitly train-derived development data, not held-out test.

`cpu_validation.json` records successful real bank/suite loading and fresh Actor std conversion via actual action scales. The active training consumer reads the seven config paths, not a directory or manifest. Initial sampled groups 33 stand/95 move across 128 samples; configured probabilities .30/.70. This check is not learning evidence.

Stop on internal/nonfinite failure. Sustained collapse requests the existing stop-file mechanism so a complete update is saved; brief fluctuations do not establish collapse. Interpret full command windows, actual velocity, EE errors and contact support together. Supported movement returns the decision to the full EE mix; static/fall result returns the recipe decision without self-extension.

Launch (GPU ownership requires root signal):
```bash
env -u PYTHONPATH PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 /home/lyb/miniconda3/envs/pawweaver-train/bin/python scripts/train.py --asset assets/generated/diagnostic --diagnostic --provisional-spec artifacts/runs/diagnostic_pose_learning/wbc_low_noise_learning/spec.json --config artifacts/runs/diagnostic_pose_learning/plan_v3_neutral_learning/config.json --output artifacts/runs/diagnostic_pose_learning/plan_v3_neutral_learning/train500 --seed 0 --num-envs 4096 --iterations 500 --stop-file artifacts/runs/diagnostic_pose_learning/plan_v3_neutral_learning/STOP --headless
```

## Executed result — 2026-09-11

Fresh500 training and both final engines exited0. Budget completed exactly:49,152,000 transitions/10,000 optimizer updates, finite checks throughout;4,980 falls, last100 iterations0. Actual stand/move transitions14,312,948/34,839,052. PhysX evaluation used the default1 environment (seven sequential cases), preserving all results. Checkpoint250 represents251 completed iterations and MuJoCo7/7 full60s; final500 both engines7/7 full60s.

Actual neutral movement emerged, including backward at final500 (checkpoint250 backward was static). **PhysX right remains static while MuJoCo right moves**; do not claim all-command two-engine success. Return to full EE task decisions, no automatic neutral extension.

[Existing-trace comparison of the rightward gap](rightward_gap.md) confirms matching task inputs and actual command observations; the trajectories diverge in support transitions. It does not establish a causal parameter defect or justify changing the command interface.

| Engine | Case | Actual mean vx / vy / yawdot | EE position RMSE m | EE orientation RMSE rad |
|---|---|---|---:|---:|
| physx | stand | -0.0000 / 0.0000 / 0.0000 | 0.00411 | 0.00280 |
| physx | forward | 0.2015 / 0.0008 / 0.0042 | 0.00823 | 0.01102 |
| physx | backward | -0.1509 / -0.0002 / 0.0125 | 0.00818 | 0.00258 |
| physx | left | -0.0038 / 0.1167 / 0.0055 | 0.00895 | 0.00333 |
| physx | right | 0.0003 / -0.0002 / 0.0000 | 0.00760 | 0.00385 |
| physx | yaw | -0.0016 / 0.0019 / 0.2581 | 0.00453 | 0.00253 |
| physx | arc | 0.1521 / 0.0035 / -0.2356 | 0.00512 | 0.00397 |
| mujoco | stand | -0.0000 / 0.0000 / 0.0000 | 0.00388 | 0.00399 |
| mujoco | forward | 0.1914 / 0.0015 / 0.0031 | 0.00798 | 0.01084 |
| mujoco | backward | -0.1490 / -0.0039 / 0.0142 | 0.00757 | 0.00312 |
| mujoco | left | -0.0025 / 0.1005 / 0.0035 | 0.00812 | 0.00392 |
| mujoco | right | -0.0040 / -0.1206 / 0.0011 | 0.00386 | 0.00420 |
| mujoco | yaw | -0.0018 / -0.0002 / 0.2586 | 0.00440 | 0.00265 |
| mujoco | arc | 0.1452 / -0.0077 / -0.2371 | 0.00477 | 0.00400 |

Commands respectively stand(0,0,0),forward(.20,0,0),backward(-.15,0,0),left(0,.12,0),right(0,-.12,0),yaw(0,0,.30),arc(.15,0,-.25). Velocity averages cover target8..59.98s; pose RMSE target20..59.98s.

Both engines' poststate20..60s support samples have zero robot nonfoot net force>5N and no zero-foot samples. Forward has only2 PhysX/3 MuJoCo one-foot samples out of2001; other samples/cases have2–4feet>1N. PhysX right and stand keep all4feet. MuJoCo sampled nonfoot-ground pairs are zero; PhysX net forces cannot identify ground versus self-contact. Moving-case mean base heights .314–.326m, maximum tilt .0381rad. This supports foot-supported neutral motion without collapse; foot forces are net magnitudes, not vertical loads or established gait quality. Full FK alignment max3.02e-6m PhysX/9.43e-7m MuJoCo.

Existing UMI course changed from position parameter2.0/orientation8.0 at index0 to .1/1 at30, then .005/.5 by100 through499. Learnable mean action-unit std (not radians) at0/250/499:legs .2072/.05845/.03085, arm .00864/.02344/.02672. Weighted pose/linear/yaw rewards at250 were3.8186/1.9045/.9446 and final3.9219/1.9718/.9825. Actual metadata and trends are in training_summary.json; these mixed training samples do not isolate cause.

The unchanged E3 support helper initially exited1 because it searches E3's suite root; support_first_failure.txt preserves the cause. Owned support.py changes only the imported helper's suite lookup root, preserving actual geometry/contact consumers and identity checks. Re-analysis exited0; no physics rerun or source changes.

All cases are exact neutral training-source trajectories. This is a bounded recipe result under provisional parameters, `trained=false`. No new orientation acceptance threshold, full WBC claim, world-fixed EE-only result, or hardware validity.

## Complete saved-state videos

[Forward follow view](video_replay/final500_neutral_forward_follow.mp4) and [backward follow view](video_replay/final500_neutral_backward_follow.mp4):1280×720,50fps,3000frames,60.000s each; renders and full ffmpeg decodes exited0. Start/middle/end frames visually checked: complete robot and leg visibility, readable labels. ffprobe outputs, PNG samples and hashes are in video_replay/verification.json. The original wide-view MP4s are preserved.

The E3 camera fitted the entire9m path and made the robot about30pixels tall. Owned replay_follow.py adjusts only the camera to follow saved base translation at2.4m and labels FOLLOW CAMERA; identical saved states, timestamps, task frame, commands, errors and provisional/trained=false labels remain. Ground and world-coordinate trace geometry remain unchanged. These are full saved-state MuJoCo geometry replays, not new physics, policy execution or visual control. Backward shows final improvement; the earlier250 backward gap is preserved in its readout.
