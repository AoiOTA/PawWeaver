# Fixed 25% pose-amplitude course — paired baseline and bounded training

Initial MuJoCo CPU dev11 baseline and saved-state readout both completed with exit0. Root has explicitly handed GPU ownership from full_pose_learning to amplitude_course and authorized the fixed course. Initial PhysX dev11 and readout/support also exited0. Both engines completed10/11. Training was stopped normally by root after checkpoint250 exposed loss of commanded motion:326 completed new iterations, last index325,32,047,104 transitions; trainer exited0 and saved checkpoint325/final bundle. Final dev11 evaluations and support/FK both completed with exit0 using this actual STOP policy, not a500-iteration result. GPU is released; this bounded experiment is finished.

Question: can neutral500 learn smaller non-neutral EE changes while retaining supported commanded motion? Original full28 transfer degraded tracking/motion: the saved high-target pose reward was approximately .000882 even at broad reward widths, while lateral cases retained reward but fell and std increased. Shrinking pose displacement is a bounded curriculum candidate, not proof of reachability or a common explanation for those failures.

`prepare.py` transforms the exact 28 E3 train trajectories. Every position is `reset + .25 * (original - reset)`; every orientation is the common original initial rotation composed with .25 of its shortest relative rotation vector, equivalent to per-frame SLERP. Seven neutral targets remain unchanged, including serialized quaternion arrays. Timestamps and velocity commands remain element-for-element unchanged. Original metadata, training groups and source identities remain, with amplitude .25, original path/hash and explicit train split. `config.json` copies neutral_learning config and changes only the demonstrations list. Sampling .30 stand/.70 move, reward weights/width configuration, learnable std, entropy, PD, q0, architecture, action transforms and fall criteria are preserved.

The final evaluation selection is **dev11**: all seven neutral command trajectories plus low_stand, high_stand, lateral_stand and lateral_arc. Both initial and final evaluations use this same selection and PhysX batch size11. The initial policy is the unchanged neutral500 bundle, evaluated freshly on this layout. The previous one-environment neutral7 result is historical context, not this experiment's exact paired baseline. No separate neutral7 or dev8 evaluation is scheduled.

After explicit resource handoff, root/operator selects one phase at a time with `bash artifacts/runs/diagnostic_pose_learning/plan_v3_pose_amplitude25/run.sh PHASE`:

1. `initial_mujoco_dev11`, `initial_physx_dev11`: fresh full60s baseline using neutral500.
2. `train500`: `--initialize-from ../plan_v3_neutral_learning/train500/checkpoint_000499.pt`; 500 new iterations ×4096 environments ×24 rollout steps =49,152,000 transitions and10,000 optimizer updates. Actor/Critic/std/normalizers transfer; optimizer, RNG and reward-width course are fresh. No resume or automatic extension.
3. Once checkpoint000250 exists, `export250`, then exactly one `checkpoint250_mujoco_dev11`. Index250 contains251 completed new iterations. No concurrent PhysX evaluation; one operator controls the shared resource.
4. After successful actual training exit, `final_mujoco_dev11`, `final_physx_dev11`.
5. For each completed evaluation: `env -u PYTHONPATH PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' /home/lyb/miniconda3/envs/pawweaver-runtime/bin/python artifacts/runs/diagnostic_pose_learning/plan_v3_pose_amplitude25/readout.py PHASE`.

The runner uses the existing commanded evaluators `scripts/evaluate_commanded_mujoco.py` and `scripts/evaluate_commanded_isaac.py`, not generic evaluators. Readout reuses E3 full-window metrics and support/FK with its suite root bound to this directory. Inspect per-case pose tracking, actual command response, body height/tilt, real foot and nonfoot contacts and support redistribution; keep failed fragments distinct from complete windows. Surviving60s or a low average error alone does not establish purposeful whole-body coordination. Logs retain actual subprocess exit codes; earlier execution logs cannot be overwritten. `STOP` is the owned stop-file path; it is present after the normal stop described below.

Root must inspect actual reduced-amplitude results before deciding whether to evaluate original full-amplitude dev8 or independent-source test8. Neither is in the runner. No automatic amplitude expansion, parameter sweep, new threshold or budget extension is authorized.

B-route task frame follows base XY/yaw with ground Z fixed. Simulation velocities are preset commands; intended deployment commands come from an operator. One279-input Actor produces all18 joint targets. This is provisional engineering evidence only, not world-fixed EE-only A-route success, an independent far task, hardware validity or formal acceptance; `trained=false`. Original whole-body capability goals and acceptance values remain unchanged.

Validation: existing pawweaver-runtime environment, inherited PYTHONPATH cleared and CUDA hidden. `prepare.py` exited0:28 trajectories, finite/unit poses, common initial pose, strict .25 position/rotation transform with independent SLERP samples, unchanged neutral/time/commands/metadata, manifests and all28 actual CommandedPoseBank sample tables plus128 reset selections match. Final dev11 manifest, configured observation contract, runner paths and shell syntax are separately checked in `cpu_validation.json`. No physics test or full pytest suite was run. A local editing command initially used unavailable `python` (command not found); corrected to the existing environment interpreter before runner validation, with no simulation started.

## Executed initial MuJoCo dev11 baseline

The runner and existing readout/support both exited0 with CUDA hidden and the existing runtime environment. Ten of11 cases completed60s; lateral_arc fell at17.66s before the requested20–60s pose hold. Maximum saved-state FK position alignment error was 9.43e-07 m. This is an unchanged neutral500 policy baseline, not learning evidence.

| Case | Actual seconds | Full hold position RMSE m | Full hold orientation RMSE rad | Post20 nonfoot net >5N fraction |
|---|---:|---:|---:|---:|
| neutral_stand | 60.00 | 0.003880 | 0.003987 | 0.000 |
| neutral_forward | 60.00 | 0.007980 | 0.010836 | 0.000 |
| neutral_backward | 60.00 | 0.007571 | 0.003117 | 0.000 |
| neutral_left | 60.00 | 0.008118 | 0.003916 | 0.000 |
| neutral_right | 60.00 | 0.003857 | 0.004204 | 0.000 |
| neutral_yaw | 60.00 | 0.004398 | 0.002647 | 0.000 |
| neutral_arc | 60.00 | 0.004775 | 0.004001 | 0.000 |
| low_stand | 60.00 | 0.086606 | 0.245555 | 0.000 |
| high_stand | 60.00 | 0.145655 | 0.159479 | 0.000 |
| lateral_stand | 60.00 | 0.129112 | 0.253398 | 0.000 |
| lateral_arc | 17.66 | — | — | — |

All three complex stand holds had four feet with net force>1N throughout post20, no nonfoot-ground pairs and no robot nonfoot net>5N. Their .087–.146m position errors remain materially larger than neutral tracking. Seven neutral cases retain supported commanded movement: full8–60s forward/backward/left/right mean velocities .1914/−.1490/.1005/−.1206m/s, yaw .2586rad/s, arc .1452m/s with−.2371rad/s yaw. These are actual responses, not command acceptance declarations. Lateral_arc has no complete hold or command window; its failure fragments remain in JSON.

Root retains the decision to run PhysX and whether to start500 training after inspecting this baseline and the previous full-amplitude final results. This operator stopped after the authorized CPU evaluation/readout.

## Executed initial PhysX dev11 baseline and training launch

PhysX and support/FK both exited0 on the exact11-environment layout; maximum FK position alignment error6.219e-6m. Ten cases completed60s; lateral_arc fell15.48s before the20s hold. Low/high/lateral stand hold position RMSE .097597/.143037/.131487m and orientation .242730/.149321/.250731rad; post20 robot nonfoot net>5N fractions all0. All seven neutral cases completed60s without post20 nonfoot force>5N. Full8–60s actual forward/backward/left speeds .19843/−.15081/.11676m/s, yaw .25816rad/s, arc .15220m/s and−.23573rad/s. Right remains stationary (vy−.000162m/s for−.12 command), preserving the known gap. Original JSON keeps every full window and failed fragment.

At that historical stage, following the actual evaluation and readout exits, root-authorized fixed500 training launched with neutral499 initialization. Checkpoint250 MuJoCo dev11 is the only scheduled intermediate evaluation; root decides on any early STOP based on actual behavior. No full-amplitude/test expansion is authorized.

## Checkpoint250 actual development result

CPU export, MuJoCo dev11 and saved-state support/FK all exited0. Eleven of11 completed60s, but neutral commanded movement regressed markedly: forward .1914→.02709m/s, backward−.1490→−.000146, left .10046→.000087, right−.12060→−.000458, yaw .25865→.002533rad/s, arc(.14516m/s,−.23711rad/s)→(−.000820,−.002608). Complete holds and commands remain separate from initial lateral_arc failure fragments in checkpoint250_summary.json.

Complex hold position/orientation RMSE initial→250: low .08661m/.24556rad→.05085/.24751; high .14566/.15948→.18114/.11822; lateral .12911/.25340→.17622/.23996. Lateral_arc changes from17.66s fall to60s, but hold error .18599m/.25911rad and almost no commanded motion. Post20 robot nonfoot net>5N fractions are0 for all cases, while low has nonfoot-ground pairs in59.82% of samples; zero net-force threshold exceedance does not mean no contact. Maximum FK position alignment error7.012e-8m.

The operator reported this neutral skill loss immediately to root for an early-STOP decision. No extra test or parameter change was started. This is a survival/motion tradeoff with partial low-position improvement, not useful coordinated moving manipulation.

## Root-directed normal stop at326 completed iterations

Root touched this directory’s STOP after inspecting checkpoint250. The existing trainer finished its complete update, saved checkpoint_000325.pt and train500/bundle, then actually exited0. run.json records stopped_early/stop_file,326 completed iterations, last index325 and32,047,104 transitions. No signal, restart, extension or replacement run was used. The historical directory name train500 and requested budget500 remain, but actual final evaluation is STOP325. GPU compute-apps were empty after training exit and before final evaluator launch. Final MuJoCo and PhysX dev11 use that saved bundle.

## Final STOP325 paired result and resource release

Both final evaluators and both existing saved-state readouts actually exited0. Both engines completed11/11 full60s. Final report policy hashes match the actual STOP325 bundle. The absence of falls does not establish preserved command tracking or coordinated task success. Full before→250→STOP325 case windows, fragments, actual velocity triples, force/pair contacts, foot support, base height/tilt and FK errors are in final_summary.json; PhysX was not run at250.

| Complex stand/arc | MuJoCo position RMSE initial→250→STOP325 m | MuJoCo orientation initial→250→STOP325 rad | PhysX position initial→STOP325 m | PhysX orientation initial→STOP325 rad |
|---|---|---|---|---|
| low_stand | 0.08661→0.05085→0.05280 | 0.24556→0.24751→0.21692 | 0.09760→0.05506 | 0.24273→0.22165 |
| high_stand | 0.14566→0.18114→0.21209 | 0.15948→0.11822→0.10600 | 0.14304→0.21065 | 0.14932→0.09251 |
| lateral_stand | 0.12911→0.17622→0.21782 | 0.25340→0.23996→0.27832 | 0.13149→0.21237 | 0.25073→0.27136 |
| lateral_arc | —→0.18599→0.20685 | —→0.25911→0.27926 | —→0.20378 | —→0.27816 |

Dashes mark initial lateral_arc falls before20s (MuJoCo17.66s, PhysX15.48s), not a missing successful hold. At STOP325, low MuJoCo nonfoot-ground pair fraction is.64118 despite robot nonfoot net>5N fraction0. PhysX pair counts are unavailable, preserved as null; they must not be interpreted as zero. Neutral backward has robot nonfoot net>5N in100% of post20 samples in both engines, and MuJoCo nonfoot-ground pairs throughout. Other final cases have zero post20 nonfoot net>5N; those force summaries alone do not establish correct support.

| Neutral command component | MuJoCo initial→250→STOP325 | PhysX initial→STOP325 | Requested |
|---|---|---|---|
| forward vx m/s | 0.19141→0.02709→0.18523 | 0.19843→0.20336 | 0.2 |
| backward vx m/s | -0.14900→-0.00015→-0.00140 | -0.15081→-0.00166 | -0.15 |
| left vy m/s | 0.10046→0.00009→0.00815 | 0.11676→0.00113 | 0.12 |
| right vy m/s | -0.12060→-0.00046→-0.00015 | -0.00016→-0.00196 | -0.12 |
| yaw rad/s | 0.25865→0.00253→0.01538 | 0.25816→0.00921 | 0.3 |
| arc vx m/s | 0.14516→-0.00082→0.13793 | 0.15220→0.16604 | 0.15 |
| arc yaw rad/s | -0.23711→-0.00261→-0.03187 | -0.23573→-0.02008 | -0.25 |

Forward velocity partly recovers atSTOP325, but unintended mean yaw is+.18593rad/s MuJoCo and+.17806rad/s PhysX against zero commanded yaw. Arc forward motion also partly recovers while turning remains far below the request. This is not retention of the original neutral skill. The smaller low-position error does not offset high/lateral regression and command/contact failures.

Training actually completed326 new iterations,32,047,104 transitions and6,520 optimizer updates;993 training falls,120 in the last100 iterations, all finite checks true. Requested500 is the unused upper budget, not an achieved result. After final PhysX exit, no owned training/evaluator PID remained and nvidia-smi compute-apps was empty; GPU ownership was explicitly released to root. No full-amplitude/dev8/test8, extension, rendering, commit or push was performed. A temporary console summary assumed PhysX pair counts existed and raised TypeError; it was corrected locally to preserve null without rerunning evaluation or changing source.
