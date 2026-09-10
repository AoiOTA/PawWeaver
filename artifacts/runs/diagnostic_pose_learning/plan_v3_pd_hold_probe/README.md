# Lateral source-pose PD hold probe

The kp30 candidate removes the arm_joint3 source-point torque deficit, but **does not hold the coupled robot pose for 20 seconds**. Do not integrate this gain as a demonstrated task fix. Both traces are full 20-second failures of the hold objective, despite avoiding the unchanged configured fall trigger. This is a CPU MuJoCo provisional engineering bench, with fixed joint references and no Actor, learning, WBC success, or hardware validity claim.

| Actual 20 s readout | kp10 control | kp30 candidate |
|---|---:|---:|
| Joint3 error RMSE (rad) | 0.274301 | 0.174126 |
| World TCP position RMSE (m) | 1.152807 | 1.402502 |
| World TCP orientation RMSE (rad) | 2.085450 | 2.557079 |
| First nonfoot ground contact (s) | 8.28 | 5.28 |
| Nonfoot ground contact ticks / 1000 | 584 | 735 |
| Minimum base height (m) | 0.216098 | 0.216837 |
| Maximum physical joint-limit excess (rad) | 0.027582 | 0.021562 |

Both runs exited normally in one process (OS exit 0), with finite saved trajectories. Neither triggered the existing base height <0.15m or base up-Z <0.35 fall predicate. At 50 Hz saved samples, no torque exceeded or reached the effort limits; this is sampled evidence, not a complete substep saturation count. Contacts and errors are also sampled at 50 Hz. Both ended with arm_joint3 near its lower physical limit, arm_joint2 near zero, TCP error about 1.60m, and nonfoot ground contact. No failure fragment was reported as a full run or omitted from the full-duration metrics.

Source: E3 `generation_summary.json`, lateral `train_0190` witness, full 18 joints, base [0,0,.28]m, roll .25rad. Both reset states and zero velocities are identical. The fixed world TCP reference is its initial pose. `static_solution.json` preserves the existing researcher's point-contact feasible LP solution: static balance under a mu=.8 friction pyramid, not an optimized or guaranteed dynamically realizable force distribution. Those forces derive joint references only; no external force is applied. Contact redistribution and coupled equilibrium stability remain unestablished.

For source q3=-2.770909495rad, lower=-2.9670597rad and gravity torque=-5.259050301Nm, minimum kp is 26.811342365Nm/rad. qref=qsource+tau/kp yields -3.296814525rad at kp10 (clipped), and -2.946211172rad at kp30 (in range). Actual normalized actions are -1.111139 and -.992973; the effective action/physical intersection adds no tighter bound here. All references run through unchanged `JointPD.command`, including normalized-action and physical-joint clipping, delay queue and torque clamps. Only joint3 kp and its corresponding gravity-bias reference differ. No other gain, damping, armature, joint/effort/collision/fall limit or model parameter changes. Passive parameters are set before model compilation and checked afterward.

The first 20ms confirms the local intended effect: joint3 error .015468rad becomes .000017928rad; TCP error .001693m becomes .000198m. This short improvement fails to persist in the coupled 20s hold. It supports the source-point deficit calculation, not gain-only sufficiency. A subsequent study would need to distinguish coupled arm/posture equilibrium stability and contact-load redistribution before recommending a new training recipe; no tuning or further simulation was run here.

Reproduce from repository root with the existing runtime environment (writes only this artifact directory; copy it to a fresh directory first to preserve this run):

```bash
env -u PYTHONPATH PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 /home/lyb/miniconda3/envs/pawweaver-runtime/bin/python artifacts/runs/diagnostic_pose_learning/plan_v3_pd_hold_probe/probe.py
```

Artifacts: `probe.py`, `static_solution.json`, `result.json`, `kp10_trajectory.json`, `kp30_trajectory.json`, and `console.log`. Readback additions to the script were calculated from these saved trajectories after the physics run; they do not alter the executed physics path. No production/spec edits, GPU use, rendering, installation, commit or push.
