# Rightward motion gap: existing-trace comparison

The final neutral500 policy receives the same commanded task in both engines, but follows different closed-loop trajectories. This read-only comparison found no command/frame/reset mismatch explaining PhysX's static rightward case. It does not identify a causal dynamics or policy mechanism, and supports no interface or parameter change yet.

Sources: `final_{mujoco,physx}_neutral7/neutral_{right,left,stand}/trace.npz`, the matching reports, and the actual commanded evaluators. No simulation, rendering or new instrumentation was used.

- Policy, asset and suite hashes match. All 3000 rows of time, velocity command and task position/quaternion match across engines in all three cases.
- Actual Actor inputs `observations[:,276:279]` match commands to at most 3.72e-9; previous-action observations match the preceding action exactly.
- Within each engine, stand/left/right have identical full observations during control times 0–2 s. Initial cross-engine observation/action differences are at most 4.77e-7/1.52e-6, respectively.
- At the second Actor call (control time .02 s), FR calf relative to q0 is .01966238 rad in MuJoCo versus .010523558 in PhysX, and the largest action difference is .0178031 at FR hip. This same startup difference occurs in all three cases; it is not evidence of a right-specific cause.

| Case | First saved sample with fewer than four feet above 1 N: MuJoCo | PhysX |
|---|---:|---:|
| right | 6.94 s; FR foot force zero | Four feet throughout 60 s |
| left | 5.70 s | 5.48 s |
| stand | Four feet throughout 60 s | Four feet throughout 60 s |

Rightward mean vy in saved window (6,8] s is -.05628/-.001476 m/s (MuJoCo/PhysX); in (8,10] s it is -.12015/-.001011. Thus the useful behavioral distinction is entry into sustained support transitions versus remaining nearly static.

Timing matters: row k's action/target belongs to control time k*.02 s, while its saved-state label is (k+1)*.02 s. MuJoCo body pose/contact caches precede the final 2 ms qpos integration. The table uses saved labels, not exactly synchronized contact event times. The calf comparison uses fully integrated joint q. These 50 Hz traces cannot separate that final-substep consumption difference from solver, contact or actuator response differences.

Relevant consumers are `scripts/evaluate_commanded_isaac.py`, `CommandedMujocoRunner` in `src/pawweaver/mujoco_runtime.py`, `CommandObservationBuilder` in `src/pawweaver/observations.py`, and the common task-frame mapping in `src/pawweaver/commanded_pose.py`. E3 `support_readout.py` documents the q−physics_dt*qvel FK alignment used for saved MuJoCo body/contact states.

Next decision: reuse the ongoing full28 transfer's final neutral7 evaluation. If the rightward gap persists, a focused comparison of startup-to-support-transition responses to matched states/actions would distinguish policy feedback from execution response more directly. No new probe is scheduled by this note. The existing B-frame, single 18-joint Actor and provisional-hardware boundaries remain unchanged.
