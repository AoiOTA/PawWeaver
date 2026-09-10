# Initial q0 support discrepancy

Both stand20 evaluations exited0 and completed20s without a configured fall. This is not clean cross-engine standing: MuJoCo rear-thigh floor contacts begin at7.88s. Its post2 robot-nonfoot>5N fraction is67.15%, after excluding world/empty body columns; PhysX records0. The four foot values are net force magnitudes, not vertical loads.

Independent read-only analysis of saved states: post15–20s MuJoCo RL/RR hip angles are+.317/−.318rad, versus approximately+.010/+.010 in PhysX. Rear thigh angles are about.575 vs.747rad and torque about11 vs4.12Nm. Hip differences are already present by.5s, before thigh contact. Both have zero actions and no torque saturation; common-PD recomputation matches saved torques (mean maximum discrepancies.00024/.00014Nm, not a strict per-substep equality check).

Static mj_forward of saved10/15/20s states names floor↔RL_thigh_collision_2 and floor↔RR_thigh_collision_2; no physics time advanced, reconstructed TCP error<1.83e-8m. These are ground contacts, not fixed self pairs. Review found no demonstrated PD, limit or passive constant initialization bug. A friction-unit suspicion was withdrawn after checking the installed IsaacSim6 version and actual backend semantics. Contact/friction causality is not isolated; no parameter change is justified by this readout alone.

Sources: initial_{mujoco,physx}_stand20/initial_stand/trace.npz, respective runtime/config and actual initialization log. All evidence uses provisional robot parameters. Continued PhysX learning does not erase this independent MuJoCo limitation.
