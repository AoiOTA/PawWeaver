# One-sided folded-foot reward probe

Does a one-sided cost for foot sphere bottom above same-side thigh reduce the observed folded-leg behavior while retaining task progress?

The completed E2 continuous1000 run demonstrated actual base motion and partial task learning, but prolonged folded feet and failed support transitions remain. The enabled planar feet-under-hips term cannot detect upward folding. Saved PhysX training cases body_high/step00/step02 show the left-front sphere bottom above the same-side thigh for84.34%/83.55%/81.83% of post2 samples. This motivates one bounded, matched repair rather than an unchanged extension.

Both arms use E2checkpoint500 weights, fresh Adam and UMI EMA, seed0 and initialLR copied from its actual metric. Only the new reward coefficient differs (0 vs-10m^-2). Each receives100 iterations; actual full-duration task/support results decide the next action. It is a controlled local repair, not proof of whole-body learning or a claim that one penalty solves all foot hovering. See `decision.json` for the selected budget and boundaries.
