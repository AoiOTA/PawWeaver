# One-sided folded-foot reward probe

Does a one-sided cost for foot sphere bottom above same-side thigh reduce the observed folded-leg behavior while retaining task progress?

The completed E2 continuous1000 run demonstrated actual base motion and partial task learning, but prolonged folded feet and failed support transitions remain. The enabled planar feet-under-hips term cannot detect upward folding. Saved PhysX training cases body_high/step00/step02 show the left-front sphere bottom above the same-side thigh for84.34%/83.55%/81.83% of post2 samples. This motivates one bounded, matched repair rather than an unchanged extension.

Both arms use E2checkpoint500 weights, fresh Adam and UMI EMA, seed0 and initialLR copied from its actual metric. Only the new reward coefficient differs (0 vs-10m^-2). Each receives100 iterations; actual full-duration task/support results decide the next action. It is a controlled local repair, not proof of whole-body learning or a claim that one penalty solves all foot hovering. See `decision.json` for the selected budget and boundaries.

Implementation628c515 reads the actual centered foot-sphere radius only when enabled, pairs named feet/thighs and adds the one-sided square through the existing reward/dt/diagnostic consumer.60 relevant CPU tests passed; independent read-only review found no substantive issue. Control100 is the first GPU run; no outcome is assumed from these checks.

## 已完成结果：不采用候选

control/candidate各100轮实际退出0，均9,830,400交互、2,000更新、数值有限；训练跌倒183/193。六项MuJoCo与两项PhysX train10评估均实际退出0。MuJoCo候选新增lateral 4.66秒、train step01 13.54秒跌倒，控制分别完整20/60秒。PhysX high的FL足底高于大腿比例82.59%→75.55%，但位置4.42→5.73cm、最长连续折足30.92→35.50秒；其他折足案例也没有一致改善。详见summary.json逐例结果。

本项负结果已足以回答是否采用−10成本，不追加轮数、不扫描系数；剩余四项PhysX workspace4/test8明确not_run。转入条件性的E3方法比较，不宣告EE-only不可能。
