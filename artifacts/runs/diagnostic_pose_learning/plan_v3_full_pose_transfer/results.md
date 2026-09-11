# Full28 transfer: actual paired evidence

Each requested episode is60s. A dash in hold or command response means the full requested window was unavailable; executed fragments remain in paired_results.json. EEhold uses target20..59.98s; command response uses target8..59.98s. Support uses poststate20..60s. Checkpoint250 contains251 new training iterations and was evaluated only in MuJoCo. No intermediate PhysX values are inferred.

Full duration is not task success. Original task thresholds remain unchanged; orientation acceptance is unspecified. All results use provisional hardware and trained=false.

## mujoco_dev8

| Case | Stage | Actual seconds | EE hold m/rad | Actual vx/vy/yawdot |
|---|---|---:|---|---|
| neutral_stand | initial | 60.00 | 0.00388/0.00399 | -0.00001/0.00000/0.00001 |
| neutral_stand | 250 | 60.00 | 0.00641/0.03255 | -0.00004/0.00000/-0.00008 |
| neutral_stand | 500 | 60.00 | 0.01972/0.02937 | 0.00004/0.00006/0.00013 |
| neutral_forward | initial | 60.00 | 0.00798/0.01084 | 0.19141/0.00155/0.00308 |
| neutral_forward | 250 | 60.00 | 0.05632/0.09234 | 0.15568/0.00204/-0.01932 |
| neutral_forward | 500 | 60.00 | 0.05542/0.07594 | 0.18892/-0.00932/-0.03050 |
| neutral_left | initial | 60.00 | 0.00812/0.00392 | -0.00247/0.10046/0.00349 |
| neutral_left | 250 | 60.00 | 0.03154/0.06608 | -0.02161/0.08534/-0.04184 |
| neutral_left | 500 | 60.00 | 0.03619/0.04984 | 0.00060/0.05370/0.07934 |
| neutral_yaw | initial | 60.00 | 0.00440/0.00265 | -0.00184/-0.00020/0.25865 |
| neutral_yaw | 250 | 60.00 | 0.00623/0.05242 | 0.00492/-0.01419/0.17013 |
| neutral_yaw | 500 | 60.00 | 0.02484/0.08226 | 0.00115/-0.00023/0.01704 |
| low_stand | initial | 60.00 | 0.54774/1.46519 | -0.00320/0.00200/-0.00962 |
| low_stand | 250 | 14.88 fall | — | — |
| low_stand | 500 | 60.00 | 0.32610/0.74283 | 0.00116/-0.00132/-0.00204 |
| high_stand | initial | 20.04 fall | — | — |
| high_stand | 250 | 60.00 | 0.80079/0.84108 | -0.00054/-0.00127/-0.00445 |
| high_stand | 500 | 60.00 | 0.90208/1.12779 | 0.00070/-0.00068/-0.00457 |
| lateral_stand | initial | 11.52 fall | — | — |
| lateral_stand | 250 | 16.24 fall | — | — |
| lateral_stand | 500 | 11.58 fall | — | — |
| lateral_arc | initial | 9.44 fall | — | — |
| lateral_arc | 250 | 15.24 fall | — | — |
| lateral_arc | 500 | 7.98 fall | — | — |

| Case | Stage | Foot-force counts >1N: samples | Nonfoot net>5N fraction | Mean baseZ m | Max tilt rad | Max foot-center above-thigh fraction |
|---|---|---|---:|---:|---:|---:|
| neutral_stand | initial | 4:2001 | 0.00000 | 0.32055 | 0.00562 | 0.00000 |
| neutral_stand | 250 | 4:2001 | 0.00000 | 0.32076 | 0.01385 | 0.00000 |
| neutral_stand | 500 | 4:2001 | 0.00000 | 0.30370 | 0.02780 | 0.00000 |
| neutral_forward | initial | 1:3, 2:1487, 3:510, 4:1 | 0.00000 | 0.31409 | 0.03809 | 0.00000 |
| neutral_forward | 250 | 2:1515, 3:437, 4:49 | 0.00000 | 0.28559 | 0.05360 | 0.00000 |
| neutral_forward | 500 | 2:1197, 3:641, 4:163 | 0.00000 | 0.28827 | 0.10786 | 0.00000 |
| neutral_left | initial | 2:1045, 3:546, 4:410 | 0.00000 | 0.32490 | 0.01575 | 0.00000 |
| neutral_left | 250 | 2:890, 3:634, 4:477 | 0.00000 | 0.30394 | 0.03657 | 0.00000 |
| neutral_left | 500 | 2:213, 3:689, 4:1099 | 0.00000 | 0.29903 | 0.10702 | 0.00000 |
| neutral_yaw | initial | 2:689, 3:283, 4:1029 | 0.00000 | 0.32097 | 0.01086 | 0.00000 |
| neutral_yaw | 250 | 2:273, 3:510, 4:1218 | 0.00000 | 0.32468 | 0.00664 | 0.00000 |
| neutral_yaw | 500 | 2:10, 3:15, 4:1976 | 0.00000 | 0.29486 | 0.08569 | 0.00000 |
| low_stand | initial | 4:2001 | 1.00000 | 0.18387 | 0.26579 | 0.00000 |
| low_stand | 250 | — | — | — | — | — |
| low_stand | 500 | 3:2001 | 0.00000 | 0.21312 | 0.51426 | 0.00000 |
| high_stand | initial | 2:1, 3:2 | 0.66667 | 0.15324 | 0.31913 | 0.00000 |
| high_stand | 250 | 3:2001 | 1.00000 | 0.16854 | 0.41975 | 0.00000 |
| high_stand | 500 | 3:2001 | 0.00000 | 0.15497 | 0.54922 | 0.00000 |
| lateral_stand | initial | — | — | — | — | — |
| lateral_stand | 250 | — | — | — | — | — |
| lateral_stand | 500 | — | — | — | — | — |
| lateral_arc | initial | — | — | — | — | — |
| lateral_arc | 250 | — | — | — | — | — |
| lateral_arc | 500 | — | — | — | — | — |

## mujoco_neutral7

| Case | Stage | Actual seconds | EE hold m/rad | Actual vx/vy/yawdot |
|---|---|---:|---|---|
| neutral_stand | initial | 60.00 | 0.00388/0.00399 | -0.00001/0.00000/0.00001 |
| neutral_stand | 500 | 60.00 | 0.01972/0.02937 | 0.00004/0.00006/0.00013 |
| neutral_forward | initial | 60.00 | 0.00798/0.01084 | 0.19141/0.00155/0.00308 |
| neutral_forward | 500 | 60.00 | 0.05542/0.07594 | 0.18892/-0.00932/-0.03050 |
| neutral_backward | initial | 60.00 | 0.00757/0.00312 | -0.14900/-0.00389/0.01418 |
| neutral_backward | 500 | 60.00 | 0.07249/0.10336 | -0.00010/-0.00019/-0.00313 |
| neutral_left | initial | 60.00 | 0.00812/0.00392 | -0.00247/0.10046/0.00349 |
| neutral_left | 500 | 60.00 | 0.03619/0.04984 | 0.00060/0.05370/0.07934 |
| neutral_right | initial | 60.00 | 0.00386/0.00420 | -0.00402/-0.12060/0.00106 |
| neutral_right | 500 | 60.00 | 0.06684/0.06854 | 0.00029/-0.00094/-0.00132 |
| neutral_yaw | initial | 60.00 | 0.00440/0.00265 | -0.00184/-0.00020/0.25865 |
| neutral_yaw | 500 | 60.00 | 0.02484/0.08226 | 0.00115/-0.00023/0.01704 |
| neutral_arc | initial | 60.00 | 0.00477/0.00400 | 0.14516/-0.00773/-0.23711 |
| neutral_arc | 500 | 60.00 | 0.05262/0.09210 | 0.14628/0.00092/-0.24980 |

| Case | Stage | Foot-force counts >1N: samples | Nonfoot net>5N fraction | Mean baseZ m | Max tilt rad | Max foot-center above-thigh fraction |
|---|---|---|---:|---:|---:|---:|
| neutral_stand | initial | 4:2001 | 0.00000 | 0.32055 | 0.00562 | 0.00000 |
| neutral_stand | 500 | 4:2001 | 0.00000 | 0.30370 | 0.02780 | 0.00000 |
| neutral_forward | initial | 1:3, 2:1487, 3:510, 4:1 | 0.00000 | 0.31409 | 0.03809 | 0.00000 |
| neutral_forward | 500 | 2:1197, 3:641, 4:163 | 0.00000 | 0.28827 | 0.10786 | 0.00000 |
| neutral_backward | initial | 2:984, 3:248, 4:769 | 0.00000 | 0.32423 | 0.01194 | 0.00000 |
| neutral_backward | 500 | 4:2001 | 0.00000 | 0.27416 | 0.12193 | 0.00000 |
| neutral_left | initial | 2:1045, 3:546, 4:410 | 0.00000 | 0.32490 | 0.01575 | 0.00000 |
| neutral_left | 500 | 2:213, 3:689, 4:1099 | 0.00000 | 0.29903 | 0.10702 | 0.00000 |
| neutral_right | initial | 2:814, 3:520, 4:667 | 0.00000 | 0.31706 | 0.01682 | 0.00000 |
| neutral_right | 500 | 4:2001 | 0.00000 | 0.26976 | 0.05198 | 0.00000 |
| neutral_yaw | initial | 2:689, 3:283, 4:1029 | 0.00000 | 0.32097 | 0.01086 | 0.00000 |
| neutral_yaw | 500 | 2:10, 3:15, 4:1976 | 0.00000 | 0.29486 | 0.08569 | 0.00000 |
| neutral_arc | initial | 2:1043, 3:645, 4:313 | 0.00000 | 0.31783 | 0.01929 | 0.00000 |
| neutral_arc | 500 | 2:864, 3:848, 4:289 | 0.00000 | 0.29105 | 0.09195 | 0.00000 |

## physx_dev8

| Case | Stage | Actual seconds | EE hold m/rad | Actual vx/vy/yawdot |
|---|---|---:|---|---|
| neutral_stand | initial | 60.00 | 0.00411/0.00272 | -0.00015/-0.00000/0.00002 |
| neutral_stand | 500 | 60.00 | 0.02730/0.03090 | -0.00006/0.00019/0.00035 |
| neutral_forward | initial | 60.00 | 0.00818/0.01099 | 0.20094/0.00099/0.00424 |
| neutral_forward | 500 | 60.00 | 0.09602/0.14088 | -0.00040/0.00030/0.00015 |
| neutral_left | initial | 60.00 | 0.00894/0.00333 | -0.00403/0.11637/0.00544 |
| neutral_left | 500 | 60.00 | 0.03846/0.04024 | 0.00378/0.04963/0.07122 |
| neutral_yaw | initial | 60.00 | 0.00454/0.00253 | -0.00167/0.00191/0.25795 |
| neutral_yaw | 500 | 60.00 | 0.03244/0.08344 | -0.00016/0.00003/0.00228 |
| low_stand | initial | 60.00 | 0.51907/1.30312 | -0.00365/0.00531/-0.01021 |
| low_stand | 500 | 60.00 | 0.31755/0.69154 | -0.00007/-0.00179/-0.00552 |
| high_stand | initial | 17.08 fall | — | — |
| high_stand | 500 | 60.00 | 0.90432/1.14822 | 0.00105/0.00018/-0.00685 |
| lateral_stand | initial | 11.08 fall | — | — |
| lateral_stand | 500 | 10.66 fall | — | — |
| lateral_arc | initial | 9.16 fall | — | — |
| lateral_arc | 500 | 8.42 fall | — | — |

| Case | Stage | Foot-force counts >1N: samples | Nonfoot net>5N fraction | Mean baseZ m | Max tilt rad | Max foot-center above-thigh fraction |
|---|---|---|---:|---:|---:|---:|
| neutral_stand | initial | 4:2001 | 0.00000 | 0.32078 | 0.00465 | 0.00000 |
| neutral_stand | 500 | 4:2001 | 0.00000 | 0.29901 | 0.04512 | 0.00000 |
| neutral_forward | initial | 2:1549, 3:440, 4:12 | 0.00000 | 0.31468 | 0.03755 | 0.00000 |
| neutral_forward | 500 | 4:2001 | 0.00000 | 0.26173 | 0.12458 | 0.00000 |
| neutral_left | initial | 2:1003, 3:643, 4:355 | 0.00000 | 0.32571 | 0.01394 | 0.00000 |
| neutral_left | 500 | 3:432, 4:1569 | 0.00000 | 0.29706 | 0.10507 | 0.00000 |
| neutral_yaw | initial | 2:290, 3:587, 4:1124 | 0.00000 | 0.32102 | 0.01057 | 0.00000 |
| neutral_yaw | 500 | 4:2001 | 0.00000 | 0.29433 | 0.08748 | 0.00000 |
| low_stand | initial | 4:2001 | 1.00000 | 0.19133 | 0.31512 | 0.00000 |
| low_stand | 500 | 3:2001 | 0.00000 | 0.22672 | 0.57098 | 0.00000 |
| high_stand | initial | — | — | — | — | — |
| high_stand | 500 | 3:2001 | 0.00000 | 0.15622 | 0.54980 | 0.00000 |
| lateral_stand | initial | — | — | — | — | — |
| lateral_stand | 500 | — | — | — | — | — |
| lateral_arc | initial | — | — | — | — | — |
| lateral_arc | 500 | — | — | — | — | — |

## physx_neutral7

| Case | Stage | Actual seconds | EE hold m/rad | Actual vx/vy/yawdot |
|---|---|---:|---|---|
| neutral_stand | initial | 60.00 | 0.00411/0.00280 | -0.00001/0.00001/0.00002 |
| neutral_stand | 500 | 60.00 | 0.02734/0.03104 | -0.00008/0.00020/0.00036 |
| neutral_forward | initial | 60.00 | 0.00823/0.01102 | 0.20152/0.00078/0.00419 |
| neutral_forward | 500 | 60.00 | 0.09669/0.14163 | -0.00038/0.00032/0.00011 |
| neutral_backward | initial | 60.00 | 0.00818/0.00258 | -0.15093/-0.00025/0.01247 |
| neutral_backward | 500 | 60.00 | 0.05734/0.06827 | -0.00134/0.00101/0.00014 |
| neutral_left | initial | 60.00 | 0.00895/0.00333 | -0.00375/0.11674/0.00550 |
| neutral_left | 500 | 60.00 | 0.03840/0.04020 | 0.00368/0.04903/0.07143 |
| neutral_right | initial | 60.00 | 0.00760/0.00385 | 0.00026/-0.00019/0.00005 |
| neutral_right | 500 | 60.00 | 0.04917/0.04837 | 0.00037/-0.00027/-0.00130 |
| neutral_yaw | initial | 60.00 | 0.00453/0.00253 | -0.00156/0.00194/0.25811 |
| neutral_yaw | 500 | 60.00 | 0.03237/0.08337 | -0.00016/0.00004/0.00230 |
| neutral_arc | initial | 60.00 | 0.00512/0.00397 | 0.15212/0.00353/-0.23558 |
| neutral_arc | 500 | 60.00 | 0.05474/0.09623 | 0.17930/0.00891/-0.26148 |

| Case | Stage | Foot-force counts >1N: samples | Nonfoot net>5N fraction | Mean baseZ m | Max tilt rad | Max foot-center above-thigh fraction |
|---|---|---|---:|---:|---:|---:|
| neutral_stand | initial | 4:2001 | 0.00000 | 0.32075 | 0.00471 | 0.00000 |
| neutral_stand | 500 | 4:2001 | 0.00000 | 0.29900 | 0.04534 | 0.00000 |
| neutral_forward | initial | 1:2, 2:1565, 3:421, 4:13 | 0.00000 | 0.31452 | 0.03795 | 0.00000 |
| neutral_forward | 500 | 4:2001 | 0.00000 | 0.26105 | 0.12254 | 0.00000 |
| neutral_backward | initial | 2:485, 3:923, 4:593 | 0.00000 | 0.32450 | 0.01295 | 0.00000 |
| neutral_backward | 500 | 4:2001 | 0.00000 | 0.28504 | 0.11200 | 0.00000 |
| neutral_left | initial | 2:1005, 3:640, 4:356 | 0.00000 | 0.32573 | 0.01346 | 0.00000 |
| neutral_left | 500 | 3:389, 4:1612 | 0.00000 | 0.29716 | 0.10476 | 0.00000 |
| neutral_right | initial | 4:2001 | 0.00000 | 0.31764 | 0.02355 | 0.00000 |
| neutral_right | 500 | 4:2001 | 0.00000 | 0.28148 | 0.07764 | 0.00000 |
| neutral_yaw | initial | 2:293, 3:588, 4:1120 | 0.00000 | 0.32101 | 0.01054 | 0.00000 |
| neutral_yaw | 500 | 4:2001 | 0.00000 | 0.29438 | 0.08763 | 0.00000 |
| neutral_arc | initial | 2:1040, 3:722, 4:239 | 0.00000 | 0.31799 | 0.01932 | 0.00000 |
| neutral_arc | 500 | 2:978, 3:712, 4:311 | 0.00000 | 0.29070 | 0.09709 | 0.00000 |

Net force magnitude is not vertical load or a contact-pair identity. MuJoCo ground-pair evidence, foot heights, continuous durations and same-side thigh geometry are retained in paired_results.json and the full support files. Zero above-thigh fraction does not imply suitable support or successful tracking. The initial neutral7 reports are reused from neutral500; final PhysX neutral7 keeps its1-environment sequential layout.
