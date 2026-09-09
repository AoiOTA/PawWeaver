# Pinned sources and method references

## Robot assets

- AS2: `unitreerobotics/unitree_ros@7d6075f7f58588b189b940130e3edab3c839b2df`
- Piper-H: `agilexrobotics/agx_arm_urdf@f6642ce0d7872c686f29c99e9e10cd23d1d49313`
- Isaac Lab: `v3.0.0-beta2.patch1`, commit `ffff603eafc6b74264a5261cc0183d6a65390d78`
- [AS2 manual](https://support.unitree.com/home/zh/AS2_SDK_Development_Guide/about_as2)
- [Piper-H manual](https://agilexsupport.yuque.com/staff-hso6mo/alxgtf/ay1awrn2hrhn1krg?singleDoc)
- [Piper-H STEP archive entry](https://agilexsupport.yuque.com/staff-hso6mo/alxgtf/glzd7a853owrmsk0?singleDoc)
- [DC1 driver](https://github.com/orbbec/ros_astra_camera/blob/210f530144314da7c70072215bacba12130d9b20/launch/dabai_dc1.launch)

Downloaded source files retain their source licenses. `assets/upstream/provenance.json` records each URL, Git blob hash and SHA-256. Generated meshes are not represented as original authored assets.

## Control and data

- [UMI on Legs](https://umi-on-legs.github.io/), [code](https://github.com/real-stanford/umi-on-legs): world/task-frame reference trajectories and unified arm/leg joint output. Original simulator: Isaac Gym.
- [MLM](https://arxiv.org/html/2508.10538): causal trajectory prediction, proprioceptive velocity estimation, adaptive task sampling. The project implements the relevant methods; it does not claim to ship the authors' unpublished training code.
- [FastUMI](https://arxiv.org/html/2409.19499), [official tools](https://github.com/zxzm-zak/FastUMI_Data): TCP trajectory data and transformations. GoPro visual observations are not interchangeable with DC1 imagery.
- [Deep WBC](https://proceedings.mlr.press/v205/fu23a.html): unified learning and reward balancing. Its independent base commands are not used here.
- [Multi-critic twist tracking](https://arxiv.org/html/2507.08656v2): comparison for multi-objective optimization; baseline here is single-critic PPO.

