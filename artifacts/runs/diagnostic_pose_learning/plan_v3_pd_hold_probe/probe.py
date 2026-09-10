"""CPU fixed-reference lateral hold bench; engineering evidence, no Actor."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from pawweaver.contracts import JOINT_NAMES, FOOT_NAMES
from pawweaver.control import JointPD
from pawweaver.training_inputs import training_inputs
from pawweaver.task import termination_config, fall_causes

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[3]
ASSET = REPO / 'assets/generated/diagnostic'
SPEC = REPO / 'artifacts/runs/diagnostic_pose_learning/wbc_low_noise_learning/spec.json'
SOURCE = OUT.parent / 'plan_v3_commanded_pose/generation_summary.json'
CONFIG = OUT.parent / 'plan_v3_commanded_pose/config.json'


def main():
    torch.set_num_threads(1)
    manifest, spec, _ = training_inputs(ASSET, diagnostic=True, provisional_spec=SPEC)
    witness = next(x for x in json.loads(SOURCE.read_text())['endpoint_witnesses'] if x['case'] == 'lateral')['source_endpoint_witness']
    solution = json.loads((OUT / 'static_solution.json').read_text())
    tau = np.array(solution['tau_required_nm'])
    q = np.array(witness['joint_positions_rad'])
    root = ET.parse(ASSET / 'robot.xml').getroot()
    root.find('compiler').set('meshdir', str((ASSET / 'meshes').resolve()))
    for i, name in enumerate(JOINT_NAMES):
        for field in ('armature', 'damping', 'frictionloss'):
            root.find(f".//joint[@name='{name}']").set(field, str(getattr(spec, field)[i]))
        root.find(f".//motor[@name='{name}_motor']").set('ctrlrange', f'{-spec.effort[i]} {spec.effort[i]}')
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    qi = [model.joint(n).qposadr[0] for n in JOINT_NAMES]
    vi = [model.joint(n).dofadr[0] for n in JOINT_NAMES]
    mi = [model.actuator(n + '_motor').id for n in JOINT_NAMES]
    for field in ('armature', 'damping', 'frictionloss'):
        assert np.array_equal(getattr(model, 'dof_' + field)[vi], getattr(spec, field))
    assert model.opt.timestep == spec.physics_dt
    termination = termination_config(json.loads(CONFIG.read_text()))
    report = dict(evidence='provisional CPU fixed-reference dynamics; no learning, policy reachability, WBC or hardware acceptance',
                  source=witness, static_solution=solution, termination=termination,
                  minimum_joint3_kp=float(abs(tau[14]) / (q[14] - spec.lower[14])),
                  input_sha256={str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (SPEC, SOURCE, CONFIG, ASSET / 'robot.xml')}, cases={})
    for gain in (10., 30.):
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        data.qpos[:3] = witness['base_xyz_m']
        data.qpos[3:7] = Rotation.from_euler('xyz', witness['base_rpy_rad']).as_quat()[[3, 0, 1, 2]]
        data.qpos[qi] = q
        mujoco.mj_forward(model, data)
        pd = JointPD(spec, 1)
        pd.kp[14] = gain
        kp = np.array(spec.kp); kp[14] = gain
        raw_reference = q + tau / kp
        action = (torch.tensor(raw_reference, dtype=torch.float32).reshape(1, 18) - pd.default_pos) / pd.action_scale
        pd.command(action)
        reference = pd.target.clone()
        initial = data.qpos.copy()
        target_pos = data.body('tcp').xpos.copy()
        target_rot = data.body('tcp').xmat.reshape(3, 3).copy()
        rows = []
        failed = None
        for tick in range(1000):
            pd.command(action)
            for _ in range(spec.decimation):
                torque = pd.torque(torch.tensor(data.qpos[qi].copy(), dtype=torch.float32).reshape(1, 18),
                                   torch.tensor(data.qvel[vi].copy(), dtype=torch.float32).reshape(1, 18))[0].numpy()
                data.ctrl[mi] = torque
                mujoco.mj_step(model, data)
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                    raise FloatingPointError(f'Nonfinite physics at kp={gain}, time={data.time}')
            mujoco.mj_forward(model, data)
            foot_force = np.zeros(4)
            nonfoot_ground = 0
            self_contacts = 0
            for ci in range(data.ncon):
                c = data.contact[ci]
                names = [model.body(model.geom_bodyid[g]).name for g in (c.geom1, c.geom2)]
                if model.geom('floor').id in (c.geom1, c.geom2):
                    foot = next((n for n in names if n in FOOT_NAMES), None)
                    wrench = np.zeros(6); mujoco.mj_contactForce(model, data, ci, wrench)
                    if foot:
                        foot_force[FOOT_NAMES.index(foot)] += max(0., wrench[0])
                    else:
                        nonfoot_ground += 1
                else:
                    self_contacts += 1
            base = data.body('base_link')
            pos_error = np.linalg.norm(data.body('tcp').xpos - target_pos)
            angle_error = Rotation.from_matrix(target_rot.T @ data.body('tcp').xmat.reshape(3, 3)).magnitude()
            limit_error = np.maximum(np.array(spec.lower) - data.qpos[qi], data.qpos[qi] - np.array(spec.upper)).max()
            rows.append(dict(time_s=float(data.time), joint3_error_rad=float(data.qpos[qi[14]] - q[14]),
                             tcp_error_m=float(pos_error), tcp_error_rad=float(angle_error),
                             base_xyz_m=base.xpos.tolist(), base_up_z=float(base.xmat[8]),
                             foot_normal_force_n=foot_force.tolist(), nonfoot_ground_contacts=nonfoot_ground,
                             self_contacts=self_contacts, joint_limit_excess_rad=float(max(0., limit_error)),
                             torque_nm=torque.tolist(), joint_pos_rad=data.qpos[qi].tolist()))
            height, tilt = fall_causes(base.xpos[2], base.xmat[8], termination)
            if height or tilt:
                failed = dict(height=bool(height), tilt=bool(tilt), time_s=float(data.time))
                break
        name = f'kp{int(gain)}'
        (OUT / f'{name}_trajectory.json').write_text(json.dumps(rows, indent=2) + '\n')
        def rmse(field): return float(np.sqrt(np.mean(np.square([r[field] for r in rows]))))
        report['cases'][name] = dict(duration_s=float(data.time), completed_20s=len(rows) == 1000 and failed is None,
            failed=failed, initial_qpos=initial.tolist(), raw_reference_rad=raw_reference.tolist(),
            normalized_action_before_clip=action[0].tolist(),
            effective_lower_rad=np.maximum(spec.lower, np.array(spec.default_pos)-np.array(spec.action_scale)).tolist(),
            effective_upper_rad=np.minimum(spec.upper, np.array(spec.default_pos)+np.array(spec.action_scale)).tolist(),
            clipped_reference_rad=reference[0].tolist(), clipped_reference_joints=[JOINT_NAMES[i] for i in range(18) if raw_reference[i] < spec.lower[i] or raw_reference[i] > spec.upper[i]],
            joint3_error_rmse_rad=rmse('joint3_error_rad'), tcp_position_rmse_m=rmse('tcp_error_m'), tcp_orientation_rmse_rad=rmse('tcp_error_rad'),
            nonfoot_ground_contact_ticks=sum(r['nonfoot_ground_contacts'] > 0 for r in rows),
            self_contact_ticks=sum(r['self_contacts'] > 0 for r in rows),
            minimum_base_height_m=min(r['base_xyz_m'][2] for r in rows),
            max_joint_limit_excess_rad=max(r['joint_limit_excess_rad'] for r in rows), final=rows[-1])
        sampled_torque = np.array([r['torque_nm'] for r in rows])
        report['cases'][name]['readback'] = dict(
            trajectory_all_numbers_finite=all(np.isfinite(np.array(v)).all() for row in rows for v in row.values()),
            sampled_torque_limit_excess_nm=float(np.maximum(abs(sampled_torque)-np.array(spec.effort), 0).max()),
            sampled_torque_at_limit_ticks=int(np.any(abs(sampled_torque)>=np.array(spec.effort)-1e-5, axis=1).sum()),
            first_nonfoot_ground_contact_s=next((r['time_s'] for r in rows if r['nonfoot_ground_contacts']), None),
            all_four_feet_over_5n_ticks=sum(all(f>5 for f in r['foot_normal_force_n']) for r in rows), first_tick=rows[0])
        print(json.dumps(dict(case=name, result=report['cases'][name])), flush=True)
    assert report['cases']['kp10']['initial_qpos'] == report['cases']['kp30']['initial_qpos']
    (OUT / 'result.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
