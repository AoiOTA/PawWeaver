"""Training-only geometric endpoints and explicit preset command recipes."""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
import torch
from pawweaver.assets.model import RobotTree
from pawweaver.contracts import JOINT_NAMES
from pawweaver.commanded_pose import (CommandedPoseTrajectory,CommandedPoseBank,
    COMMAND_TASK_FRAME,COMMAND_TASK_KIND,COMMAND_UNITS)

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[3]
RUNS=OUT.parent

def dump(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def main():
    if (OUT/'train/manifest.json').exists():
        raise FileExistsError('Preserve the prepared inputs')
    e2=json.loads((RUNS/'plan_v3_task_curriculum/generation_summary.json').read_text())
    config=json.loads((RUNS/'plan_v3_task_curriculum/config_lr_floor.json').read_text())
    spec_path=RUNS/'wbc_low_noise_learning/spec.json'
    spec=json.loads(spec_path.read_text())
    tree=RobotTree.load(ROOT/'assets/generated/diagnostic/robot.urdf')
    pool_path=RUNS/'wbc_random_paths/train_geometry_pool.json'
    pool={x['pool_id']:x for x in json.loads(pool_path.read_text())}
    start=np.array(e2['reset_tcp_xyz_m'])
    start_r=Rotation.from_quat(np.array(e2['reset_tcp_quat_wxyz'])[[1,2,3,0]])
    endpoints={'neutral':(start,start_r,'q0 neutral pose')}
    witnesses=[]
    for name,identity in [('low','train_0244'),('high','train_0018'),('lateral','train_0190')]:
        row=pool[identity]
        root=np.eye(4)
        root[:3,:3]=Rotation.from_euler('xyz',row['base_rpy_rad']).as_matrix()
        root[:3,3]=row['base_xyz_m']
        fk=tree.forward(dict(zip(JOINT_NAMES,row['joint_positions_rad'])),root)['tcp']
        np.testing.assert_allclose(fk[:3,3],row['tcp_xyz_m'],atol=1e-10)
        r=Rotation.from_matrix(fk[:3,:3])
        assert (r.inv()*Rotation.from_quat(np.array(row['tcp_quat_wxyz'])[[1,2,3,0]])).magnitude()<1e-10
        yaw=Rotation.from_euler('z',row['base_rpy_rad'][2])
        origin=np.array([*row['base_xyz_m'][:2],0.])
        target=yaw.inv().apply(fk[:3,3]-origin)
        orientation=yaw.inv()*r
        endpoints[name]=(target,orientation,identity)
        witnesses.append(dict(case=name,source_identity=identity,source_endpoint_witness=row,
            target_position_t=target.tolist(),target_quat_t=orientation.as_quat()[[3,0,1,2]].tolist()))
    commands={'stand':[0.,0.,0.],'forward':[.2,0.,0.],'backward':[-.15,0.,0.],
        'left':[0.,.12,0.],'right':[0.,-.12,0.],'yaw':[0.,0.,.3],'arc':[.15,0.,-.25]}
    times=np.arange(3001)*.02
    def ramp(end):
        x=np.clip((times-2)/(end-2),0,1)
        return x**3*(10-15*x+6*x*x)
    pose_s,cmd_s=ramp(20),ramp(8)
    manifests={name:[] for name in ('train','dev8','initial_stand20')}
    for name,(target,r,source) in endpoints.items():
        p=start+(target-start)*pose_s[:,None]
        quats=Slerp([0.,1.],Rotation.concatenate([start_r,r]))(pose_s).as_quat()[:,[3,0,1,2]]
        for cname,c in commands.items():
            case=name+'_'+cname
            metadata=dict(split='train',training_group='stand' if cname=='stand' else 'move',
                case_id=case,source_id=source,command_source='preset_trajectory',
                task_kind=COMMAND_TASK_KIND,task_frame=COMMAND_TASK_FRAME,units=COMMAND_UNITS,
                hardware_validation=False,scope='Training-source geometric endpoints; paths and dynamics unvalidated',
                pose_transition_start_s=2.,pose_hold_start_s=20.,command_ramp_end_s=8.)
            trajectory=CommandedPoseTrajectory(times,p,quats,cmd_s[:,None]*c,metadata)
            path=OUT/'train'/f'{case}.npz'
            trajectory.save(path)
            entry=dict(case_id=case,path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            manifests['train'].append(entry)
            if (name=='neutral' and cname in ('stand','forward','left','yaw')) or (name,cname) in (
                    ('low','stand'),('high','stand'),('lateral','stand'),('lateral','arc')):
                manifests['dev8'].append(dict(entry,path='../train/'+path.name))
    case='initial_stand'
    stand=CommandedPoseTrajectory(times[:1001],np.broadcast_to(start,(1001,3)),
        np.broadcast_to(e2['reset_tcp_quat_wxyz'],(1001,4)),np.zeros((1001,3)),
        dict(metadata,case_id=case,source_id='q0 neutral pose',training_group='stand',
            pose_hold_start_s=0.,command_ramp_end_s=0.))
    stand.save(OUT/'initial_stand20/initial_stand.npz')
    manifests['initial_stand20'].append(dict(case_id=case,path='initial_stand.npz'))
    for name,entries in manifests.items():
        (OUT/name).mkdir(exist_ok=True)
        dump(OUT/name/'manifest.json',dict(schema_version=1,task_kind=COMMAND_TASK_KIND,
            task_frame=COMMAND_TASK_FRAME,units=COMMAND_UNITS,split='train',cases=entries,
            scope='training development readout, not held-out or formal acceptance'))
    paths=[str((OUT/'train'/entry['path']).relative_to(ROOT)) for entry in manifests['train']]
    weights={'stand':.3,'move':.7}
    bank=CommandedPoseBank(120,3000,'cpu',seed=617,demonstrations=paths,demonstration_group_weights=weights)
    bank.reset(torch.arange(120))
    for i,index in enumerate(bank.demonstration_index):
        t=bank.demonstrations[index]
        np.testing.assert_allclose(bank.positions[i,:3001],t.positions_task,atol=2e-7)
        np.testing.assert_allclose(bank.velocity_commands_yaw[i,:3001],t.velocity_commands_yaw,atol=2e-8)
    config.update(task_mode=COMMAND_TASK_KIND,demonstrations=paths,demonstration_group_weights=weights,
        zero_initial_actor_mean=True,initial_joint_std_rad=[.1]*12+[.02]*6,
        learning_rate=1e-4,learn_std=False,
        base_linear_velocity_sigma_mps=.15,base_yaw_rate_sigma_radps=.3)
    config['reward_weights'].update(base_linear_velocity_tracking=2.,base_yaw_rate_tracking=1.)
    for key in ('synthetic_orientation_source','orientation_reward_source'):
        config.pop(key,None)
    dump(OUT/'config.json',config)
    dump(OUT/'generation_summary.json',dict(count=28,duration_s=60.,ground_z=0.,
        frame=COMMAND_TASK_FRAME,endpoint_witnesses=witnesses,commands_yaw=commands,
        initial_tcp_position_t=start.tolist(),initial_tcp_quat_t=e2['reset_tcp_quat_wxyz'],
        consumer_check=dict(samples=120,positions_and_commands_match=True,
            source_train_only=True,selected_files=len(set(bank.demonstration_index)),
            actual_group_counts={g:sum(bank.demonstrations[j].metadata['training_group']==g for j in bank.demonstration_index) for g in weights}),
        source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [pool_path,spec_path,ROOT/'assets/generated/diagnostic/robot.urdf',Path(__file__)]}))
    print(json.dumps({'cases':28,'dev_cases':len(manifests['dev8']),'cpu_consumer_check':'passed'}))

if __name__=='__main__':
    main()
