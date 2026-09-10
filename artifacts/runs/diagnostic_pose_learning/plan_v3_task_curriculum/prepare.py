"""CPU-only first E2 batch; no simulator import, training, or default changes."""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import torch
from pawweaver.assets.model import RobotTree, origin
from pawweaver.contracts import JOINT_NAMES, FOOT_NAMES
from pawweaver.task import GoalBank
from pawweaver.trajectories import Trajectory

OUT=Path(__file__).resolve().parent
ROOT=Path('/home/lyb/pawweaver')
RUNS=OUT.parent
POOL=RUNS/'wbc_random_paths/train_geometry_pool.json'
SPEC=RUNS/'umi_recipe/spec.json'
CONFIG=RUNS/'umi_recipe/config.json'
URDF=ROOT/'assets/generated/diagnostic/robot.urdf'
DT=.02
TIMES=np.arange(3001)*DT
V_MAX=.3
POSITION_TOLERANCE=.05
ENGINEERING_MARGIN=.01


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    if (OUT/'train/manifest.json').exists():
        raise FileExistsError('Prepared output exists; preserve this batch')
    tree=RobotTree.load(URDF)
    spec=json.loads(SPEC.read_text())['actuators']
    root=np.eye(4)
    root[2,3]=json.loads(CONFIG.read_text())['initial_base_height_m']
    reset=tree.forward(dict(zip(JOINT_NAMES,spec['default_pos'])),root)
    reset_p=reset['tcp'][:3,3]
    reset_r=Rotation.from_matrix(reset['tcp'][:3,:3])
    shoulder=reset['arm_link2'][:3,3]
    tool=sum(np.linalg.norm(origin(tree.joints[n])[:3,3])
             for n in ['arm_joint6','arm_gripper_base_joint','tcp_mount'])
    wrist_radius=sum(np.linalg.norm(origin(tree.joints[n])[:3,3]) for n in ['arm_joint3','arm_joint4'])
    omega_max=V_MAX/tool
    parent={j.find('child').get('link'):j for j in tree.joints.values()}
    def chain(link):
        length=0.
        while link!=tree.root_name:
            joint=parent[link]
            length+=np.linalg.norm(origin(joint)[:3,3])
            link=joint.find('parent').get('link')
        return length
    radii=np.array([float(tree.links[name].find('collision/geometry/sphere').get('radius')) for name in FOOT_NAMES])
    contacts=np.array([reset[name][:3,3] for name in FOOT_NAMES])
    contacts[:,2]=0.
    contact_bounds=np.array([chain(name)+chain('tcp') for name in FOOT_NAMES])+radii
    pool={row['pool_id']:row for row in json.loads(POOL.read_text())}
    assert all(name.startswith('train_') for name in pool)
    sources=[POOL,SPEC,CONFIG,URDF,Path(__file__)]
    entries=[]
    records=[]

    def approach(p,r,seconds=20.):
        u=np.clip(TIMES/seconds,0,1)
        s=u**3*(10-15*u+6*u*u)
        positions=reset_p+(p-reset_p)*s[:,None]
        rotations=Slerp([0.,1.],Rotation.concatenate([reset_r,r]))(s)
        assert 1.875*np.linalg.norm(p-reset_p)/seconds<=V_MAX
        assert 1.875*(reset_r.inv()*r).magnitude()/seconds<=omega_max
        return positions,rotations

    def save(case,group,p,r,source,record,hold_start,source_id=None):
        source_id=source if source_id is None else source_id
        q=r.as_quat()[:,[3,0,1,2]]
        speeds=np.linalg.norm(np.diff(p,axis=0),axis=1)/DT
        angular=(r[:-1].inv()*r[1:]).magnitude()/DT
        assert speeds.max()<=V_MAX+1e-8 and angular.max()<=omega_max+1e-8
        np.testing.assert_allclose(p[0],reset_p,atol=1e-12)
        assert (reset_r.inv()*r[0]).magnitude()<1e-12
        np.testing.assert_allclose(p[round(hold_start/DT):],np.broadcast_to(p[-1],p[round(hold_start/DT):].shape),atol=1e-12)
        meta={'split':'train','training_group':group,'case_id':case,'source_id':source_id,
              'source_training_identity':source,'task_kind':'world_tcp_pose','family':'smooth',
              'task_frame':'fixed world frame; one reset-only position alignment; no moving base reanchor',
              'hardware_validation':False,'reachability_screened':False,
              'scope':'provisional endpoint geometry only; interpolated path not dynamics or collision validated',
              'hold_start_s':hold_start,'duration_s':60.}
        path=OUT/'train'/f'{case}.npz'
        Trajectory(TIMES,p,q,meta).save(path)
        entries.append({'file':path.name,'sha256':sha(path),'case_id':case,'source_id':source_id,
                        'family':'smooth','training_group':group})
        wrist=p[-1]-tool*r[-1].as_matrix()[:,2]
        excess=float(np.linalg.norm(wrist-shoulder)-wrist_radius)
        records.append(dict(case_id=case,training_group=group,source_training_identity=source,
            target_xyz_m=p[-1].tolist(),target_quat_wxyz=q[-1].tolist(),hold_start_s=hold_start,
            hold_duration_s=60-hold_start,max_fd_speed_m_s=float(speeds.max()),
            max_fd_angular_speed_rad_s=float(angular.max()),
            fixed_stand_wrist_bound_excess_m=excess,
            fixed_stand_excess_below_position_tolerance=bool(excess<POSITION_TOLERANCE),**record))

    for i in range(4):
        path=RUNS/'references/train'/f'train_local_{i:02d}.npz'
        source=Trajectory.load(path)
        assert source.metadata['split']=='train'
        sources.append(path)
        aligned=source.align(reset_p)
        start_r=Rotation.from_quat(aligned.orientations_wxyz[0,[1,2,3,0]])
        p,r=approach(reset_p,start_r,10.)
        mask=TIMES>=10
        p[mask]=aligned.sample(np.minimum(TIMES[mask]-10,20))
        quats=r.as_quat()
        quats[mask]=aligned.sample_orientation(np.minimum(TIMES[mask]-10,20))[:,[1,2,3,0]]
        save(f'near_{i:02d}','near',p,Rotation.from_quat(quats),str(path.relative_to(ROOT)),
             {'schedule':'10s reset-orientation approach, original 20s local motion, 30s final hold'},30.,
             source_id=source.metadata.get('source_id',str(path.relative_to(ROOT))))

    cases=[('body_low','body','train_0244',None),('body_high','body','train_0018',None),
           ('body_lateral','body','train_0190',None),('step_00','step','train_0086',[1,1]),
           ('step_01','step','train_0363',[1,1]),('step_02','step','train_0081',[-1,1])]
    for case,group,pool_id,direction in cases:
        row=pool[pool_id]
        p=np.array(row['tcp_xyz_m'])
        r=Rotation.from_quat(np.array(row['tcp_quat_wxyz'])[[1,2,3,0]])
        witness_root=np.eye(4)
        witness_root[:3,:3]=Rotation.from_euler('xyz',row['base_rpy_rad']).as_matrix()
        witness_root[:3,3]=row['base_xyz_m']
        witness=tree.forward(dict(zip(JOINT_NAMES,row['joint_positions_rad'])),witness_root)['tcp']
        np.testing.assert_allclose(witness[:3,3],p,atol=1e-10)
        assert (Rotation.from_matrix(witness[:3,:3]).inv()*r).magnitude()<1e-10
        shift=np.zeros(3)
        if direction is not None:
            direction=np.array([*direction,0.],dtype=float)
            direction/=np.linalg.norm(direction)
            def excess(distance):
                return float((np.linalg.norm(p+distance*direction-contacts,axis=1)-contact_bounds).max())
            lo,hi=0.,2.
            assert excess(lo)<POSITION_TOLERANCE+ENGINEERING_MARGIN<excess(hi)
            for _ in range(60):
                mid=(lo+hi)/2
                if excess(mid)<POSITION_TOLERANCE+ENGINEERING_MARGIN:
                    lo=mid
                else:
                    hi=mid
            shift=hi*direction
        p=p+shift
        endpoint_excess=float((np.linalg.norm(p-contacts,axis=1)-contact_bounds).max())
        if group=='step':
            assert endpoint_excess>=POSITION_TOLERANCE+ENGINEERING_MARGIN-1e-12
        positions,rotations=approach(p,r)
        save(case,group,positions,rotations,f'{POOL.relative_to(ROOT)}#{pool_id}',
             {'offline_whole_geometry_translation_xyz_m':shift.tolist(),
              'source_endpoint_witness':row,'original_contact_chain_excess_m':endpoint_excess,
              'remaining_chain_excess_after_position_tolerance_m':endpoint_excess-POSITION_TOLERANCE,
              'schedule':'20s quintic approach and shortest-path SLERP, 40s final hold'},20.)

    manifest={'schema_version':2,'task_kind':'world_tcp_pose','quaternion_convention':'wxyz',
              'split':'train','count':len(entries),'reachability_screened':False,'cases':entries}
    # Exercise the actual consumer, including its reset-only world translation.
    paths=[OUT/'train'/entry['file'] for entry in entries]
    weights={'near':.4,'body':.3,'step':.3}
    bank=GoalBank(120,3000,'cpu',seed=617,demonstrations=paths,demonstrations_only=True,
                  demonstration_group_weights=weights)
    starts=torch.tensor(np.repeat(reset_p[None],120,axis=0),dtype=torch.float32)
    quats=torch.tensor(np.repeat(reset_r.as_quat()[[3,0,1,2]][None],120,axis=0),dtype=torch.float32)
    bank.reset(torch.arange(120),starts,quats)
    actual={group:0 for group in weights}
    for env_id,index in enumerate(bank.demonstration_index):
        loaded=Trajectory.load(paths[index])
        actual[loaded.metadata['training_group']]+=1
        np.testing.assert_allclose(bank.positions[env_id,:3001],loaded.positions,atol=2e-7)
        delta=(Rotation.from_quat(bank.orientations_wxyz[env_id,:3001].numpy()[:,[1,2,3,0]]).inv()
               *Rotation.from_quat(loaded.orientations_wxyz[:,[1,2,3,0]])).magnitude()
        assert delta.max()<2e-7
    assert set(bank.demonstration_index)==set(range(10))
    shifted=starts[[0]]+torch.tensor([[2.,-1.,0.]])
    bank.reset(torch.tensor([0]),shifted,quats[[0]])
    selected=Trajectory.load(paths[bank.demonstration_index[0]])
    np.testing.assert_allclose(bank.positions[0,:3001],selected.positions+[2.,-1.,0.],atol=4e-7)
    dump(OUT/'consumer_check.json',{'cpu_only':True,'gpu_used':False,'mj_step_calls':0,
         'batch':120,'configured_group_probabilities':weights,'actual_group_draw_counts':actual,
         'actual_group_fractions':{k:v/120 for k,v in actual.items()},
         'all_10_files_selected':True,'world_pose_matches_after_reset_alignment':True,
         'nonzero_origin_fixed_translation_checked':True,'source_split_train_only':True,
         'test_pool_or_test_endpoint_reads':False,'default_recipe_enabled':False})
    dump(OUT/'generation_summary.json',{'reset_tcp_xyz_m':reset_p.tolist(),
         'reset_tcp_quat_wxyz':reset_r.as_quat()[[3,0,1,2]].tolist(),'duration_s':60.,'dt_s':DT,
         'translation_pace_m_s':V_MAX,'rotation_pace_rad_s':omega_max,
         'position_tolerance_m':POSITION_TOLERANCE,'additional_engineering_margin_m':ENGINEERING_MARGIN,
         'original_ground_contacts_xyz_m':contacts.tolist(),'contact_to_tcp_chain_bound_m':contact_bounds.tolist(),
         'source_hashes':{str(path.relative_to(ROOT)):sha(path) for path in sources},'cases':records})
    dump(OUT/'train/manifest.json',manifest)
    print(json.dumps({'cases':len(entries),'actual_group_draw_counts':actual,'cpu_checks':'passed'}))


if __name__=='__main__':
    main()
