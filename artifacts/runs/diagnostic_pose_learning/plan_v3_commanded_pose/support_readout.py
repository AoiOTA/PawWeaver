"""CPU full-FK support readout for saved 279-input commanded-pose evaluations."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from pawweaver.assets.model import RobotTree
from pawweaver.commanded_pose import COMMAND_TASK_FRAME,load_command_suite
from pawweaver.contracts import JOINT_NAMES,FOOT_NAMES,canonical_hash
from pawweaver.observations import CommandObservationSpec

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[3]
METHOD=ROOT.parent/'plan_v3_reward_signal/support_geometry.py'
module_spec=importlib.util.spec_from_file_location('existing_support_geometry',METHOD)
previous=importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(previous)
geometry=previous.geometry
default_check=previous.default_check
summarize=previous.summarize
hip_height_window=previous.hip_height_window
longest=previous.longest
distribution=previous.height_distribution
DT=.02


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def movement_window(mask,times,centers,base,ground,up):
    if not mask.any():
        return None
    time=times[mask];feet={}
    for index,name in enumerate(FOOT_NAMES):
        xy=centers[mask,index,:2]
        feet[name]={
            'xy_path_length_m':float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum()),
            'xy_max_displacement_from_window_start_m':float(np.linalg.norm(xy-xy[0],axis=1).max()),
            'xy_endpoint_displacement_m':float(np.linalg.norm(xy[-1]-xy[0])),
            'xy_min_m':xy.min(axis=0).tolist(),'xy_max_m':xy.max(axis=0).tolist()}
    return {'samples':len(time),'first_sample_s':float(time[0]),'last_sample_s':float(time[-1]),
            'base_height_above_ground_m':distribution(base[mask,2]-ground[mask]),
            'base_tilt_from_world_up_rad':distribution(np.arccos(np.clip(up[mask],-1,1))),
            'base_up_z':distribution(up[mask]),'feet':feet}


def case_result(folder,episode,engine,tree,spheres,physics_dt):
    case_id=episode['trajectory']['case_id']
    path=folder/case_id/'trace.npz'
    with np.load(path,allow_pickle=False) as archive:
        data={key:archive[key] for key in archive.files}
    if any(not np.isfinite(value).all() for value in data.values() if value.dtype.kind in 'fiu'):
        raise ValueError(f'Nonfinite saved trace: {path}')
    times=data['times'];n=len(times);ticks=np.rint(times/DT).astype(int)
    if (n<1 or n!=episode['actual_steps'] or not np.array_equal(ticks,np.arange(1,n+1))
            or not np.allclose(times,ticks*DT,rtol=0,atol=1e-8)):
        raise ValueError(f'Actual steps and contiguous control times differ: {path}')
    required={'observations':(n,279),'joint_positions':(n,18),'velocities':(n,18),
              'base':(n,3),'base_quat_w':(n,4),'tcp':(n,3),'tcp_quat_w':(n,4),'base_up_z':(n,)}
    if any(data[key].shape!=shape for key,shape in required.items()):
        raise ValueError(f'Expected complete 279 observation and post-state fields: {path}')
    q=data['joint_positions'].astype(float)
    if engine=='MuJoCo':
        # implicitfast saved xpos/xquat/contact precede the last qpos integration.
        q-=physics_dt*data['velocities']
    quats=data['base_quat_w'].astype(float)
    norm_error=float(np.abs(np.linalg.norm(quats,axis=1)-1).max())
    rotations=Rotation.from_quat(quats[:,[1,2,3,0]]).as_matrix()
    up=rotations[:,2,2]
    centers=np.empty((n,4,3));thighs=np.empty((n,4,3));tcp=np.empty((n,3));tcp_r=np.empty((n,3,3))
    for index,joints in enumerate(q):
        root=np.eye(4);root[:3,:3]=rotations[index];root[:3,3]=data['base'][index]
        fk=tree.forward(dict(zip(JOINT_NAMES,joints)),root)
        centers[index]=[(fk[name]@offset)[:3,3] for name,(offset,_) in zip(FOOT_NAMES,spheres)]
        thighs[index]=[fk[name.replace('_foot','_thigh')][:3,3] for name in FOOT_NAMES]
        tcp[index]=fk['tcp'][:3,3];tcp_r[index]=fk['tcp'][:3,:3]
    observed_r=Rotation.from_quat(data['tcp_quat_w'][:,[1,2,3,0]])
    angle=(Rotation.from_matrix(tcp_r).inv()*observed_r).magnitude()
    checks={'base_quaternion_norm_max_error':norm_error,
            'base_up_z_max_error':float(np.abs(up-data['base_up_z']).max()),
            'fk_tcp_world_xyz_max_error_m':float(np.linalg.norm(tcp-data['tcp'],axis=1).max()),
            'fk_tcp_world_orientation_max_error_rad':float(angle.max())}
    if any(value>1e-4 for value in checks.values()):
        raise ValueError(f'Full-FK saved-state alignment failed {path}: {checks}')
    if 'ground_z' in data:
        ground=data['ground_z']
        if ground.shape!=(n,):
            raise ValueError('ground_z must match post-state samples')
        ground_source='saved ground_z'
    elif engine=='MuJoCo':
        # This runtime explicitly initializes ground_z=zeros(1) on a flat plane.
        ground=np.zeros(n)
        ground_source='CommandedMujocoRunner fixed ground_z=0'
    else:
        raise ValueError('PhysX support readout requires saved ground_z')
    bottoms=centers[:,:,2]-np.array([radius for _,radius in spheres])
    names=data['contact_body_names'].tolist()
    if len(names)!=len(set(names)) or any(name not in names for name in FOOT_NAMES):
        raise ValueError(f'Missing or duplicate actual foot contact names: {path}')
    force=data['contacts']
    if force.shape!=(n,len(names)):
        raise ValueError('Contact columns differ from saved body names')
    foot_force=force[:,[names.index(name) for name in FOOT_NAMES]]
    nonfoot_names=[name for name in names if name not in (*FOOT_NAMES,'world','')]
    nonfoot=(force[:,[names.index(name) for name in nonfoot_names]]>5.).any(axis=1)
    pairs=data.get('nonfoot_ground_contact_count')
    if pairs is not None and (pairs.shape!=(n,) or (pairs<0).any()):
        raise ValueError('Invalid saved nonfoot ground contact-pair counts')
    windows={'all_aligned':np.ones(n,dtype=bool),
             **{f'post{seconds}':ticks>=round(seconds/DT) for seconds in (2,8,20)}}
    return {'case_id':case_id,'trace':str(path),'trace_sha256':sha(path),
            'original_episode_metrics':episode,'aligned_samples':n,'excluded_poststep_samples':0,
            'mujoco_hinge_kinematic_offset_s':-physics_dt if engine=='MuJoCo' else 0.,
            'alignment_checks':checks,'ground_height_source':ground_source,
            'robot_nonfoot_force_body_names':nonfoot_names,
            'windows':{name:summarize(mask,times,bottoms-ground[:,None],foot_force,nonfoot,pairs)
                       for name,mask in windows.items()},
            'hip_height':{name:hip_height_window(mask,times,data['base'][:,2],thighs[:,:,2],centers[:,:,2],bottoms)
                          for name,mask in windows.items()},
            'movement':{name:movement_window(mask,times,centers,data['base'],ground,up)
                        for name,mask in windows.items()},
            'whole_trace_nonfoot_force_longest':longest(nonfoot,times)}


def build(evaluation,bundle_path):
    evaluation=evaluation.resolve();bundle_path=bundle_path.resolve()
    bundle=read(bundle_path/'manifest.json')
    checksum=bundle.pop('bundle_hash')
    if canonical_hash(bundle)!=checksum or sha(bundle_path/'policy.pt')!=bundle['policy_sha256']:
        raise ValueError('Bundle manifest or policy hash differs')
    if (bundle['schema_version']!=2 or bundle['observation']!=CommandObservationSpec().to_dict()
            or bundle['training_config'].get('task_mode')!='velocity_ee_pose'):
        raise ValueError('Readout requires the explicit 279 velocity_ee_pose bundle')
    asset=REPO/'assets/generated/diagnostic';asset_manifest=read(asset/'manifest.json')
    urdf=asset/'robot.urdf'
    if bundle['asset_hash']!=asset_manifest['asset_hash'] or sha(urdf)!=asset_manifest['files']['robot.urdf']:
        raise ValueError('Bundle and current canonical diagnostic URDF identity differ')
    report=read(evaluation/'report.json')
    if (report['task_kind']!='velocity_ee_pose' or report['ee_target_frame']!=COMMAND_TASK_FRAME
            or report['engine'] not in ('MuJoCo','PhysX')
            or report['asset_hash']!=bundle['asset_hash'] or report['policy_sha256']!=bundle['policy_sha256']):
        raise ValueError('Report and bundle task/engine/asset/policy identity differ')
    # These experiment-local suites are preserved alongside their evaluations.
    suites=[p for p in ROOT.glob('*/manifest.json') if sha(p)==report['suite_sha256']]
    if not suites:
        raise ValueError('No preserved experiment suite matches report suite_sha256')
    suite=suites[0].parent
    trajectories=load_command_suite(suite)  # Checks each original NPZ checksum and B contract.
    expected=[trajectory.metadata['case_id'] for trajectory in trajectories]
    if report['case_ids']!=expected or [e['trajectory']['case_id'] for e in report['episodes']]!=expected:
        raise ValueError('Report does not retain every specified suite case in order')
    for episode,trajectory in zip(report['episodes'],trajectories):
        if (episode['task_kind']!='velocity_ee_pose' or episode['policy_sha256']!=bundle['policy_sha256']
                or episode['trajectory']!=trajectory.metadata):
            raise ValueError('Episode policy or trajectory identity differs')
    actuators=bundle['actuators']
    if actuators['joint_names']!=list(JOINT_NAMES):
        raise ValueError('Bundle joint ordering differs from the named FK inputs')
    tree=RobotTree.load(urdf)
    if tree.root_name!='base_link':
        raise ValueError('Expected canonical base_link root')
    spheres=geometry(tree)
    default=default_check(tree,np.array(actuators['default_pos']),spheres)
    cases=[case_result(evaluation,episode,report['engine'],tree,spheres,actuators['physics_dt'])
           for episode in report['episodes']]
    return {'evaluation':str(evaluation),'bundle':str(bundle_path),'engine':report['engine'],
            'task_kind':'velocity_ee_pose','observation_dim':279,
            'report_sha256':sha(evaluation/'report.json'),'bundle_hash':checksum,
            'policy_sha256':bundle['policy_sha256'],'asset_hash':bundle['asset_hash'],
            'suite':str(suite),'suite_sha256':report['suite_sha256'],
            'method_source':str(Path(__file__).resolve()),'method_source_sha256':sha(Path(__file__)),
            'reused_geometry_source':str(METHOD),'reused_geometry_source_sha256':sha(METHOD),
            'default_fk_self_check':default,'original_evaluation_summary':report['summary'],'cases':cases,
            'time_alignment':'Every saved post-state is used, including the last. MuJoCo q alone backs up physics_dt*qvel to match saved implicitfast body poses and contacts; PhysX uses direct post q.',
            'window_scope':'post2/post8/post20 include samples at or after the stated second. Missing windows are null; a single endpoint sample is not a sustained hold.',
            'geometry_scope':'Foot sphere bottom world Z is reported relative to saved ground; thigh differences use same-side world link origins. XY motion is sphere-center displacement, not contact-point slip or a footstep label.',
            'contact_scope':'50Hz net-force magnitudes, not vertical loading or contact pairs. world and empty names are excluded from robot nonfoot force statistics; saved nonfoot ground-pair counts are separate.',
            'threshold_scope':'Existing 3cm clearance, net foot force 1N, robot nonfoot force 5N and zero thigh-height difference are descriptions, not new acceptance thresholds. FK tolerance 1e-4 validates saved-state alignment only.',
            'evidence_limit':'CPU saved-state FK on a provisional model; no new simulation, hardware validity, world-fixed EE-only success or proof of gait/support success.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('evaluation','bundle','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    result=build(args.evaluation,args.bundle)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'cases':len(result['cases']),'engine':result['engine'],
        'max_fk_tcp_position_error_m':max(row['alignment_checks']['fk_tcp_world_xyz_max_error_m'] for row in result['cases']),
        'output':str(args.output)}))
