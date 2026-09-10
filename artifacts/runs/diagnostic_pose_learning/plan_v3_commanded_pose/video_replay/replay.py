"""Replay every saved B-task state; check-only uses CPU URDF FK and no GL."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from pawweaver.assets.model import RobotTree
from pawweaver.commanded_pose import COMMAND_TASK_FRAME
from pawweaver.contracts import JOINT_NAMES,canonical_hash
from pawweaver.observations import CommandObservationSpec

OUT=Path(__file__).resolve().parent
RUN=OUT.parent
ROOT=OUT.parents[4]
ASSET=ROOT/'assets/generated/diagnostic'
DT=.02


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def load_states(evaluation,bundle_path,case):
    evaluation=evaluation.resolve();bundle_path=bundle_path.resolve()
    bundle=read(bundle_path/'manifest.json');bundle_hash=bundle.pop('bundle_hash')
    if canonical_hash(bundle)!=bundle_hash or sha(bundle_path/'policy.pt')!=bundle['policy_sha256']:
        raise ValueError('Bundle manifest or policy checksum differs')
    if (bundle['schema_version']!=2 or bundle['observation']!=CommandObservationSpec().to_dict()
            or bundle['training_config'].get('task_mode')!='velocity_ee_pose'):
        raise ValueError('Replay requires explicit B-task 279 observation contract')
    manifest=read(ASSET/'manifest.json')
    if manifest['asset_hash']!=bundle['asset_hash']:
        raise ValueError('Bundle and canonical asset differ')
    for name in ('robot.urdf','robot.xml'):
        if sha(ASSET/name)!=manifest['files'][name]:
            raise ValueError(f'Canonical asset checksum differs: {name}')
    report=read(evaluation/'report.json')
    if (report['task_kind']!='velocity_ee_pose' or report['ee_target_frame']!=COMMAND_TASK_FRAME
            or report['policy_sha256']!=bundle['policy_sha256'] or report['asset_hash']!=bundle['asset_hash']):
        raise ValueError('Report and bundle policy/asset/task identity differ')
    matches=[episode for episode in report['episodes'] if episode['trajectory']['case_id']==case]
    if len(matches)!=1 or case not in report['case_ids']:
        raise ValueError('Requested case is not unique in the saved report')
    path=evaluation/case/'trace.npz';metrics=read(path.with_name('metrics.json'))
    if metrics!=matches[0] or metrics['engine'] not in ('MuJoCo','PhysX'):
        raise ValueError('Case metrics differ from saved report')
    if metrics['task_kind']!='velocity_ee_pose' or metrics['policy_sha256']!=bundle['policy_sha256']:
        raise ValueError('Saved episode task or policy identity differs')
    with np.load(path,allow_pickle=False) as saved:
        trace={key:saved[key] for key in saved.files}
    if any(not np.isfinite(value).all() for value in trace.values() if value.dtype.kind in 'fiu'):
        raise ValueError('Nonfinite saved trace')
    n=len(trace['times']);ticks=np.rint(trace['times']/DT).astype(int)
    if (n<1 or n!=metrics['actual_steps'] or not np.array_equal(ticks,np.arange(1,n+1))
            or not np.allclose(trace['times'],ticks*DT,rtol=0,atol=1e-8)):
        raise ValueError('Trace is not the complete recorded 50Hz sequence')
    shapes={'observations':(n,279),'joint_positions':(n,18),'velocities':(n,18),
            'base':(n,3),'base_quat_w':(n,4),'tcp':(n,3),'tcp_quat_w':(n,4),
            'goal':(n,3),'goal_quat_w':(n,4),'velocity_commands_yaw':(n,3),
            'base_velocity_yaw':(n,3),'base_yaw_rate':(n,),'base_up_z':(n,),
            'errors':(n,),'orientation_errors_rad':(n,),'fall_height':(n,),'fall_tilt':(n,)}
    if any(trace[key].shape!=shape for key,shape in shapes.items()):
        raise ValueError('Missing or incompatible B-task trace fields')
    actuators=bundle['actuators']
    if actuators['joint_names']!=list(JOINT_NAMES):
        raise ValueError('Joint ordering differs from canonical FK')
    q=trace['joint_positions'].astype(float)
    offset=actuators['physics_dt'] if metrics['engine']=='MuJoCo' else 0.
    q-=offset*trace['velocities']
    base_quat=trace['base_quat_w'].astype(float)
    base_rotation=Rotation.from_quat(base_quat[:,[1,2,3,0]]).as_matrix()
    target_rotation=Rotation.from_quat(trace['tcp_quat_w'][:,[1,2,3,0]])
    tree=RobotTree.load(ASSET/'robot.urdf')
    position_error=orientation_error=0.
    for index,joints in enumerate(q):
        root=np.eye(4);root[:3,:3]=base_rotation[index];root[:3,3]=trace['base'][index]
        tcp=tree.forward(dict(zip(JOINT_NAMES,joints)),root)['tcp']
        position_error=max(position_error,float(np.linalg.norm(tcp[:3,3]-trace['tcp'][index])))
        orientation_error=max(orientation_error,float((Rotation.from_matrix(tcp[:3,:3]).inv()*target_rotation[index]).magnitude()))
    checks={'fk_tcp_position_residual_max_m':position_error,
            'fk_tcp_orientation_residual_max_rad':orientation_error,
            'base_up_residual_max':float(np.abs(base_rotation[:,2,2]-trace['base_up_z']).max()),
            'base_quaternion_norm_max_error':float(np.abs(np.linalg.norm(base_quat,axis=1)-1).max())}
    if any(value>1e-4 for value in checks.values()):
        raise ValueError(f'Saved-state FK alignment failed: {checks}')
    evidence=dict(evaluation=str(evaluation),bundle=str(bundle_path),case_id=case,engine=metrics['engine'],
        task_kind='velocity_ee_pose',ee_target_frame=COMMAND_TASK_FRAME,observation_dim=279,
        command_source='preset velocity_commands_yaw from the saved simulation trajectory',
        aligned_frames=n,first_replayed_s=float(trace['times'][0]),last_replayed_s=float(trace['times'][-1]),
        recorded_terminal_s=float(trace['times'][-1]),omitted_terminal_frames=0,
        alignment_source='Direct post joint_positions/base_quat_w; MuJoCo q backs up physics_dt*qvel to match saved xpos/xquat',
        mujoco_hinge_kinematic_offset_s=-offset,alignment_checks=checks,saved_metrics=metrics,
        bundle_hash=bundle_hash,policy_sha256=bundle['policy_sha256'],asset_hash=bundle['asset_hash'],
        suite_sha256=report['suite_sha256'],rendered=False,gl_context_created=False,mj_step_calls=0,
        playback='Every saved 50Hz state including the terminal frame; no smoothing or new physical trajectory',
        evidence_limit='MuJoCo geometry replay of saved states; no policy execution, visual control, hardware validity, world-fixed EE-only success or gait-success claim',
        source_hashes={str(p):sha(p) for p in (path,path.with_name('metrics.json'),evaluation/'report.json',
            bundle_path/'manifest.json',ASSET/'robot.xml',ASSET/'robot.urdf',Path(__file__))})
    return trace,metrics,q,base_quat,evidence


def render(states,output):
    # Rendering imports and context creation are unreachable from --check-only.
    import mujoco
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw,ImageFont
    trace,metrics,q,base_quat,evidence=states
    output.parent.mkdir(parents=True,exist_ok=True)
    for path in (output,output.with_suffix('.json')):
        if path.exists():
            raise FileExistsError(path)
    model=mujoco.MjModel.from_xml_path(str(ASSET/'robot.xml'));data=mujoco.MjData(model)
    addresses=np.array([model.joint(name).qposadr[0] for name in JOINT_NAMES])
    model.vis.headlight.ambient[:]=.5;model.vis.headlight.diffuse[:]=.8;model.vis.headlight.specular[:]=.2
    model.vis.global_.offwidth=1280;model.vis.global_.offheight=720
    camera=mujoco.MjvCamera()
    points=np.concatenate([trace[key] for key in ('base','tcp','goal')])
    lower,upper=points.min(0),points.max(0)
    camera.lookat[:]=(lower+upper)/2
    camera.distance=max(2.4,1.6*float(np.linalg.norm(upper-lower))+1.2)
    camera.elevation=-22;camera.azimuth=135
    options=mujoco.MjvOption();options.geomgroup[3]=0
    font=ImageFont.load_default(size=18)
    renderer=mujoco.Renderer(model,height=720,width=1280,max_geom=10000)
    evidence['gl_context_created']=True
    def sphere(position,color,size):
        geom=renderer.scene.geoms[renderer.scene.ngeom]
        mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_SPHERE,np.full(3,size),position,np.eye(3).ravel(),color)
        renderer.scene.ngeom+=1
    def line(a,b,color,width=.003):
        if np.linalg.norm(b-a)<1e-10:
            return
        geom=renderer.scene.geoms[renderer.scene.ngeom]
        mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),color)
        mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,width,a,b)
        renderer.scene.ngeom+=1
    def frame_at(index):
        data.qpos[:3]=trace['base'][index];data.qpos[3:7]=base_quat[index];data.qpos[addresses]=q[index]
        mujoco.mj_forward(model,data)
        renderer.update_scene(data,camera=camera,scene_option=options)
        for key,color in [('goal',[0.,.85,1.,1.]),('tcp',[1.,.4,.05,1.])]:
            position=trace[key][index]
            sphere(position,color,.023 if key=='goal' else .014)
            for j in range(max(1,index-99),index+1):
                line(trace[key][j-1],trace[key][j],color)
            rotation=Rotation.from_quat(trace[key+'_quat_w'][index,[1,2,3,0]]).as_matrix()
            for axis,rgb in enumerate([[1.,.15,.15,1.],[.15,1.,.15,1.],[.3,.3,1.,1.]]):
                line(position,position+(.11 if key=='goal' else .075)*rotation[:,axis],rgb,.004)
        frame=Image.fromarray(renderer.render());draw=ImageDraw.Draw(frame)
        draw.rectangle((0,0,1280,201),fill=(20,28,35))
        command=trace['velocity_commands_yaw'][index];actual=trace['base_velocity_yaw'][index]
        fall=bool(trace['fall_height'][index] or trace['fall_tilt'][index])
        terminal='FALL' if metrics['fallen'] else ('completed' if metrics['completed'] else 'incomplete')
        lines=[f"PawWeaver B | recorded {metrics['engine']} | {evidence['case_id']} | saved-state MuJoCo geometry replay",
            'EE frame: base XY / yaw + fixed ground Z (not following base height / roll / pitch)',
            f"Preset command: vx={command[0]:+.3f}m/s | vy={command[1]:+.3f}m/s | yawdot={command[2]:+.3f}rad/s",
            f"Actual yaw-frame: vx={actual[0]:+.3f}m/s | vy={actual[1]:+.3f}m/s | yawrate={trace['base_yaw_rate'][index]:+.3f}rad/s",
            f"Elapsed {trace['times'][index]:.2f}s / recorded {trace['times'][-1]:.2f}s / requested {metrics['requested_steps']*DT:.2f}s | current fall={fall} | recorded end: {terminal}",
            f"EE error: {trace['errors'][index]:.3f}m / {trace['orientation_errors_rad'][index]:.3f}rad | base upZ={trace['base_up_z'][index]:.3f}",
            'Cyan=target; orange=TCP; RGB=pose axes | provisional model | trained=false',
            'Saved-state replay; no new physics / policy execution / visual control. All saved frames included.']
        for j,text in enumerate(lines):
            draw.text((12,5+j*24),text,font=font,fill=(240,240,240))
        return np.asarray(frame)
    try:
        with imageio.get_writer(output,fps=50,codec='libx264') as writer:
            for index in range(len(q)):
                writer.append_data(frame_at(index))
        evidence.update(rendered=True,video=str(output),video_sha256=sha(output),fps=50)
        output.with_suffix('.json').write_text(json.dumps(evidence,indent=2,allow_nan=False)+'\n')
    finally:
        renderer.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('evaluation','bundle','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--case',required=True)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    states=load_states(args.evaluation,args.bundle,args.case)
    if args.check_only:
        output=args.output.with_suffix('.check.json')
        if output.exists():
            raise FileExistsError(output)
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(states[-1],indent=2,allow_nan=False)+'\n')
        print(json.dumps({'check_only':True,'frames':len(states[2]),'checks':states[-1]['alignment_checks'],'output':str(output)}))
    else:
        render(states,args.output)
        print(f'Rendered {args.output}')


if __name__=='__main__':
    main()
