"""Reconstruct E1 foot sphere-bottom heights from saved states; no simulation."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from pawweaver.assets.model import RobotTree, origin
from pawweaver.contracts import JOINT_NAMES, FOOT_NAMES
from pawweaver.math import rpy_matrix
from pawweaver.observations import ObservationSpec

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[3]
DT=.02


def read(path):
    return json.loads(path.read_text())


def geometry(tree):
    result=[]
    for name in FOOT_NAMES:
        collisions=tree.links[name].findall("collision")
        if len(collisions)!=1 or collisions[0].find("geometry/sphere") is None:
            raise ValueError(f"Expected one spherical collision for {name}")
        result.append((origin(collisions[0]),float(collisions[0].find("geometry/sphere").get("radius"))))
    return result


def default_check(tree,q0,spheres):
    # Independent planar-link result for this exact zero-hip default posture.
    if not np.allclose(q0[:12],np.tile([0.,.85,-1.65],4),atol=1e-12,rtol=0):
        raise ValueError("Default configuration differs from the analytic self-check")
    fk=tree.forward(dict(zip(JOINT_NAMES,q0)))
    centers=np.array([(fk[name]@col)[:3,3] for name,(col,_) in zip(FOOT_NAMES,spheres)])
    x_delta=-.212*np.sin(.85)-.21344*np.sin(-.8)
    expected=np.array([[x+x_delta,y,-.212*np.cos(.85)-.21344*np.cos(-.8)]
                       for x,y in ((.2119,-.1654),(.2119,.1654),(-.2119,-.1654),(-.2119,.1654))])
    error=float(np.abs(centers-expected).max())
    if error>1e-12:
        raise ValueError(f"Default planar analytic FK disagrees: {error}")
    # Projected gravity supplies world vertical independently of unknown yaw.
    rotation=rpy_matrix([.2,-.3,.7])
    gravity=rotation.T@np.array([0.,0.,-1.])
    projected=.4-centers@gravity-np.array([radius for _,radius in spheres])
    explicit=.4+(rotation@centers.T).T[:,2]-np.array([radius for _,radius in spheres])
    if not np.allclose(projected,explicit,atol=1e-12,rtol=0):
        raise ValueError("Gravity/world vertical convention self-check failed")
    return {"default_joint_positions_rad":q0.tolist(),"foot_sphere_centers_base_m":dict(zip(FOOT_NAMES,centers.tolist())),
            "analytic_fk_max_abs_error_m":error,"gravity_projection_max_abs_error_m":float(np.abs(projected-explicit).max()),
            "sphere_radii_m":dict(zip(FOOT_NAMES,[radius for _,radius in spheres])),
            "analytic_model":"Hip0, thigh0.85rad, calf-1.65rad; lengths0.212/0.21344m; root hip origins x±0.2119,y±0.06 and lateral hip offset±0.1054m."}


def longest(mask,times):
    if not mask.any():
        return None
    starts=np.flatnonzero(mask&np.r_[True,~mask[:-1]])
    ends=np.flatnonzero(mask&np.r_[~mask[1:],True])
    index=int(np.argmax(ends-starts))
    first,last=int(starts[index]),int(ends[index])
    return {"first_sample_s":float(times[first]),"last_sample_s":float(times[last]),
            "consecutive_samples":last-first+1,"sample_count_times_control_dt_s":(last-first+1)*DT,
            "observed_endpoint_span_s":float(times[last]-times[first]),
            "touches_window_boundary":bool(first==0 or last==len(times)-1)}


def summarize(mask,times,bottoms,forces,nonfoot,ground_pairs):
    if not mask.any():
        return None
    time=times[mask];height=bottoms[mask];force=forces[mask]
    feet={}
    for index,name in enumerate(FOOT_NAMES):
        high=height[:,index]>.03
        weak=force[:,index]<=1.
        feet[name]={"sphere_bottom_height_min_m":float(height[:,index].min()),
                    "sphere_bottom_height_max_m":float(height[:,index].max()),
                    "sphere_bottom_height_mean_m":float(height[:,index].mean()),
                    "above3cm_fraction":float(high.mean()),"longest_above3cm":longest(high,time),
                    "above3cm_and_net_force_le1N_fraction":float((high&weak).mean()),
                    "longest_above3cm_and_net_force_le1N":longest(high&weak,time),
                    "above3cm_and_net_force_gt1N_fraction":float((high&~weak).mean()),
                    "net_force_gt1N_fraction":float((~weak).mean())}
    result={"samples":len(time),"first_sample_s":float(time[0]),"last_sample_s":float(time[-1]),"feet":feet,
            "foot_net_force_gt1N_count_samples":{str(n):int(((force>1.).sum(axis=1)==n).sum()) for n in range(5)},
            "robot_nonfoot_net_force_gt5N_any_fraction":float(nonfoot[mask].mean()),
            "robot_nonfoot_net_force_gt5N_longest":longest(nonfoot[mask],time),
            "any_foot_above3cm_and_robot_nonfoot_net_gt5N_fraction":float(((height>.03).any(axis=1)&nonfoot[mask]).mean()),
            "nonfoot_ground_pairs":None}
    if ground_pairs is not None:
        result["nonfoot_ground_pairs"]={"any_fraction":float((ground_pairs[mask]>0).mean()),
            "maximum_count":int(ground_pairs[mask].max()),"longest_any":longest(ground_pairs[mask]>0,time)}
    return result


def case_result(path,episode,engine,tree,q0,spheres,physics_dt):
    with np.load(path,allow_pickle=False) as saved:
        data={key:saved[key] for key in saved.files}
    if any(not np.isfinite(value).all() for value in data.values() if value.dtype.kind in "fiu"):
        raise ValueError(f"Nonfinite trace: {path}")
    ticks=np.rint(data["times"]/DT).astype(int)
    if not np.array_equal(ticks,np.arange(1,len(ticks)+1)) or not np.allclose(data["times"],ticks*DT,atol=1e-8,rtol=0):
        raise ValueError(f"Expected contiguous 50Hz ticks: {path}")
    if len(ticks)!=episode["actual_steps"] or len(ticks)<2:
        raise ValueError(f"No sufficient matching report/trace steps: {path}")
    spec=ObservationSpec()
    if (spec.history_frames,spec.proprio_dim,spec.size)!=(5,42,276) or data["observations"].shape!=(len(ticks),276):
        raise ValueError("Expected actual 276-input observation contract")
    # Row i observation was sampled before row i's step. Row i+1 latest
    # proprioception therefore supplies the saved post-step state of row i.
    q=data["observations"][1:,168:186].astype(float)+q0
    gravity=data["observations"][1:,207:210].astype(float)
    if engine=="mujoco":
        # Existing implicitfast traces save kinematics/contact before the final
        # 2ms hinge position integration; undo that integration in q only.
        q-=physics_dt*data["velocities"][:-1]
    n=len(q)
    center_local=np.empty((n,4,3));tcp_local=np.empty((n,3))
    for index,joints in enumerate(q):
        fk=tree.forward(dict(zip(JOINT_NAMES,joints)))
        center_local[index]=[(fk[name]@col)[:3,3] for name,(col,_) in zip(FOOT_NAMES,spheres)]
        tcp_local[index]=fk["tcp"][:3,3]
    # R_world_from_base's third row is -gravity_base. No yaw/XY recovery.
    bottoms=data["base"][:-1,2,None]-np.einsum("ni,nfi->nf",gravity,center_local)-np.array([r for _,r in spheres])
    tcp_z=data["base"][:-1,2]-np.einsum("ni,ni->n",gravity,tcp_local)
    checks={"gravity_norm_max_error":float(np.abs(np.linalg.norm(gravity,axis=1)-1.).max()),
            "base_up_z_max_error":float(np.abs(-gravity[:,2]-data["base_up_z"][:-1]).max()),
            "fk_tcp_base_xyz_max_error_m":float(np.linalg.norm(tcp_local-data["observations"][1:,228:231],axis=1).max()),
            "fk_tcp_world_z_max_error_m":float(np.abs(tcp_z-data["tcp"][:-1,2]).max())}
    if any(value>1e-4 for value in checks.values()):
        raise ValueError(f"Trace/FK alignment failed {path}: {checks}")
    names=data["contact_body_names"].tolist()
    if len(names)!=len(set(names)) or any(name not in names for name in FOOT_NAMES):
        raise ValueError(f"Missing or duplicate foot contact names: {path}")
    force=data["contacts"][:-1]
    if force.shape!=(n,len(names)):
        raise ValueError("Contact columns differ from actual body names")
    foot_force=force[:,[names.index(name) for name in FOOT_NAMES]]
    nonfoot_ids=[index for index,name in enumerate(names) if name not in (*FOOT_NAMES,"world","")]
    nonfoot=(force[:,nonfoot_ids]>5.).any(axis=1)
    pairs=data["nonfoot_ground_contact_count"][:-1] if "nonfoot_ground_contact_count" in data else None
    times=data["times"][:-1]
    duration=episode["requested_steps"]*DT
    windows={"all_aligned":np.ones(n,dtype=bool),"post2":ticks[:-1]>=100,
             "designed_middle_hold":(ticks[:-1]>=round(duration*.4/DT))&(ticks[:-1]<=round(duration*.6/DT))}
    return {"case_id":episode["trajectory"]["case_id"],"trace":str(path),"original_episode_metrics":episode,
            "aligned_samples":n,"excluded_final_poststep_samples":1,
            "mujoco_hinge_kinematic_offset_s":-physics_dt if engine=="mujoco" else 0.,"alignment_checks":checks,
            "designed_middle_hold_planned_s":[duration*.4,duration*.6],
            "windows":{key:summarize(mask,times,bottoms,foot_force,nonfoot,pairs) for key,mask in windows.items()}}


def build():
    asset=REPO/"assets/generated/diagnostic"
    manifest=read(asset/"manifest.json")
    urdf=asset/"robot.urdf"
    if hashlib.sha256(urdf.read_bytes()).hexdigest()!=manifest["files"]["robot.urdf"]:
        raise ValueError("URDF differs from asset manifest")
    tree=RobotTree.load(urdf);spheres=geometry(tree)
    if tree.root_name!="base_link":
        raise ValueError("Expected canonical base_link root")
    results={};default=None
    for side in ("control","candidate"):
        bundle=read(ROOT/f"{side}100/bundle/manifest.json")
        actuators=bundle["actuators"]
        if actuators["joint_names"]!=list(JOINT_NAMES):
            raise ValueError("Bundle joint order differs from observation joint order")
        q0=np.array(actuators["default_pos"])
        current=default_check(tree,q0,spheres)
        if default is not None and current!=default:
            raise ValueError("Different default geometry in control/candidate")
        default=current
        for engine in ("physx","mujoco"):
            folder=ROOT/f"{side}_{engine}_workspace4"
            report=read(folder/"report.json")
            if report["asset_hash"]!=manifest["asset_hash"] or report["policy_sha256"]!=bundle["policy_sha256"]:
                raise ValueError("Report policy/asset identity differs")
            if report["case_ids"]!=["low","high","lateral","far"]:
                raise ValueError("Missing or reordered workspace cases")
            results[f"{side}_{engine}"]=[case_result(folder/e["trajectory"]["case_id"]/"trace.npz",e,engine,
                                                       tree,q0,spheres,actuators["physics_dt"]) for e in report["episodes"]]
    return {"default_fk_self_check":default,"evaluations":results,"foot_order":list(FOOT_NAMES),
            "height_definition":"World Z of the modeled foot collision sphere bottom relative to the flat z=0 ground. Projected gravity determines vertical only; world XY/yaw are not reconstructed.",
            "thresholds":"3cm sphere-bottom clearance and net force >1N / robot nonfoot net force >5N are engineering descriptions, not added acceptance standards.",
            "time_alignment":"Trace observation is pre-step; next row latest42 supplies prior post-step q/gravity. Drop final post-step row. MuJoCo q is backed up one physics_dt*qvel to match existing implicitfast saved body/contact kinematics.",
            "contact_scope":"50Hz net-force snapshots are not vertical loads or contact pairs. MuJoCo's separately saved nonfoot_ground_contact_count is actual floor contact-pair count at those snapshots; PhysX has no such pair record here. Excludes world from robot nonfoot forces. No statement about all physics substeps.",
            "duration_scope":"Consecutive sample count*0.02 and first-to-last span both reported; neither proves continuous inter-sample clearance/contact. Final failure metrics retained even though last post-state has no next observation.",
            "evidence_limit":"CPU saved-state geometry readout on provisional model; no new simulation, formal acceptance, or proof of gait success."}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"support_geometry.json")
    args=parser.parse_args()
    result=build()
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({key:{row["case_id"]:{name:row["windows"]["all_aligned"]["feet"][name]["above3cm_fraction"]
                            for name in FOOT_NAMES} for row in rows} for key,rows in result["evaluations"].items()},indent=2))
