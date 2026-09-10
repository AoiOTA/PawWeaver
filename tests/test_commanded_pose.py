from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import torch

from pawweaver.commanded_pose import (
    COMMAND_SCHEMA_VERSION, COMMAND_TASK_KIND, COMMAND_TASK_FRAME, COMMAND_UNITS,
    CommandedPoseTrajectory, CommandedPoseBank, load_command_suite,
    task_pose_to_world, yaw_angle, yaw_quaternion, yaw_linear_velocity, yaw_rate,
)
from pawweaver.math import rpy_quat
from pawweaver.trajectories import Trajectory


def metadata(case="case0",split="train",group="near"):
    return dict(task_kind=COMMAND_TASK_KIND,task_frame=COMMAND_TASK_FRAME,units=dict(COMMAND_UNITS),
                case_id=case,source_id="source/"+case,split=split,training_group=group,
                command_source="preset simulation trajectory")


def trajectory(case="case0",split="train",group="near",offset=0.,duration=.1):
    return CommandedPoseTrajectory([0.,duration],[[2.+offset,3.,4.],[3.+offset,4.,5.]],
        [[1.,0.,0.,0.],[-np.sqrt(.5),0.,0.,-np.sqrt(.5)]],
        [[.1+offset,.2,-.3],[.3+offset,-.2,.5]],metadata(case,split,group))


def save_suite(tmp_path,trajectories):
    cases=[]
    for item in trajectories:
        path=tmp_path/(item.metadata["case_id"]+".npz")
        item.save(path)
        cases.append({"case_id":item.metadata["case_id"],"path":path.name,
                      "sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest=dict(schema_version=COMMAND_SCHEMA_VERSION,task_kind=COMMAND_TASK_KIND,
                  task_frame=COMMAND_TASK_FRAME,units=dict(COMMAND_UNITS),cases=cases)
    (tmp_path/"manifest.json").write_text(json.dumps(manifest))
    return manifest


@pytest.mark.parametrize("dtype",[torch.float32,torch.float64])
def test_task_pose_uses_xy_yaw_ground_origin_for_all_history(dtype):
    rpys=np.array([[.6,-.4,np.pi/2],[-.3,.7,-np.pi/3]])
    base=torch.tensor([[10.,20.,100.],[30.,-8.,-22.]],dtype=dtype)
    base_q=torch.tensor(np.array([rpy_quat(rpy) for rpy in rpys]),dtype=dtype)
    ground=torch.tensor([4.,9.],dtype=dtype)
    positions=torch.tensor([[[1.,0.,.2],[0.,1.,.7],[1.,2.,3.],[3.,1.,2.]],
                            [[2.,1.,.5],[1.,2.,1.],[1.,3.,2.],[4.,2.,1.]]],dtype=dtype)
    local_q=np.array([rpy_quat([.1,.2,.3])]*8).reshape(2,4,4)
    orientations=torch.tensor(local_q,dtype=dtype)
    p,q=task_pose_to_world(positions,orientations,base,base_q,ground)
    expected_p=[];expected_q=[]
    for index in range(2):
        yaw=Rotation.from_euler("z",rpys[index,2])
        origin=np.array([base[index,0].item(),base[index,1].item(),ground[index].item()])
        expected_p.append(origin+yaw.apply(positions[index].numpy()))
        expected_q.append((yaw*Rotation.from_quat(local_q[index][:,[1,2,3,0]])).as_quat()[:,[3,0,1,2]])
    np.testing.assert_allclose(p.numpy(),expected_p,atol=3e-6,rtol=1e-6)
    np.testing.assert_allclose(q.numpy(),expected_q,atol=3e-7,rtol=1e-6)
    assert p.dtype==q.dtype==dtype and p.device==q.device==base.device
    # Change base Z and roll/pitch alone: every history target stays unchanged.
    changed=base.clone();changed[:,2]+=77.
    new_q=torch.tensor(np.array([rpy_quat([-.2,.5,rpy[2]]) for rpy in rpys]),dtype=dtype)
    p2,q2=task_pose_to_world(positions,orientations,changed,new_q,ground)
    torch.testing.assert_close(p,p2,rtol=1e-6,atol=3e-6)
    torch.testing.assert_close(q,q2,rtol=1e-6,atol=3e-7)
    # The batched one-target route matches the same member of the history route.
    single_p,single_q=task_pose_to_world(positions[:,2],orientations[:,2],base,base_q,ground)
    torch.testing.assert_close(single_p,p[:,2]);torch.testing.assert_close(single_q,q[:,2])
    moved=base.clone();moved[:,:2]+=torch.tensor([2.,-3.],dtype=dtype)
    moved_p,_=task_pose_to_world(positions,orientations,moved,base_q,ground)
    torch.testing.assert_close(moved_p-p,torch.tensor([2.,-3.,0.],dtype=dtype).expand_as(p),atol=3e-6,rtol=1e-6)


def test_yaw_coordinates_use_full_body_velocity_then_remove_only_yaw():
    rpys=np.array([[.4,-.5,1.2],[0.,0.,-2.]])
    q=torch.tensor(np.array([rpy_quat(rpy) for rpy in rpys]),dtype=torch.float64)
    velocity=torch.tensor([[1.,2.,3.],[-2.,1.,.5]],dtype=torch.float64)
    expected=[]
    for rpy,v in zip(rpys,velocity.numpy()):
        world=Rotation.from_euler("xyz",rpy).apply(v)
        expected.append(Rotation.from_euler("z",rpy[2]).inv().apply(world))
    result=yaw_linear_velocity(velocity,q)
    np.testing.assert_allclose(result.numpy(),expected,atol=1e-12)
    assert not torch.allclose(result[0],velocity[0])
    torch.testing.assert_close(result[1],velocity[1])
    torch.testing.assert_close(yaw_angle(q),torch.tensor(rpys[:,2]))
    np.testing.assert_allclose(yaw_quaternion(q).numpy(),[rpy_quat([0.,0.,v]) for v in rpys[:,2]],atol=1e-12)


def test_yaw_rate_wrap_and_quaternion_sign_not_body_z_rate():
    before=torch.tensor(np.array([rpy_quat([.4,.5,np.deg2rad(v)]) for v in (179.,-179.)]))
    after=torch.tensor(np.array([rpy_quat([-.3,.2,np.deg2rad(v)]) for v in (-179.,179.)]))
    rate=yaw_rate(before,-after,torch.tensor([.1,.2],dtype=torch.float64))
    np.testing.assert_allclose(rate.numpy(),np.deg2rad([2.,-2.])/[.1,.2],atol=1e-12)
    torch.testing.assert_close(yaw_rate(before,-before,.02),torch.zeros(2,dtype=torch.float64))
    for bad in (0.,-.1,float("nan"),float("inf"),torch.ones(3),torch.ones(2,1)):
        with pytest.raises(ValueError,match="dt"):
            yaw_rate(before,after,bad)


def test_commanded_roundtrip_interpolation_endpoints_and_no_world_alias(tmp_path):
    original=trajectory(duration=2.)
    original.metadata["extra_source_context"]={"unaltered":True}
    path=tmp_path/"commanded.npz";original.save(path)
    loaded=CommandedPoseTrajectory.load(path)
    assert loaded.metadata==original.metadata
    np.testing.assert_allclose(loaded.sample([-1.,0.,1.,2.,3.]),[[2,3,4],[2,3,4],[2.5,3.5,4.5],[3,4,5],[3,4,5]])
    np.testing.assert_allclose(loaded.sample_command(1.),[.2,0.,.1],atol=1e-15)
    np.testing.assert_allclose(loaded.sample_command([-1.,3.]),[[.1,.2,-.3],[.3,-.2,.5]])
    np.testing.assert_allclose(abs(np.dot(loaded.sample_orientation(1.),rpy_quat([0,0,np.pi/4]))),1.,atol=1e-12)
    assert loaded.sample(np.zeros((2,3))).shape==(2,3,3)
    assert loaded.sample_orientation(np.zeros((2,3))).shape==(2,3,4)
    assert not hasattr(loaded,"align") and not isinstance(loaded,Trajectory)
    with pytest.raises(ValueError):
        Trajectory.load(path)
    old=tmp_path/"world.npz"
    Trajectory([0,1],[[1,2,3],[2,3,4]],[[1,0,0,0]]*2,{"task_kind":"world_tcp_pose","split":"train"}).save(old)
    with pytest.raises(ValueError,match="independent schema"):
        CommandedPoseTrajectory.load(old)
    for method in (loaded.sample,loaded.sample_orientation,loaded.sample_command):
        with pytest.raises(ValueError,match="sample times"):
            method([0.,np.nan])


@pytest.mark.parametrize("change",[
    {"task_kind":"world_tcp_pose"},{"task_frame":"world"},{"units":{}},
    {"command_source":""},{"source_id":None},{"case_id":""},{"split":"implicit"},{"schema_version":2},
])
def test_commanded_rejects_ambiguous_frame_units_or_identity(change):
    values=metadata();values.update(change)
    with pytest.raises(ValueError):
        CommandedPoseTrajectory([0,1],[[0,0,0]]*2,[[1,0,0,0]]*2,[[0,0,0]]*2,values)


@pytest.mark.parametrize("field,value",[
    ("timestamps",[1,2]),("timestamps",[0,0]),("timestamps",[0,float("inf")]),
    ("positions_task",[[0,0,0],[0,np.nan,0]]),
    ("orientations_task_wxyz",[[1,0,0,0],[0,0,0,0]]),
    ("velocity_commands_yaw",[[0,0,0],[0,0,float("nan")]]),
    ("velocity_commands_yaw",[[0,0],[0,0]]),
])
def test_commanded_rejects_invalid_arrays(field,value):
    args=dict(timestamps=[0,1],positions_task=[[0,0,0]]*2,orientations_task_wxyz=[[1,0,0,0]]*2,
              velocity_commands_yaw=[[0,0,0]]*2,metadata=metadata())
    args[field]=value
    with pytest.raises(ValueError):
        CommandedPoseTrajectory(**args)


def test_load_suite_preserves_split_source_and_rejects_relabeling(tmp_path):
    item=trajectory(split="test")
    manifest=save_suite(tmp_path,[item])
    result=load_command_suite(tmp_path)
    assert len(result)==1 and result[0].metadata==item.metadata
    assert result[0].metadata["split"]=="test"
    for mutation in ({"split":"train"},{"schema_version":2},{"task_kind":"world_tcp_pose"},{"units":{}}):
        changed=deepcopy(manifest);changed.update(mutation)
        (tmp_path/"manifest.json").write_text(json.dumps(changed))
        with pytest.raises(ValueError):
            load_command_suite(tmp_path)
    changed=deepcopy(manifest);changed["cases"][0]["source_id"]="other"
    (tmp_path/"manifest.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError,match="source_id"):
        load_command_suite(tmp_path)
    changed=deepcopy(manifest);changed["cases"]*=2
    (tmp_path/"manifest.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError,match="duplicate"):
        load_command_suite(tmp_path)
    changed=deepcopy(manifest);changed["cases"][0]["sha256"]="wrong"
    (tmp_path/"manifest.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError,match="Changed"):
        load_command_suite(tmp_path)
    with pytest.raises(ValueError,match="train demonstrations"):
        CommandedPoseBank(1,10,"cpu",demonstrations=[tmp_path/"case0.npz"])


def test_uniform_bank_consumes_pose_command_and_padding_without_alignment(tmp_path):
    paths=[]
    for index in range(3):
        path=tmp_path/f"{index}.npz"
        trajectory(str(index),offset=index,duration=.06).save(path);paths.append(path)
    bank=CommandedPoseBank(30,5,"cpu",seed=17,demonstrations=paths)
    bank.sampler.sample=lambda *args:pytest.fail("Fixed sampling must not use adaptive family draws")
    bank.reset(torch.arange(30))
    selected=np.random.default_rng(17).integers(3,size=30)
    np.testing.assert_array_equal(bank.demonstration_index,selected)
    np.testing.assert_array_equal(bank.family,np.zeros(30,dtype=int))
    assert bank.sampler.families==("velocity_pose",)
    bank.sampler.update("velocity_pose",.4)
    bank.sampler.load_state_dict(bank.sampler.state_dict())
    assert bank.sampler.counts.tolist()==[1]
    for index,file_index in enumerate(selected):
        item=CommandedPoseTrajectory.load(paths[file_index])
        np.testing.assert_allclose(bank.positions[index],item.sample(np.arange(10)*.02),atol=2e-7)
        np.testing.assert_allclose(bank.velocity_commands_yaw[index],item.sample_command(np.arange(10)*.02),atol=2e-7)
        torch.testing.assert_close(bank.positions[index,3:],bank.positions[index,3].expand(7,3))
    indices=torch.arange(30)%6
    torch.testing.assert_close(bank.current(indices),bank.positions[torch.arange(30),indices])
    torch.testing.assert_close(bank.current_orientation(indices),bank.orientations_wxyz[torch.arange(30),indices])
    torch.testing.assert_close(bank.current_command(indices),bank.velocity_commands_yaw[torch.arange(30),indices])
    future,valid=bank.future(indices)
    future_steps=indices[:,None]+torch.arange(1,5)
    torch.testing.assert_close(future,bank.positions[torch.arange(30)[:,None],future_steps])
    assert torch.equal(valid,(future_steps<=5).unsqueeze(-1).expand(-1,-1,3))
    old=bank.positions.clone();old_commands=bank.velocity_commands_yaw.clone();old_selected=bank.demonstration_index.copy()
    bank.reset(torch.tensor([3]))
    untouched=torch.arange(30)!=3
    assert torch.equal(bank.positions[untouched],old[untouched])
    assert torch.equal(bank.velocity_commands_yaw[untouched],old_commands[untouched])
    np.testing.assert_array_equal(bank.demonstration_index[untouched],old_selected[untouched])


def test_group_bank_reproduces_two_stage_draws_and_keeps_sixty_second_endpoint(tmp_path):
    paths=[]
    for index,group in enumerate(["near","near","far","excluded"]):
        path=tmp_path/f"{index}.npz";trajectory(str(index),group=group,offset=index,duration=60.).save(path);paths.append(path)
    bank=CommandedPoseBank(12,3000,"cpu",seed=42,demonstrations=paths,
        demonstration_group_weights={"near":1.,"far":3.,"absent_zero":0.})
    bank.reset(torch.arange(12))
    rng=np.random.default_rng(42);expected=[]
    for _ in range(12):
        members=[[0,1],[2]][rng.choice(2,p=[.25,.75])]
        expected.append(members[rng.integers(len(members))])
    np.testing.assert_array_equal(bank.demonstration_index,expected)
    assert 3 not in expected
    assert bank.positions.shape==(12,3005,3)
    assert bank.orientations_wxyz.shape==(12,3005,4)
    assert bank.velocity_commands_yaw.shape==(12,3005,3)
    torch.testing.assert_close(bank.velocity_commands_yaw[:,3000:],bank.velocity_commands_yaw[:,3000,None,:].expand(-1,5,-1))
    future,valid=bank.future(torch.full((12,),2999))
    assert valid[:,0].all() and not valid[:,1:].any()
    torch.testing.assert_close(future,bank.positions[:,3000,None,:].expand(-1,4,-1))


@pytest.mark.parametrize("weights",[{},[],{"near":-1},{"near":float("nan")},{"near":float("inf")},
                                     {"near":0},{"missing":1},{"":1},{"near":True},{"near":"1"}])
def test_bank_rejects_invalid_fixed_group_weights(tmp_path,weights):
    path=tmp_path/"demo.npz";trajectory().save(path)
    with pytest.raises(ValueError,match="group"):
        CommandedPoseBank(1,10,"cpu",demonstrations=[path],demonstration_group_weights=weights)


def test_bank_rejects_empty_or_legacy_inputs(tmp_path):
    with pytest.raises(ValueError,match="nonempty"):
        CommandedPoseBank(1,10,"cpu")
    path=tmp_path/"world.npz"
    Trajectory([0,1],[[1,2,3],[2,3,4]],[[1,0,0,0]]*2,{"split":"train"}).save(path)
    with pytest.raises(ValueError,match="independent schema"):
        CommandedPoseBank(1,10,"cpu",demonstrations=[path])
