import numpy as np
import pytest
from pawweaver.trajectories import synthetic, AdaptiveSampler, split_for_source, Trajectory

def test_fixed_frame_and_sampler():
    trajectory = synthetic("eight",7)
    aligned = trajectory.align([3,4,1],np.pi/2)
    assert np.allclose(aligned.positions[0],[3,4,1])
    assert np.linalg.norm(np.diff(trajectory.positions,axis=0)/.02,axis=-1).max()<=.3
    sampler = AdaptiveSampler()
    for _ in range(50):
        sampler.update("eight",10.)
    assert sampler.probabilities[3]>sampler.probabilities[0]
    assert sampler.probabilities.min()>=.05
    assert np.isclose(sampler.probabilities.sum(),1)
    assert split_for_source("episode-7")==split_for_source("episode-7")
    with pytest.raises(ValueError):
        Trajectory([0,0],[[0,0,0],[1,0,0]],{})

def test_sensor_tcp_offset_rotates_and_time_scale(tmp_path):
    h5py = pytest.importorskip("h5py")
    pytest.importorskip("scipy")
    from pawweaver.data import convert
    path = tmp_path/"episode.h5"
    q = np.zeros((101,7)); q[:,0]=np.linspace(0,1,101)
    q[:,5:7]=np.sqrt(.5)
    with h5py.File(path,"w") as file:
        file.create_dataset("observations/qpos",data=q)
    extrinsic = np.eye(4); extrinsic[0,3]=.1
    config = {"pose_kind":"sensor","sensor_to_tcp":extrinsic.tolist(),"source_to_task":np.eye(4).tolist(),
        "timestamp_scale_s":1.,"position_scale_m":1.,"max_gap_s":.02,
        "max_speed_m_s":.2,"max_acceleration_m_s2":1.,"task_bounds_m":[[-2,-2,-2],[2,2,2]]}
    result = convert(path,np.linspace(0,1,101),config,"test-session")
    assert np.allclose(result.positions[0],[0,.1,0])
    assert result.metadata["time_scale"]>=5-1e-6
    assert np.max(np.linalg.norm(np.diff(result.positions,axis=0)/.02,axis=-1))<=.20001
