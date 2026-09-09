import numpy as np
from pawweaver.vision import Intrinsics,register_depth,DelayedMeasurements
from pawweaver.contracts import GoalSample

def test_depth_registration_extrinsic_and_delay():
    intr=Intrinsics(10,8,10,10,5,4,.1,5.)
    depth=np.ones((8,10))
    extrinsic=np.eye(4); extrinsic[0,3]=.1
    aligned=register_depth(depth,intr,intr,extrinsic)
    assert np.isnan(aligned[:,0]).all()
    assert np.allclose(aligned[:,1:],1.)
    queue=DelayedMeasurements(.5)
    queue.enqueue(GoalSample(0.,(1,2,3),(1,0,0,0)),.1)
    assert queue.update(.09) is None
    assert queue.update(.1).position==(1,2,3)
    queue.enqueue(GoalSample(.2,(4,5,6),(0,1,0,0)),.25)
    queue.update(.25)
    queue.enqueue(GoalSample(.1,(9,9,9),(0,0,1,0)),.3)
    assert queue.update(.3).position==(4,5,6)
    assert queue.latest.orientation_wxyz==(0,1,0,0)
    queue.enqueue(GoalSample(.4,(7,8,9),(0,0,0,1),False,0.),.45)
    assert queue.update(.45).timestamp==.2
    assert queue.latest.orientation_wxyz==(0,1,0,0)
    assert queue.hold_required(.8)

def test_marker_pose_composes_camera_and_marker_calibration():
    import cv2
    import pytest
    from pawweaver.vision import MarkerEstimator, quaternion_rotation
    intr=Intrinsics(640,480,400,400,320,240,.1,5.)
    calibration_q=np.array([np.cos(.3),0,np.sin(.3),0])
    estimator=MarkerEstimator(intr,intr,np.eye(4),7,.16,[.02,.01,0],calibration_q)
    rvec=np.array([2.9,.2,.1]); rotation=cv2.Rodrigues(rvec)[0]
    corners=np.array([[-.08,.08,0],[.08,.08,0],[.08,-.08,0],[-.08,-.08,0]])
    projected=cv2.projectPoints(corners,rvec,np.array([0.,0.,1.]),intr.matrix,np.zeros(5))[0].reshape(1,4,2)
    class Detector:
        def detectMarkers(self,gray):
            return [projected],np.array([[7]]),[]
    estimator.detector=Detector()
    world=np.eye(4); world[:3,:3]=quaternion_rotation([np.cos(.2),np.sin(.2),0,0]); world[:3,3]=[1,2,3]
    sample=estimator.measure(np.zeros((480,640,3),np.uint8),np.ones((480,640)),.25,world)
    assert sample is not None and sample.timestamp==.25
    assert np.allclose(quaternion_rotation(sample.orientation_wxyz),world[:3,:3]@rotation@quaternion_rotation(calibration_q),atol=1e-7)
    with pytest.raises(TypeError):
        MarkerEstimator(intr,intr,np.eye(4),7,.16,[0,0,0])
    with pytest.raises(ValueError):
        MarkerEstimator(intr,intr,np.eye(4),7,.16,[0,0,0],[0,0,0,0])

def test_rendered_marker_follows_entire_target_pose(monkeypatch):
    from types import SimpleNamespace
    import pawweaver.visual_runtime as runtime
    from pawweaver.vision import quaternion_rotation
    # Exercise the physical callback without allocating a renderer or simulator.
    target_q=np.array([np.cos(.4),0,0,np.sin(.4)])
    trajectory=SimpleNamespace(sample=lambda t:np.array([1.,2.,3.]),sample_orientation=lambda t:target_q)
    vision=runtime.RenderedVision.__new__(runtime.RenderedVision)
    vision.trajectory=trajectory; vision.hold=True; vision.reference_time=.7
    vision.marker_id=0; vision.offset=np.array([.1,.2,.3])
    vision.marker_to_goal_rotation=quaternion_rotation([np.cos(.3),np.sin(.3),0,0])
    vision.next_capture=2.
    runner=SimpleNamespace(data=SimpleNamespace(time=1.,mocap_pos=np.zeros((1,3)),mocap_quat=np.zeros((1,4))),model=None)
    monkeypatch.setattr(runtime.mujoco,"mj_forward",lambda *args:None)
    vision.tick(runner)
    marker_rotation=quaternion_rotation(runner.data.mocap_quat[0])
    assert np.allclose(marker_rotation@vision.marker_to_goal_rotation,quaternion_rotation(target_q))
    assert np.allclose(runner.data.mocap_pos[0]+marker_rotation@vision.offset,[1,2,3])
    assert vision.reference_time==.7
    # A new orientation moves the marker consistently with its nonzero goal offset.
    target_q[:]=[1,0,0,0]
    vision.tick(runner)
    marker_rotation=quaternion_rotation(runner.data.mocap_quat[0])
    assert np.allclose(marker_rotation@vision.marker_to_goal_rotation,np.eye(3))
    assert np.allclose(runner.data.mocap_pos[0]+marker_rotation@vision.offset,[1,2,3])

def test_rendered_reference_clock_starts_at_trajectory_timestamp(monkeypatch):
    from types import SimpleNamespace
    import pawweaver.visual_runtime as runtime
    from pawweaver.trajectories import Trajectory
    trajectory=Trajectory([5.,6.],[[0,0,0],[1,0,0]],[[1,0,0,0],[0,0,0,1]],{})
    intr=dict(width=10,height=8,fx=10,fy=10,cx=5,cy=4,min_depth=.1,max_depth=5.)
    calibration=dict(color=intr,depth=intr,depth_to_color=np.eye(4))
    scenario=dict(marker_id=7,marker_size_m=.16,marker_to_goal=[0,0,0],marker_to_goal_quat_wxyz=[1,0,0,0])
    runner=SimpleNamespace(model=SimpleNamespace(body=lambda name:SimpleNamespace(mocapid=[0])),
        data=SimpleNamespace(time=0.,mocap_pos=np.zeros((1,3)),mocap_quat=np.zeros((1,4))),
        spec=SimpleNamespace(physics_dt=.02))
    monkeypatch.setattr(runtime.mujoco,"Renderer",lambda *args:SimpleNamespace(enable_depth_rendering=lambda:None))
    monkeypatch.setattr(runtime.mujoco,"mj_forward",lambda *args:None)
    vision=runtime.RenderedVision(runner,calibration,scenario,trajectory,None)
    vision.next_capture=1.  # This test exercises target scheduling without image capture.
    vision.tick(runner)
    assert vision.reference_time==5.
    assert np.allclose(runner.data.mocap_pos[0],[0,0,0])
    vision.hold=False
    vision.tick(runner)
    assert np.isclose(vision.reference_time,5.02)
    assert np.allclose(runner.data.mocap_pos[0],[.02,0,0])
    assert np.allclose(runner.data.mocap_quat[0],trajectory.sample_orientation(5.02))
