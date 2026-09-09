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
    queue.enqueue(GoalSample(0.,(1,2,3)),.1)
    assert queue.update(.09) is None
    assert queue.update(.1).position==(1,2,3)
    queue.enqueue(GoalSample(.2,(4,5,6)),.25)
    queue.update(.25)
    queue.enqueue(GoalSample(.1,(9,9,9)),.3)
    assert queue.update(.3).position==(4,5,6)
    assert queue.hold_required(.8)
