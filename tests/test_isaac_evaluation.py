"""CPU checks for the real evaluator's batching arithmetic, without mocking Isaac."""
import importlib.util
from pathlib import Path
import numpy as np
from pawweaver.trajectories import Trajectory

SPEC=importlib.util.spec_from_file_location("evaluate_isaac",Path(__file__).resolve().parents[1]/"scripts/evaluate_isaac.py")
evaluator=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluator)


def test_uneven_batch_targets_keep_case_clocks_and_world_orientation():
    cases=[]
    for index in range(3):
        q=np.array([[1.,0.,0.,0.],[np.sqrt(.5),0.,0.,np.sqrt(.5)]])
        cases.append(Trajectory(np.array([7.+index,7.04+index]),np.array([[index,0.,1.],[index+.4,0.,1.]]),q,{"case_id":str(index)}))
    origins=np.array([[20.,30.,0.],[-10.,-20.,0.]])
    batches=[cases[:2],cases[2:]]
    for batch in batches:
        positions,orientations=evaluator.batch_reference(batch,origins,4)
        assert positions.shape==(len(batch),4,3)  # Last partial batch has no padding case.
        assert orientations.shape==(len(batch),4,4)
        for slot,case in enumerate(batch):
            np.testing.assert_allclose(positions[slot]-origins[slot],case.sample(case.timestamps[0]+np.arange(4)*.02))
            np.testing.assert_allclose(orientations[slot],case.sample_orientation(case.timestamps[0]+np.arange(4)*.02))
            np.testing.assert_allclose(positions[slot,0],case.positions[0]+origins[slot])


def test_early_fall_and_short_case_stop_archiving_while_batch_continues():
    rows=[{"times":[],"goal":[]} for _ in range(3)]
    lengths=[5,2,4]
    fallen=np.zeros(3,dtype=bool)
    ended=[]
    for step in range(1,5):
        # Fourth entry is a padding environment and must never be archived.
        values={"goal":np.array([[step,0.,0.],[step,1.,0.],[step,2.,0.],[999.,999.,999.]])}
        ended.append(evaluator.append_batch_step(rows,lengths,fallen,step,values,[step==1,step==3,False,True]))
        values["goal"][:]=-123.  # Records must own their data, not alias simulator buffers.
    assert ended==[False,False,False,True]
    assert fallen.tolist()==[True,False,False]
    assert [r["times"] for r in rows]==[[.02],[.02,.04],[.02,.04,.06,.08]]
    assert [len(r["goal"]) for r in rows]==[1,2,4]
    np.testing.assert_array_equal(rows[0]["goal"][0],[1.,0.,0.])
    np.testing.assert_array_equal(rows[2]["goal"][-1],[4.,2.,0.])
