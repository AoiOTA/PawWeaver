"""Prepare only the fixed 25% train-source pose course and validate CPU consumers."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import torch
from pawweaver.commanded_pose import CommandedPoseTrajectory, CommandedPoseBank, load_command_suite

O=Path(__file__).resolve().parent
ROOT=O.parents[3]
E=O.parent/'plan_v3_commanded_pose'
N=O.parent/'plan_v3_neutral_learning'
A=.25

def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    assert not (O/'config.json').exists(), 'Preserve existing prepared inputs'
    source_config=json.loads((E/'config.json').read_text())
    neutral_config=json.loads((N/'config.json').read_text())
    sources=[ROOT/p for p in source_config['demonstrations']]
    assert len(sources)==len(set(sources))==28
    original=load_command_suite(E/'train')
    assert [t.metadata['case_id'] for t in original]==[p.stem for p in sources]
    start=original[0].positions_task[0]
    start_q=original[0].orientations_task_wxyz[0]
    start_r=Rotation.from_quat(start_q[[1,2,3,0]])
    entries=[];rows=[]
    for path,old in zip(sources,original):
        np.testing.assert_allclose(old.positions_task[0],start,rtol=0,atol=1e-14)
        assert (start_r.inv()*Rotation.from_quat(old.orientations_task_wxyz[0,[1,2,3,0]])).magnitude()<1e-14
        with np.load(path,allow_pickle=False) as raw:
            raw_q=raw['orientations_task_wxyz'].copy()
        rotations=Rotation.from_quat(old.orientations_task_wxyz[:,[1,2,3,0]])
        transformed=start_r*Rotation.from_rotvec((start_r.inv()*rotations).as_rotvec()*A)
        positions=start+A*(old.positions_task-start)
        quats=transformed.as_quat()[:,[3,0,1,2]]
        neutral=path.stem.startswith('neutral_')
        if neutral:
            positions=old.positions_task.copy();quats=raw_q
        metadata=deepcopy(old.metadata)
        metadata.update(amplitude=A,original_source_path=str(path.relative_to(ROOT)),original_source_sha256=sha(path),
            amplitude_reference='Original common reset pose; shortest relative rotation scaled per frame',
            amplitude_evidence_limit='Train-source curriculum only; no reachability, task success or hardware validation')
        new=CommandedPoseTrajectory(old.timestamps,positions,quats,old.velocity_commands_yaw,metadata)
        # Preserve neutral serialized orientations exactly; they are already unit quaternions.
        if neutral: new.orientations_task_wxyz=raw_q
        target=O/'train'/path.name
        new.save(target)
        loaded=CommandedPoseTrajectory.load(target)
        with np.load(target,allow_pickle=False) as raw:
            np.testing.assert_array_equal(raw['timestamps'],old.timestamps)
            np.testing.assert_array_equal(raw['velocity_commands_yaw'],old.velocity_commands_yaw)
            if neutral: np.testing.assert_array_equal(raw['orientations_task_wxyz'],raw_q)
        np.testing.assert_allclose(loaded.positions_task,start+A*(old.positions_task-start),rtol=0,atol=1e-14)
        delta=(Rotation.from_quat(loaded.orientations_task_wxyz[:,[1,2,3,0]]).inv()*transformed).magnitude()
        assert float(delta.max())<1e-14
        # Independent per-frame SLERP cross-check, including moving and hold samples.
        for index in (0,101,400,750,1000,3000):
            expected=Slerp([0.,1.],Rotation.concatenate([start_r,rotations[index]]))([A])
            assert (expected.inv()*Rotation.from_quat(loaded.orientations_task_wxyz[index,[1,2,3,0]])).magnitude()[0]<1e-14
        assert np.isfinite(loaded.positions_task).all() and np.isfinite(loaded.orientations_task_wxyz).all()
        np.testing.assert_allclose(np.linalg.norm(loaded.orientations_task_wxyz,axis=1),1,rtol=0,atol=1e-14)
        if neutral: np.testing.assert_array_equal(loaded.positions_task,old.positions_task)
        for key,value in old.metadata.items(): assert loaded.metadata[key]==value
        entry=dict(case_id=path.stem,path=path.name,sha256=sha(target))
        entries.append(entry)
        rows.append(dict(case_id=path.stem,original_sha256=sha(path),position_amplitude_max_error_m=float(np.abs(loaded.positions_task-(start+A*(old.positions_task-start))).max()),rotation_amplitude_max_error_rad=float(delta.max()),neutral_arrays_unchanged=neutral))
    manifest=json.loads((E/'train/manifest.json').read_text())
    manifest.update(cases=entries,amplitude=A,scope='25% train-derived course; not held-out or formal acceptance')
    dump(O/'train/manifest.json',manifest)
    dev=json.loads((E/'dev8/manifest.json').read_text())
    by_id={entry['case_id']:entry for entry in entries}
    dev_ids=[entry['case_id'] for entry in entries if entry['case_id'].startswith('neutral_')]+['low_stand','high_stand','lateral_stand','lateral_arc']
    dev.update(cases=[dict(by_id[case_id],path='../train/'+by_id[case_id]['path']) for case_id in dev_ids],amplitude=A,scope=manifest['scope'])
    dump(O/'dev11/manifest.json',dev)
    config=deepcopy(neutral_config)
    config['demonstrations']=[str((O/'train'/entry['path']).relative_to(ROOT)) for entry in entries]
    assert [k for k in config if config[k]!=neutral_config[k]]==['demonstrations']
    dump(O/'config.json',config)
    assert len(load_command_suite(O/'train'))==28 and len(load_command_suite(O/'dev11'))==11
    bank=CommandedPoseBank(128,3000,'cpu',seed=0,demonstrations=config['demonstrations'],demonstration_group_weights=config['demonstration_group_weights'])
    times=np.arange(3005)*.02
    for index,trajectory in enumerate(bank.demonstrations):
        np.testing.assert_allclose(bank._positions[index],trajectory.sample(times),rtol=0,atol=6e-8)
        np.testing.assert_allclose(bank._orientations[index],trajectory.sample_orientation(times),rtol=0,atol=6e-8)
        np.testing.assert_allclose(bank._commands[index],trajectory.sample_command(times),rtol=0,atol=6e-8)
    bank.reset(torch.arange(128))
    for index,selected in enumerate(bank.demonstration_index):
        assert torch.equal(bank.positions[index],bank._positions[selected])
        assert torch.equal(bank.orientations_wxyz[index],bank._orientations[selected])
        assert torch.equal(bank.velocity_commands_yaw[index],bank._commands[selected])
    assert [len(g) for g in bank.demonstration_groups]==[4,24]
    np.testing.assert_allclose(bank.demonstration_group_probabilities,[.3,.7],rtol=0,atol=1e-15)
    result=dict(status='CPU input validation passed',simulation_run=False,trained=False,amplitude=A,train_cases=28,dev_cases=11,
        config_changed_keys=['demonstrations'],bank_all_28_sample_tables_match=True,bank_reset_samples=128,
        group_sizes=[len(g) for g in bank.demonstration_groups],group_probabilities=bank.demonstration_group_probabilities.tolist(),
        reset_position_task=start.tolist(),reset_quat_task_wxyz=start_q.tolist(),cases=rows,
        source_config_sha256=sha(N/'config.json'),original_full_config_sha256=sha(E/'config.json'))
    dump(O/'generation_summary.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases',)}))

if __name__=='__main__': main()
