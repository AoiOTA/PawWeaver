"""Real diagnostic path keeps formal gates, name mapping and checkpoint provenance."""
import copy
import json
from pathlib import Path
import pytest
from pawweaver.contracts import canonical_hash
from pawweaver.training_inputs import training_inputs, check_training_identity

ROOT=Path(__file__).resolve().parents[1]
ASSET=ROOT/'assets/generated/diagnostic'
SPEC=ROOT/'configs/diagnostic_actuators.json'


def test_explicit_diagnostic_and_formal_gate(tmp_path):
    with pytest.raises(ValueError,match='Diagnostic/source-only'):
        training_inputs(ASSET)
    for diagnostic,path in ((True,None),(False,SPEC)):
        with pytest.raises(ValueError,match='requires --provisional-spec'):
            training_inputs(ASSET,diagnostic=diagnostic,provisional_spec=path)
    manifest,spec,provisional=training_inputs(ASSET,diagnostic=True,provisional_spec=SPEC)
    assert not manifest['ready_for_training']
    assert spec.lower[-1]==-2.0943951 and spec.upper[-1]==2.0943951
    provisional['actuators']['upper'][-1]=3.14159265
    path=tmp_path/'spec.json'; path.write_text(json.dumps(provisional))
    with pytest.raises(ValueError,match='canonical URDF'):
        training_inputs(ASSET,diagnostic=True,provisional_spec=path)


def test_wrong_identity_and_checksums(tmp_path):
    original=json.loads((ASSET/'manifest.json').read_text())
    for name in original['files']:
        link=tmp_path/name; link.parent.mkdir(parents=True,exist_ok=True)
        link.symlink_to(ASSET/name)
    changed=copy.deepcopy(original); changed['robot']='synthetic_software_fixture'
    changed.pop('asset_hash'); changed['asset_hash']=canonical_hash(changed)
    (tmp_path/'manifest.json').write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='real AS2'):
        training_inputs(tmp_path,diagnostic=True,provisional_spec=SPEC)
    changed=copy.deepcopy(original); changed['files']['robot.urdf']='0'*64
    changed.pop('asset_hash'); changed['asset_hash']=canonical_hash(changed)
    (tmp_path/'manifest.json').write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='Asset checksum mismatch'):
        training_inputs(tmp_path,diagnostic=True,provisional_spec=SPEC)


def test_checkpoint_mode_spec_and_provenance():
    current={'asset_hash':'same','diagnostic':True,'provisional_spec':json.loads(SPEC.read_text())}
    check_training_identity(copy.deepcopy(current),current)
    check_training_identity({'asset_hash':'same'},{'asset_hash':'same','diagnostic':False,'provisional_spec':None})
    for mutate in ('mode','spec','source','asset'):
        previous=copy.deepcopy(current)
        if mutate=='mode': previous['diagnostic']=False
        if mutate=='spec': previous['provisional_spec']['actuators']['kp'][0]+=1
        if mutate=='source': previous['provisional_spec']['sources']['arm_passive_dynamics']='changed'
        if mutate=='asset': previous['asset_hash']='other'
        with pytest.raises(ValueError,match='Checkpoint'):
            check_training_identity(previous,current)
