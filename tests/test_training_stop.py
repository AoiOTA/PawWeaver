"""Run the actual trainer loop with CPU fakes, without launching Isaac Lab."""
import ast
from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
import time
from types import SimpleNamespace

import pytest
import torch


def run_loop(tmp_path, *, stop_after=None, failure=None, export_failure=False):
    source=Path(__file__).resolve().parents[1]/"scripts/train.py"
    tree=ast.parse(source.read_text())
    loop=next(node for node in ast.walk(tree) if isinstance(node,ast.For)
              and ast.unparse(node.target)=="iteration")
    stop_file=tmp_path/"stop"
    updates=[]
    saves=[]
    exports=[]

    def update():
        if len(updates)+1==stop_after:
            stop_file.touch()
        if failure is not None:
            raise failure("during PPO update")
        updates.append(1)
        return {"loss":1.}

    def save(checkpoint,path):
        saves.append((deepcopy(checkpoint),path))

    def export(*args,**kwargs):
        if export_failure:
            raise OSError("export failed")
        exports.append(deepcopy(kwargs["metadata"]))

    reference=SimpleNamespace(sampler=SimpleNamespace(state_dict=lambda:{}),
        rng=SimpleNamespace(bit_generator=SimpleNamespace(state={})))
    extras={"tracking_error_m":.1,"orientation_error_rad":.2,"fall_fraction":0.}
    extras["reward_diagnostics"]={"weighted_nonterminal_terms":{"tracking":torch.tensor([1.,3.])},
        "nonterminal_preclip":torch.tensor([-1.,3.]),"nonterminal_postclip":torch.tensor([0.,3.])}
    env=SimpleNamespace(umi_pose_reward=None,reference=reference,spec=None,
        step=lambda actions: ({},0.,False,extras))
    algorithm=SimpleNamespace(act=lambda obs:None,process_env_step=lambda *args:None,
        compute_returns=lambda obs:None,update=update,save=lambda:{"updates":len(updates)},
        optimizer=SimpleNamespace(param_groups=[{"lr":.001}]))
    actor=SimpleNamespace(distribution=SimpleNamespace(std_param=torch.tensor([.2]*12+[.4]*6)))
    namespace=dict(args=SimpleNamespace(output=tmp_path,stop_file=stop_file,iterations=4,
        num_envs=2,diagnostic=False),start_iteration=7,config={"rollout_steps":3},
        torch=SimpleNamespace(inference_mode=nullcontext,get_rng_state=lambda:[],
            cuda=SimpleNamespace(get_rng_state_all=lambda:[]),save=save),
        time=time,json=json,obs={},env=env,algorithm=algorithm,metadata={},actor=actor,
        manifest={},export_bundle=export,writer=SimpleNamespace(add_scalar=lambda *args:None))
    code=compile(ast.Module(body=[loop],type_ignores=[]),str(source),"exec")
    return lambda:exec(code,namespace),updates,saves,exports


@pytest.mark.parametrize("stop_after,completed,status",[(2,2,"stopped_early"),(None,4,"completed"),(4,4,"completed")])
def test_stop_saves_complete_update_and_actual_budget(tmp_path,stop_after,completed,status):
    run,updates,saves,exports=run_loop(tmp_path,stop_after=stop_after)
    run()
    assert len(updates)==completed
    assert len(saves)==len(exports)==1
    checkpoint,path=saves[0]
    assert checkpoint["algorithm"]["updates"]==completed
    assert checkpoint["iteration"]==7+completed-1
    assert path.name==f"checkpoint_{7+completed-1:06d}.pt"
    metadata=json.loads((tmp_path/"run.json").read_text())
    assert metadata==checkpoint["metadata"]==exports[0]
    progress=metadata["training_progress"]
    assert progress["status"]==status
    assert progress["requested_iterations"]==4
    assert progress["completed_iterations"]==completed
    assert progress["completed_transitions"]==completed*2*3
    assert progress["stop_reason"]==("stop_file" if stop_after else "iteration_budget")
    metrics=[json.loads(line) for line in (tmp_path/"metrics.jsonl").read_text().splitlines()]
    assert len(metrics)==completed
    for row in metrics:
        assert row["learning_rate"]==.001
        assert row["leg_action_std_mean"]==pytest.approx(.2)
        assert row["arm_action_std_mean"]==pytest.approx(.4)
        assert row["reward_weighted_tracking_mean"]==2.
        assert row["reward_nonterminal_preclip_mean"]==1.
        assert row["reward_nonterminal_postclip_mean"]==1.5
    assert (tmp_path/"stop").exists()==bool(stop_after)  # Do not consume requests silently.


@pytest.mark.parametrize("failure",[KeyboardInterrupt,FloatingPointError,RuntimeError])
def test_update_failure_does_not_save_partial_update_even_with_stop(tmp_path,failure):
    run,updates,saves,exports=run_loop(tmp_path,stop_after=1,failure=failure)
    with pytest.raises(failure,match="during PPO update"):
        run()
    assert updates==saves==exports==[]
    assert not (tmp_path/"run.json").exists()


def test_export_failure_propagates_without_completed_run_status(tmp_path):
    run,updates,saves,exports=run_loop(tmp_path,stop_after=1,export_failure=True)
    with pytest.raises(OSError,match="export failed"):
        run()
    assert len(updates)==len(saves)==1
    assert exports==[]
    assert not (tmp_path/"run.json").exists()
