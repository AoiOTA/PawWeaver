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


def run_loop(tmp_path, *, stop_after=None, failure=None, export_failure=False,group_steps=None,command_steps=None):
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
    steps=[]
    def step(actions):
        if group_steps is not None:
            extras["demonstration_group_stats"]=group_steps[len(steps)%len(group_steps)]
        if command_steps is not None:
            extras.update(command_steps[len(steps)%len(command_steps)])
        steps.append(1)
        return {},0.,False,extras
    env=SimpleNamespace(umi_pose_reward=None,reference=reference,spec=None,step=step)
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
        assert "base_linear_velocity_error_mps_mean" not in row
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


def test_group_training_statistics_weight_samples_and_ended_episodes(tmp_path):
    groups=[]
    for samples,error,ended,seconds in [(1,10.,1,.2),(2,2.,1,20.),(0,0.,0,0.)]:
        groups.append({"near":{"transitions":samples,"position_error_sum_m":error,
            "orientation_error_sum_rad":error/2,"falls":ended,"resets":ended,"timeouts":0,
            "ended_episodes":ended,"ended_episode_seconds_sum":seconds,
            "base_linear_velocity_error_sum_mps":error/5,"base_yaw_rate_error_sum_radps":error/10,
            "nonterminal_preclip_sum":-error,"nonterminal_postclip_sum":0.},
            "empty":{"transitions":0,"position_error_sum_m":0.,"orientation_error_sum_rad":0.,
                "falls":0,"resets":0,"timeouts":0,"ended_episodes":0,"ended_episode_seconds_sum":0.}})
    command_steps=[{'base_linear_velocity_error_mps':linear,'base_yaw_rate_error_radps':yaw}
                   for linear,yaw in [(1.,.1),(2.,.2),(6.,.6)]]
    run,_,_,_=run_loop(tmp_path,stop_after=1,group_steps=groups,command_steps=command_steps)
    run()
    row=json.loads((tmp_path/"metrics.jsonl").read_text())
    assert row["demonstration_near_transitions"]==3
    assert row["demonstration_near_position_error_sum_m"]==12.
    assert row["demonstration_near_position_error_mean_m"]==4.
    assert row["demonstration_near_orientation_error_mean_rad"]==2.
    assert row["demonstration_near_nonterminal_preclip_mean"]==-4.
    assert row["demonstration_near_base_linear_velocity_error_mean_mps"]==pytest.approx(.8)
    assert row["demonstration_near_base_yaw_rate_error_mean_radps"]==pytest.approx(.4)
    assert row["base_linear_velocity_error_mps_mean"]==3.
    assert row["base_yaw_rate_error_radps_mean"]==pytest.approx(.3)
    assert row["demonstration_near_ended_episode_seconds_mean"]==pytest.approx(10.1)
    assert row["demonstration_empty_transitions"]==0
    assert "demonstration_empty_position_error_mean_m" not in row
    assert "demonstration_empty_ended_episode_seconds_mean" not in row
