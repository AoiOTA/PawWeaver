from types import SimpleNamespace
import pytest
import torch
from test_umi_pose_reward import fixed_env
from test_coupled_pose_reward import reward_inputs
from pawweaver.task import reward_terms
from pawweaver.math import quat_apply_inverse
from pawweaver.commanded_pose import task_pose_to_world, yaw_rate


def test_actual_command_tick_scores_same_target_with_post_base(fixed_env):
    env=fixed_env
    state=env.state()
    state.base_lin_vel_b=torch.zeros(2,3)
    state.base_pos_w[:,2]=.3
    pre_quat=state.base_quat_w.clone()
    env.config.update(task_mode="velocity_ee_pose",umi_clip_nonnegative=False,
        base_linear_velocity_sigma_mps=.15,base_yaw_rate_sigma_radps=.3)
    env.config['reward_weights'].update(base_linear_velocity_tracking=2.,base_yaw_rate_tracking=1.)
    env.scene.env_origins=torch.tensor([[10.,-2.,0.],[20.,1.,0.]])
    q=torch.tensor([[1.,0.,0.,0.]]).repeat(2,1)
    target=torch.tensor([[.4,.1,.6],[.4,.1,.6]])
    command=torch.tensor([[.2,-.1,.3],[.2,-.1,.3]])
    env.reference.current=lambda steps:target+steps[:,None]*.01
    env.reference.current_orientation=lambda steps:q
    env.reference.current_command=lambda steps:command+steps[:,None]*.02
    # A real command tick moves the base before the scoring state is read.
    def advance(_):
        state.base_pos_w[:,:2]+=torch.tensor([.3,-.2])
        angle=torch.tensor(.12)
        state.base_quat_w[:]=torch.tensor([torch.cos(angle/2),0.,0.,torch.sin(angle/2)])
        state.base_lin_vel_b[:]=quat_apply_inverse(state.base_quat_w,torch.tensor([[.2,-.1,0.]]).repeat(2,1))
        state.tcp_pos_w[:],state.tcp_quat_w[:]=task_pose_to_world(target,q,state.base_pos_w,
            state.base_quat_w,env.scene.env_origins[:,2])
    env.pd.command=advance
    _,_,done,extras=env.step(torch.zeros(2,18),auto_reset=False)
    assert not done.any()
    assert extras['tracking_error_m']==pytest.approx(0.,abs=1e-7)
    torch.testing.assert_close(extras['commanded_position_t'],target)
    torch.testing.assert_close(extras['velocity_command_yaw'],command)
    torch.testing.assert_close(extras['base_yaw_rate'],yaw_rate(pre_quat,state.base_quat_w,.02))
    pushed=env.observations.push_goal.call_args
    torch.testing.assert_close(pushed.args[0],target+.01)
    torch.testing.assert_close(pushed.args[1],torch.full((2,),.02))


def test_command_velocity_rewards_keep_tcp_velocity_separate(reward_inputs):
    baseline=reward_terms(**reward_inputs)
    terms=reward_terms(**reward_inputs,base_linear_velocity_error_yaw=torch.tensor([[0.,0.],[.15,0.],[0.,.3]]),
        base_yaw_rate_error=torch.tensor([0.,.3,.6]))
    expected=torch.exp(-torch.tensor([0.,1.,4.]))
    torch.testing.assert_close(terms['base_linear_velocity_tracking'],expected)
    torch.testing.assert_close(terms['base_yaw_rate_tracking'],expected)
    for key in baseline:
        torch.testing.assert_close(terms[key],baseline[key])


@pytest.mark.parametrize('sigma',[0.,-1.,float('nan'),float('inf')])
def test_command_reward_rejects_invalid_width(reward_inputs,sigma):
    with pytest.raises(ValueError):
        reward_terms(**reward_inputs,base_linear_velocity_error_yaw=torch.zeros(3,2),
            base_yaw_rate_error=torch.zeros(3),base_linear_velocity_sigma_mps=sigma)
