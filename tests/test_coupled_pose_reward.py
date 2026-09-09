import pytest
import torch

from pawweaver.task import sum_reward_terms


def test_joint_pose_reward_requires_both_errors_small():
    # Exact existing widths: good pose, poor position, poor orientation.
    position=torch.exp(-torch.tensor([0.,.6,0.]).square()/.15**2)
    orientation=torch.exp(-torch.tensor([0.,0.,2.]).square()/.5**2)
    terms={"tracking":position,"orientation_tracking":orientation}
    weights={"tracking":2.,"orientation_tracking":1.}
    old=sum_reward_terms(terms,weights)
    coupled=sum_reward_terms(terms,weights,coupled_pose=True)
    assert coupled[0]==old[0]==3
    assert torch.all(coupled[1:]<1e-6)
    assert old[1]>=1 and old[2]>=2


def test_default_sum_preserves_other_rewards_and_separate_termination():
    terms={"tracking":torch.tensor([.7,.3]),"orientation_tracking":torch.tensor([.2,.8]),
           "foot_slip":torch.tensor([.05,.1]),"termination":torch.ones(2)}
    weights={"tracking":2.,"orientation_tracking":1.,"foot_slip":-.1,"termination":-5.}
    previous=sum(weights[name]*value for name,value in terms.items() if name!="termination")
    assert torch.equal(sum_reward_terms(terms,weights),previous)
    assert torch.equal(sum_reward_terms(terms,weights,coupled_pose=False),previous)
    expected=3*terms["tracking"]*terms["orientation_tracking"]-.1*terms["foot_slip"]
    torch.testing.assert_close(sum_reward_terms(terms,weights,coupled_pose=True),expected)


@pytest.mark.parametrize("coupled",[False,True])
@pytest.mark.parametrize("missing",["tracking","orientation_tracking","foot_slip"])
def test_missing_required_reward_weight_still_fails(coupled,missing):
    terms={name:torch.ones(1) for name in ("tracking","orientation_tracking","foot_slip")}
    weights={"tracking":2.,"orientation_tracking":1.,"foot_slip":-.1}
    del weights[missing]
    with pytest.raises(KeyError,match=missing):
        sum_reward_terms(terms,weights,coupled_pose=coupled)
