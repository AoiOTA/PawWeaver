import numpy as np
import pytest
import torch

from pawweaver.task import GoalBank


def test_static_offsets_are_balanced_translated_and_hold_through_future_clock():
    bank=GoalBank(4,1000,"cpu",static_goal_offsets_m=[[0,-.05,0],[0,.05,0]])
    starts=torch.tensor([[0.,0.,.6],[8.,0.,.6],[0.,8.,.6],[8.,8.,.6]])
    orientations=torch.tensor([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]])
    ids=torch.arange(4)
    bank.reset(ids,starts,orientations)
    expected=starts+torch.tensor([[0,-.05,0],[0,.05,0]]*2)
    assert torch.equal(bank.positions,expected[:,None,:].expand(-1,1005,-1))
    assert np.array_equal(bank.family,np.zeros(4,dtype=int))
    for times in (torch.zeros(4,dtype=torch.long),torch.full((4,),999),torch.full((4,),1000)):
        assert torch.equal(bank.current(times),expected)
        assert torch.equal(bank.current_orientation(times),orientations)
        future,valid=bank.future(times)
        assert torch.equal(future,expected[:,None,:].expand(-1,4,-1))
        assert torch.equal(valid,(times[:,None]+torch.arange(1,5)<=1000)[:,:,None].expand(-1,-1,3))
    # Reset a reordered subset: offset follows environment identity, not subset order.
    old=bank.positions.clone()
    old_orientation=bank.orientations_wxyz.clone()
    bank.reset(torch.tensor([3,0]),starts[[3,0]]+1.,orientations[[0,3]])
    assert torch.equal(bank.positions[[1,2]],old[[1,2]])
    assert torch.equal(bank.positions[[3,0],0],expected[[3,0]]+1.)
    assert torch.equal(bank.orientations_wxyz[[1,2]],old_orientation[[1,2]])
    assert torch.equal(bank.orientations_wxyz[[3,0],0],orientations[[0,3]])
    bank.reset(torch.tensor([],dtype=torch.long),torch.empty(0,3),torch.empty(0,4))


@pytest.mark.parametrize("offsets",[[],[0,0,0],[[0,0]],[[0,float('nan'),0]],[[0,float('inf'),0]]])
def test_invalid_static_offsets_fail(offsets):
    with pytest.raises(ValueError,match="finite"):
        GoalBank(2,10,"cpu",static_goal_offsets_m=offsets)


def test_default_sampling_remains_seeded_and_dynamic():
    plain=GoalBank(8,10,"cpu",seed=21)
    explicit_none=GoalBank(8,10,"cpu",seed=21,static_goal_offsets_m=None)
    ids=torch.arange(8);starts=torch.zeros(8,3)
    orientations=torch.tensor([[.5,.5,.5,.5]]).repeat(8,1)
    plain.reset(ids,starts,orientations);explicit_none.reset(ids,starts,orientations)
    assert torch.equal(plain.positions,explicit_none.positions)
    assert np.array_equal(plain.family,explicit_none.family)
    assert bool((plain.positions[:,1:]-plain.positions[:,:-1]).abs().max()>0)
    assert torch.equal(plain.orientations_wxyz,orientations[:,None,:].expand(-1,15,-1))
