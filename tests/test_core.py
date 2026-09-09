import dataclasses
import unittest

import numpy as np
import torch

from loco_manipulation.contracts import ActuatorSpec, JOINT_NAMES, RobotState, named_indices
from loco_manipulation.control import JointPD
from loco_manipulation.math import quat_apply, rpy_quat
from loco_manipulation.observations import ObservationBuilder, ObservationSpec


def fixture_spec(delay=0):
    """Synthetic actuator test fixture; deliberately not a hardware calibration."""
    return ActuatorSpec(JOINT_NAMES, (0.,)*18, (1.,)*18, (10.,)*18, (1.,)*18,
                        (-2.,)*18, (2.,)*18, (5.,)*18, (4.,)*18, (0.,)*18, (0.,)*18, (0.,)*18, (delay,)*18)


def fixture_state(batch=2):
    return RobotState(torch.zeros(batch,18), torch.zeros(batch,18), torch.zeros(batch,3),
                      torch.tensor([[1.,0.,0.,0.]]).repeat(batch,1), torch.zeros(batch,3),
                      torch.tensor([[0.3,0.,0.5]]).repeat(batch,1), torch.zeros(batch,3))


class TestCore(unittest.TestCase):
    def test_pd_saturation_and_physics_delay(self):
        pd = JointPD(fixture_spec(delay=2), 2)
        pd.command(torch.ones(2,18))
        z = torch.zeros(2,18)
        self.assertTrue(torch.equal(pd.torque(z,z), z))
        self.assertTrue(torch.equal(pd.torque(z,z), z))
        self.assertTrue(torch.equal(pd.torque(z,z), z+5))
        pd.reset(torch.tensor([0]))
        self.assertTrue(torch.equal(pd.torque(z,z)[0], z[0]))
        self.assertTrue(torch.equal(pd.torque(z,z)[1], z[1]+5))

    def test_named_mapping_and_limits_fail_closed(self):
        names = list(reversed(JOINT_NAMES))
        self.assertEqual(named_indices(names, JOINT_NAMES), list(range(17,-1,-1)))
        with self.assertRaises(ValueError):
            named_indices(names[:-1], JOINT_NAMES)
        with self.assertRaises(ValueError):
            dataclasses.replace(fixture_spec(), effort=(0.,)*18)

    def test_translation_rotation_equivariance(self):
        state = fixture_state()
        goal = torch.tensor([[1.,2.,0.6],[2.,1.,0.8]])
        b1 = ObservationBuilder(2,torch.zeros(18))
        b1.reset(torch.arange(2),state,goal,torch.zeros(2))
        first = b1.build(state,torch.zeros(2,18),torch.zeros(2))
        rot = torch.tensor(rpy_quat((0,0,1.2)),dtype=torch.float32).repeat(2,1)
        shift = torch.tensor([[30.,-10.,2.],[60.,10.,5.]])
        moved = fixture_state()
        moved.base_quat_w = rot
        moved.base_pos_w = shift
        moved.tcp_pos_w = quat_apply(rot,state.tcp_pos_w)+shift
        b2 = ObservationBuilder(2,torch.zeros(18))
        b2.reset(torch.arange(2),moved,quat_apply(rot,goal)+shift,torch.zeros(2))
        second = b2.build(moved,torch.zeros(2,18),torch.zeros(2))
        torch.testing.assert_close(first,second,atol=6e-6,rtol=1e-5)

    def test_history_remains_world_fixed_and_reset_is_per_environment(self):
        state = fixture_state()
        builder = ObservationBuilder(2,torch.zeros(18))
        goal = torch.ones(2,3)
        builder.reset(torch.arange(2),state,goal,torch.zeros(2))
        original = builder.goals_w.clone()
        state.base_pos_w[:,0] += 0.5
        obs = builder.build(state,torch.zeros(2,18),torch.zeros(2))
        self.assertTrue(torch.equal(builder.goals_w,original))
        start = 5*42+18+3
        torch.testing.assert_close(obs[:,start:start+3],torch.tensor([[.5,1.,1.]]).repeat(2,1))
        builder.reset(torch.tensor([0]),state,goal*2,torch.ones(2))
        torch.testing.assert_close(builder.goals_w[1],original[1])

    def test_bad_or_stale_measurements_do_not_poison_history(self):
        builder = ObservationBuilder(2,torch.zeros(18))
        state=fixture_state()
        builder.reset(torch.arange(2),state,torch.ones(2,3),torch.ones(2))
        builder.push_goal(torch.full((2,3),float('nan')),torch.zeros(2),torch.ones(2,dtype=torch.bool),torch.ones(2))
        obs=builder.build(state,torch.zeros(2,18),torch.ones(2)*2)
        self.assertTrue(torch.isfinite(obs).all())
        self.assertTrue(torch.equal(obs[:,-3],torch.zeros(2)))
        self.assertEqual(obs.shape,(2,ObservationSpec().size))


if __name__ == '__main__':
    unittest.main()
