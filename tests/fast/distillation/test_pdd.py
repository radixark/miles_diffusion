from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

import pytest
import torch
from torch import nn

from miles.distillation.pdd import ParallelLinear, block_update, pdd_loss, teacher_mean_velocity


@pytest.mark.parametrize("solver", ["euler", "midpoint"])
@pytest.mark.parametrize("descending", [False, True])
def test_target_uses_detached_student_state_and_signed_step(solver, descending):
    grid = torch.tensor([0.0, 0.2, 0.7, 1.0])
    if descending:
        grid = grid.flip(0)
    x = torch.tensor([[1.0, 2.0]], requires_grad=True)
    predictions = torch.tensor([[[2.0, 3.0]], [[4.0, 5.0]], [[6.0, 7.0]]], requires_grad=True)
    teacher_weight = nn.Parameter(torch.tensor(0.5))
    seen = []

    def teacher(state, time):
        seen.append((state.detach().clone(), torch.is_grad_enabled()))
        return teacher_weight * state + time

    loss = pdd_loss(predictions, x, grid, 0, 1, teacher, solver=solver).mean()
    expected_state = x.detach() + (grid[1] - grid[0]) * predictions[0].detach()
    velocity = 0.5 * expected_state + grid[1]
    if solver == "midpoint":
        h = grid[2] - grid[1]
        velocity = 0.5 * (expected_state + h * 0.5 * velocity) + grid[1] + h * 0.5
    torch.testing.assert_close(seen[0][0], expected_state)
    torch.testing.assert_close(loss, (predictions[1] - velocity).square().mean())
    assert len(seen) == (1 if solver == "euler" else 2)
    assert all(not enabled for _, enabled in seen)
    loss.backward()
    assert x.grad is None and teacher_weight.grad is None
    assert predictions.grad[0].count_nonzero() == 0
    assert predictions.grad[1].count_nonzero() > 0
    assert predictions.grad[2].count_nonzero() == 0


def test_head_initialization_and_shared_backbone_gradient():
    source = nn.Linear(3, 2)
    head = ParallelLinear(source, 4)
    hidden = torch.randn(5, 3, requires_grad=True)
    outputs = head(hidden)
    for output in outputs:
        torch.testing.assert_close(output, source(hidden))
    outputs[2].square().mean().backward()
    assert hidden.grad.abs().sum() > 0
    assert head.weight.grad[2].abs().sum() > 0
    assert head.weight.grad[[0, 1, 3]].count_nonzero() == 0


@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("block_size", [1, 2, 4, 8])
def test_oracle_mean_velocities_follow_exact_linear_ode_across_block_sizes(descending, block_size):
    """Independent analytic oracle: dx/dt = a*x on a nonuniform grid."""
    grid = torch.tensor([0, 0.03, 0.11, 0.24, 0.42, 0.58, 0.73, 0.9, 1], dtype=torch.float64)
    if descending:
        grid = grid.flip(0)
    a = torch.tensor(-0.7, dtype=torch.float64)
    initial = torch.tensor([[1.3, -2.1]], dtype=torch.float64)

    def oracle(state, time):
        # Each head is the analytic integral from its own interval, relative to
        # this call's block-start state. No sampler helpers construct the oracle.
        left = torch.exp(a * (grid[:-1] - time))
        right = torch.exp(a * (grid[1:] - time))
        means = (right - left) / (grid[1:] - grid[:-1])
        return means[:, None, None] * state

    actual = initial
    for start in range(0, 8, block_size):
        actual = block_update(actual, oracle(actual, grid[start]), grid, start, start + block_size)
    expected = initial * torch.exp(a * (grid[-1] - grid[0]))
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_invalid_inputs_fail_instead_of_broadcasting():
    with pytest.raises(ValueError):
        pdd_loss(torch.zeros(3, 2, 1), torch.zeros(2, 1), torch.tensor([0.0, 0.5, 0.4, 1.0]), 0, 1, None)
    with pytest.raises(ValueError):
        teacher_mean_velocity(lambda x, t: x[:, :1], torch.zeros(2, 3), torch.tensor(0.0), torch.tensor(0.1))
