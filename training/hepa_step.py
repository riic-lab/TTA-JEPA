import torch
from training.runtime import move


def step(model, microbatches, objective, device):
    """Differentiate the effective-batch loss, then replay each encoder microbatch."""
    with torch.no_grad():
        outputs = [model.pretrain(move(batch, device)) for batch in microbatches]
    prediction = torch.cat([p for p, _ in outputs]).requires_grad_()
    target = torch.cat([t for _, t in outputs]).requires_grad_()
    loss = objective.hepa(prediction, target)
    grad_p, grad_t = torch.autograd.grad(loss, (prediction, target))
    cursor = 0
    for cpu in microbatches:
        p, t = model.pretrain(move(cpu, device))
        count = len(p)
        torch.autograd.backward((p, t), (grad_p[cursor:cursor+count], grad_t[cursor:cursor+count]))
        cursor += count
    return float(loss.detach())
