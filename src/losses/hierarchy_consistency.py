import torch


def hierarchy_consistency_loss(logits, parent_child_pairs):
    if not parent_child_pairs:
        return torch.tensor(0.0, device=logits.device)
    probs = torch.sigmoid(logits)
    loss = 0.0
    for p, c in parent_child_pairs:
        gap = probs[:, c] - probs[:, p]
        loss = loss + torch.relu(gap).mean()
    return loss / len(parent_child_pairs)
