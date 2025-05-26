import torch


def _damerau_levenshtein(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute the raw Damerau-Levenshtein distance between two 1D integer tensors a and b.
    """
    # lengths
    la, lb = a.size(0), b.size(0)
    # maxdist for padding
    maxdist = la + lb
    # DP table with extra two rows/cols
    d = torch.full((la + 2, lb + 2), maxdist, dtype=torch.int64, device=a.device)
    d[0, 0] = maxdist
    d[1:, 1] = torch.arange(0, la + 1, dtype=torch.int64, device=a.device)
    d[1, 1:] = torch.arange(0, lb + 1, dtype=torch.int64, device=a.device)

    # last row each symbol was seen in a
    last = {}
    # fill initial last seen positions for all symbols in a or b
    for sym in torch.cat([a, b]).unique().tolist():
        last[sym] = 0

    for i in range(1, la + 1):
        db = 0
        for j in range(1, lb + 1):
            i1 = last[b[j - 1].item()]
            j1 = db
            cost = 0 if a[i - 1] == b[j - 1] else 1
            if cost == 0:
                db = j
            # substitution / match
            sub = d[i, j] + cost
            # insertion
            ins = d[i + 1, j] + 1
            # deletion
            delete = d[i, j + 1] + 1
            # transposition
            transp = d[i1, j1] + (i - i1 - 1) + 1 + (j - j1 - 1)
            d[i + 1, j + 1] = min(sub, ins, delete, transp)
        last[a[i - 1].item()] = i

    return d[la + 1, lb + 1]


def _handle_pred_length(predicted, pad_idx=1, eos_idx=2):
    """
    using argmax wasn't working due to the cases where eos is at position 0;
    so I had to improvise :D

    """
    pred_lengths = (predicted == eos_idx).long()  # get the positions of eos_idx, if any
    pred_lengths = (
        (
            torch.flip(
                pred_lengths, dims=[1]
            )  # reverse each row in order to use cumsum
            .cumsum(dim=1)  # fill with ones all the columns after the first eos_idx
            .gt(0)  # convert to boolean any value greater than 0
        )
        .flip(dims=[1])
        .sum(dim=1)
    )  # reverse again and sum the columns to get the length of the sequence

    pad_mask = (predicted != pad_idx).sum(dim=1)

    # fill 0s with the length of the sequence, since these sequences have no eos_idx
    pred_lengths = torch.where(pred_lengths == 0, pad_mask, pred_lengths)

    return pred_lengths


def normalized_damerau_levenshtein_batch(
    ground_truth: torch.Tensor,
    predicted: torch.Tensor,
    pad_idx: int = 0,
    eos_idx: int = 2,
) -> torch.Tensor:
    """
    Given:
      - ground_truth:   LongTensor [batch, prefix_len]
      - predicted:      LongTensor [batch, prefix_len]
      - pad_idx:        int, padding index to limit GT length
      - eos_idx:        int, end of sequence index to limit predicted length
    Returns:
      - FloatTensor [batch] of normalized distances in [0, 1].
    """

    batch = ground_truth.size(0)
    out = torch.zeros(batch, device=ground_truth.device)

    gt_lengths = (ground_truth != pad_idx).sum(dim=1)
    pred_lengths = _handle_pred_length(predicted, pad_idx, eos_idx)

    for idx in range(batch):
        gt_len = gt_lengths[idx].item()
        pr_len = pred_lengths[idx].item()
        a = ground_truth[idx, :gt_len]
        b = predicted[idx, :pr_len]

        if gt_len == 0 and pr_len == 0:
            # both empty: zero distance
            out[idx] = 0.0
            continue

        dist = _damerau_levenshtein(a, b).float()
        norm = 1 - (dist / max(gt_len, pr_len))
        out[idx] = norm

    return out
