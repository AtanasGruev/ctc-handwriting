import torch


def blank_collapse(logprobs, audio_features_len, threshold, blank_idx):
    """
    :param logprobs: softmax-normalized probabilities in log-space, [1, T, N+1]
    :param audio_features_len: length of T as [1]
    :param threshold: collapse threshold probability in log-space
    :param blank_idx: index of blank label, i.e. N+1
    """

    values, indices = torch.max(logprobs, dim=-1)  # [1, T]
    mask = (values >= threshold) & (indices == blank_idx)  # [1, T]
    _, counts = torch.unique_consecutive(mask, return_counts=True)  # [1, T']

    # Boolean values to indicate start and end of predictions
    blank_begin, blank_end = mask[0].item(), mask[-1].item()

    initial_blank_cnt = counts[0].item() if blank_begin else 0
    final_blank_cnt = counts[-1].item() if blank_end else 0

    true_counts, false_counts = counts[::2], counts[1::2]
    if not blank_begin:
        true_counts, false_counts = false_counts, true_counts

    # TODO: Re-write with torch.roll() and omit double negations

    # Subtask 1: Collapse (strongly) blank labels via proper masking
    collapse_mask = mask[initial_blank_cnt:-(final_blank_cnt)]
    collapse_mask_shift = torch.cat((torch.tensor([False]), collapse_mask[:-1]))
    collapse_mask = collapse_mask & (~collapse_mask_shift)

    logprobs = logprobs[initial_blank_cnt:-(final_blank_cnt), :][~collapse_mask]

    # Subtask 2: Adjust audio_features_len to match collapsed length
    if blank_begin:
        true_counts = true_counts[1:]
    if blank_end:
        true_counts = true_counts[:-1]

    # Subtract mid-blanks that now have 1-count, also initial blanks and final blanks
    audio_features_len -= (
        (true_counts.sum() - len(true_counts)) + initial_blank_cnt + final_blank_cnt
    )

    # assert logprobs.shape[0] == audio_features_len
    return logprobs, audio_features_len
