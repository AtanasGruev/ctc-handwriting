import torch


def blank_collapse(logprobs, audio_features_len, threshold, blank_idx):
    """
    :param logprobs: softmax-normalized probabilities in log-space, [1, T, N+1]
    :param audio_features_len: length of T as [1]
    :param threshold: collapse threshold probability in log-space
    :param blank_idx: index of blank label, i.e. N+1
    """

    # Apply implicit length mask from 'audio_features_len'
    masked_logprobs = logprobs[:audio_features_len]

    values, indices = torch.max(masked_logprobs, dim=-1)  # [1, T]
    mask = (values >= threshold) & (indices == blank_idx)  # [1, T]
    _, counts = torch.unique_consecutive(mask, return_counts=True)  # [1, T']

    # Boolean values to indicate start and end of predictions
    blank_begin, blank_end = mask[0].item(), mask[-1].item()

    # Store counts for initial and final blank predictions
    initial_blank_cnt, final_blank_cnt = 0, 0

    # Store counts for all blank or non-blank predictions
    blank_counts, non_blank_counts = counts[::2], counts[1::2]

    # Handling of blank predictions in the beginning
    if blank_begin:
        initial_blank_cnt = counts[0].item()  # fix count
    else:
        blank_counts = non_blank_counts  # swap to match
    initial_slice = initial_blank_cnt  # fix correct slice

    # Handling of blank predictions in the end
    final_slice = len(mask)
    if blank_end:
        final_blank_cnt = counts[-1].item()  # fix count
        final_slice = -final_blank_cnt  # fix correct slice

    # Keep only mid-blanks, initial and final are recorded
    if blank_begin:
        blank_counts = blank_counts[1:]
    if blank_end:
        blank_counts = blank_counts[:-1]

    # Collapsing strong blanks via masking
    mask = mask[initial_slice:final_slice]
    mask_shift = torch.roll(mask, shift=1)
    mask_shift[0] = False

    # Collapsed log-probablities and audio length
    collapsed_logprobs = masked_logprobs[initial_slice:final_slice][
        ~(mask & mask_shift)
    ]
    collapsed_audio_features_len = audio_features_len - (
        (blank_counts.sum() - len(blank_counts)) + initial_blank_cnt + final_blank_cnt
    )

    assert (
        collapsed_logprobs.shape[0] == collapsed_audio_features_len
    ), "Length mismatch, %s for log-probabilities and %s for audio-lengths" % (
        collapsed_logprobs.shape[0],
        collapsed_audio_features_len,
    )
    return collapsed_logprobs, collapsed_audio_features_len
