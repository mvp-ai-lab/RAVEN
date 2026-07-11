"""
@author: Yanzuo Lu
@email:  oliveryanzuolu@gmail.com
"""
import bisect
import random
from typing import Any, List

import torch
from torch.nn.attention.flex_attention import and_masks, or_masks
from torch.utils.data import default_collate, get_worker_info


def partition_by_groups(data: List[Any], groups: int) -> List[List[Any]]:
    """
    Partition a list by groups.
    When indivisible, some groups may have more items than others.

    Examples:
        - data: [1,2,3,4,5]
        - groups: 2
        - return: [[1,3,5], [2,4]]
    """
    assert groups > 0
    return [data[i::groups] for i in range(groups)]


def get_portion_for_rank_and_worker(
    items: List[Any],
    rank: int,
    world_size:int,
    worker_id: int,
    num_workers: int,
    seed: int  # should be identical across all ranks
) -> List[Any]:
    """
    Get the portion of items for current rank and worker.
    """
    if world_size * num_workers < len(items):
        items = partition_by_groups(items, world_size)[rank]
        items = partition_by_groups(items, num_workers)[worker_id]
    else:
        seed = seed + rank * num_workers + worker_id
        random.Random(seed).shuffle(items)
    return items


def merge_dicts(dict1, dict2):
    """
    Recursively merge two dictionaries.
    """
    return {
        **dict1,
        **{k: merge_dicts(dict1[k], v) if k in dict1 and isinstance(dict1[k], dict) and isinstance(v, dict) else v
           for k, v in dict2.items()}
    }


def get_worker_id() -> int:
    """
    Get the current dataloader worker id.
    """
    return get_worker_info().id if get_worker_info() is not None else 0


def get_num_workers() -> int:
    """
    Get the total dataloader worker count.
    """
    return get_worker_info().num_workers if get_worker_info() is not None else 1


def get_collate_fn(collate_fn_name: str):
    """
    Get the collate function by name.

    Args:
        collate_fn_name (str): name of the collate function.
    Returns:
        collate_fn (callable): collate function.
    """
    if collate_fn_name == "default":
        return default_collate
    elif collate_fn_name == "to_list":
        return lambda examples: {key: [example[key] for example in examples] for key in examples[0].keys()}
    elif collate_fn_name == "first":
        return lambda x: x[0]


def prepare_attention_mask_per_sample(split_lens, attn_modes, device="cpu"):
    """
    nested_split_lens: A list of N lists of ints. Each int indicates the length of a split within
        a sample, where each sample contains multiple splits with different attn modes.
    nested_attn_modes: whether to use full attn in each split.
    """
    sample_len = sum(split_lens)
    attention_mask = torch.zeros((sample_len, sample_len), dtype=torch.bool, device=device)

    csum = 0
    for s, attn_mode in zip(split_lens, attn_modes):
        assert attn_mode in ['causal', 'full', 'noise']
        if attn_mode == "causal":
            attention_mask[csum:csum + s, csum:csum + s] = torch.ones((s, s), device=device).tril()
            attention_mask[csum:csum + s, :csum] = 1
        else:
            attention_mask[csum:csum + s, csum:csum + s] = torch.ones((s, s))
            attention_mask[csum:csum + s, :csum] = 1
        csum += s

    csum = 0
    for s, attn_mode in zip(split_lens, attn_modes):
        if attn_mode == "noise":
            attention_mask[:, csum : csum + s] = torch.zeros((sample_len, s))
            attention_mask[csum : csum + s, csum : csum + s] = torch.ones((s, s))
        csum += s

    attention_mask = torch.zeros_like(attention_mask, dtype=torch.float).masked_fill_(
        ~attention_mask, float("-inf")
    )

    return attention_mask


def prepare_flex_attention_mask(split_lens, attn_modes, sink=0, window_size=None, device="cpu", kv_split_lens=None):
    """
    Args:
        split_lens: List[int], the length of each query split.
        attn_modes: List[str], the mode of each split ('causal', 'full', 'noise').
        sink: int, the initial sink of non-noise splits that are always visible (default 0).
        window_size: int or None, the number of non-noise splits to look back from the current position (None means unlimited).
        device: device string or torch.device
        kv_split_lens: List[int] or None, the length of each key/value split. When None, use split_lens.

    Returns:
        q_ranges: (N, 2)
        k_ranges: (N, 2)
        attn_type_map: (N,)
    """
    if kv_split_lens is None:
        kv_split_lens = split_lens
    assert len(split_lens) == len(attn_modes) == len(kv_split_lens), \
        "split_lens, attn_modes and kv_split_lens should have the same length."

    q_ranges_list = []
    k_ranges_list = []
    attn_type_list = []

    q_split_starts = []
    q_split_ends = []
    csum = 0
    for l in split_lens:
        q_split_starts.append(csum)
        csum += l
        q_split_ends.append(csum)

    kv_split_starts = []
    kv_split_ends = []
    csum = 0
    for l in kv_split_lens:
        kv_split_starts.append(csum)
        csum += l
        kv_split_ends.append(csum)

    TYPE_FULL = 0
    TYPE_CAUSAL = 1

    # Preprocessing: establish index mapping for non-noise splits
    non_noise_indices = []  # Store indices of non-noise splits
    split_to_non_noise_idx = {}  # split index -> position in non-noise sequence
    for idx, mode in enumerate(attn_modes):
        if mode != 'noise':
            split_to_non_noise_idx[idx] = len(non_noise_indices)
            non_noise_indices.append(idx)

    # Double loop: for each Query split (i), determine which Key splits (j) it can see
    for i, (q_len, q_mode) in enumerate(zip(split_lens, attn_modes)):
        if q_len == 0:
            continue
        q_start = q_split_starts[i]
        q_end = q_split_ends[i]
        k_start = kv_split_starts[i]
        k_end = kv_split_ends[i]

        if k_start == k_end:
            raise ValueError(f"Query split {i} has non-zero q_len but zero kv_len.")

        # --- 1. Handle "seeing itself" (Diagonal) ---
        if q_mode == 'causal':
            self_type = TYPE_CAUSAL
        else:
            # 'full' and 'noise' internally see themselves as Full
            self_type = TYPE_FULL

        q_ranges_list.append([q_start, q_end])
        k_ranges_list.append([k_start, k_end])
        attn_type_list.append(self_type)

        # --- 2. Handle "seeing history" (Off-Diagonal) ---
        # Determine the logical position of the current Query
        # If it is text, look it up; if it is noise, it logically falls at the position of "the current processed non-noise number"
        if i in split_to_non_noise_idx:
            q_logic_idx = split_to_non_noise_idx[i]
        else:
            # Find the insertion position of i in non_noise_indices, which is the number of non-noise blocks before it
            q_logic_idx = bisect.bisect_right(non_noise_indices, i)

        for j in range(i):
            k_len = kv_split_lens[j]
            k_mode = attn_modes[j]

            if k_len == 0: continue

            # Core constraint: no one can see historical Noise types
            if k_mode == 'noise':
                continue

            # New: apply sink and window_size constraints
            j_non_noise_idx = split_to_non_noise_idx[j]  # j must be in non-noise

            # Check if j is in sink range
            is_in_sink = j_non_noise_idx < sink

            # Check if j is in window range
            is_in_window = True
            if window_size is not None:
                # At this point, q_logic_idx is defined for both noise and text and can be directly subtracted
                # Logic: current block's logical position - historical block's logical position
                distance = q_logic_idx - j_non_noise_idx
                is_in_window = distance <= window_size

            # Can only see if it meets either sink or window conditions
            if not (is_in_sink or is_in_window):
                continue

            # As long as history is not noise, the current segment (whether text or noise) can see it
            # This "seeing history" relationship is usually Full Attention
            k_start = kv_split_starts[j]
            k_end = kv_split_ends[j]

            q_ranges_list.append([q_start, q_end])
            k_ranges_list.append([k_start, k_end])
            attn_type_list.append(TYPE_FULL)

    # Convert to Tensor
    if len(q_ranges_list) > 0:
        q_ranges = torch.tensor(q_ranges_list, dtype=torch.int32, device=device)
        k_ranges = torch.tensor(k_ranges_list, dtype=torch.int32, device=device)
        attn_type_map = torch.tensor(attn_type_list, dtype=torch.int32, device=device)
    else:
        q_ranges = torch.zeros((0, 2), dtype=torch.int32, device=device)
        k_ranges = torch.zeros((0, 2), dtype=torch.int32, device=device)
        attn_type_map = torch.zeros((0,), dtype=torch.int32, device=device)

    """
    Quickly calculate the total number of valid elements in the sparse mask.
    Complexity: O(N), where N is the number of blocks. No memory allocation required.
    """
    # 1. Calculate the height (H) and width (W) of each Block
    q_lens = q_ranges[:, 1] - q_ranges[:, 0]
    k_lens = k_ranges[:, 1] - k_ranges[:, 0]

    # 2. Calculate the area of Full Attention (H * W)
    full_counts = q_lens * k_lens

    # 3. Calculate the area of Causal Attention (assuming a standard diagonal block: H*(H+1)/2)
    # Note: If a Causal Block is not square, through the bottom right alignment logic, it is usually a full-rank triangle
    # Here we use the most common autoregressive assumption: Causal Block is always square (q_len == k_len)
    causal_counts = (q_lens * (q_lens + 1)) // 2

    # 4. Select the corresponding count based on attn_type_map
    # 0: Full, 1: Causal
    # Use torch.where for vectorized selection
    block_counts = torch.where(attn_type_map == 1, causal_counts, full_counts)

    # 5. Sum
    attn_workloads = block_counts.sum().item()

    return q_ranges, k_ranges, attn_type_map, attn_workloads


def create_sparse_mask(document_lens, split_lens, attn_modes, device="cpu", sink=0, window_size=None):
    assert len(set(sink)) == 1, "Currently only support same sink for all samples."
    sink = sink[0]
    assert len(set(window_size)) == 1, "Currently only support same window_size for all samples."
    window_size = window_size[0]

    def causal_mask(b, h, q_idx, kv_idx):
        """ Standard lower triangle """
        return q_idx >= kv_idx

    def full_and_noise_mask(b, h, q_idx, kv_idx):
        """ Same sample's same mimo segment & non-text ==> within the same image """
        return (full_and_noise_seq_id[q_idx] == full_and_noise_seq_id[kv_idx]) & (full_and_noise_seq_id[q_idx] >= 0)

    def remove_noise_mask(b, h, q_idx, kv_idx):
        """ ~(is a noise image & in different mimo segment) ==> clean image or text or within the same noise image """
        return ~((noise_seq_id[kv_idx] >= 0) & (noise_seq_id[q_idx] != noise_seq_id[kv_idx]))

    def sample_mask(b, h, q_idx, kv_idx):
        """ Within the same sample """
        return document_id[q_idx] == document_id[kv_idx]

    def sink_window_mask(b, h, q_idx, kv_idx):
        q_non_noise_idx = non_noise_idx_map[q_idx]
        kv_non_noise_idx = non_noise_idx_map[kv_idx]

        same_split = (split_id[q_idx] == split_id[kv_idx])
        in_sink = kv_non_noise_idx < sink

        if window_size is None:
            # 0-dim constant: a shape-(1,) tensor leaks an extra dim through
            # create_block_mask's vmap (5D mask -> unpack error) on the
            # torch flex-attention fallback path.
            in_window = torch.ones((), dtype=torch.bool, device=device)
        else:
            distance = q_non_noise_idx - kv_non_noise_idx
            in_window = distance <= window_size

        return same_split | in_sink | in_window

    full_and_noise_tmp = []
    noise_tmp = []

    for i, (length, model) in enumerate(zip(split_lens, attn_modes)):
        value = i if model in ['full', 'noise'] else -1
        full_and_noise_tmp.extend([value] * length)
        value_noise = i if model == 'noise' else -1
        noise_tmp.extend([value_noise] * length)

    full_and_noise_seq_id = torch.Tensor(full_and_noise_tmp).to(device)
    noise_seq_id = torch.Tensor(noise_tmp).to(device)

    document_id = torch.cat([torch.full((l,), i) for i, l in enumerate(document_lens, start=1)]).to(device)

    # New: Construct split_id and non_noise_idx_map
    split_id_tmp = []
    non_noise_idx_tmp = []

    # First construct index mapping for non-noise splits
    non_noise_counter = 0
    non_noise_indices = []
    for i, mode in enumerate(attn_modes):
        if mode != 'noise':
            non_noise_indices.append(non_noise_counter)
            non_noise_counter += 1
        else:
            # Logical position of Noise = number of non-noise before it
            # Note: here we do not increment non_noise_counter because noise itself does not count towards non-noise sequence length
            non_noise_indices.append(non_noise_counter)

    # Assign split_id and non_noise_idx for each token
    for i, (length, mode) in enumerate(zip(split_lens, attn_modes)):
        split_id_tmp.extend([i] * length)
        non_noise_idx = non_noise_indices[i]
        non_noise_idx_tmp.extend([non_noise_idx] * length)

    split_id = torch.Tensor(split_id_tmp).to(device)
    non_noise_idx_map = torch.Tensor(non_noise_idx_tmp).to(device)

    """
    Positions in mask where 1 is satisfied:
    (Standard lower triangle or within the same image)
    and (is a clean image or text or within the same noise image)
    and (is within the same sample)
    and (satisfies sink/window constraints)
    """
    return and_masks(
        or_masks(causal_mask, full_and_noise_mask),
        remove_noise_mask,
        sample_mask,
        sink_window_mask
    )
