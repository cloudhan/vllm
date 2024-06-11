import random
from typing import List, Optional, Tuple

import torch

from vllm import _custom_ops as ops
from vllm.utils import get_max_shared_memory_bytes, is_hip

FLOAT32_BYTES = torch.finfo(torch.float).bits // 8
# This will change depending on the compute capability.
# - 512 as a buffer
MAX_SEQ_LEN = get_max_shared_memory_bytes() // FLOAT32_BYTES - 512
# There may not be enough gpu memory due to large NUM_BLOCKS.
# Reduce NUM_BLOCKS when it happens.
# NUM_BLOCKS = 4321  # Arbitrary values for testing
NUM_BLOCKS = 10000
PARTITION_SIZE = 512
# flshattF and tritonflashattF supported: {torch.float16, torch.bfloat16}
DTYPES = [torch.half, torch.bfloat16, torch.float
          ] if not is_hip() else [torch.half, torch.bfloat16]
NUM_GEN_SEQS = [7]  # Arbitrary values for testing
NUM_PREFILL_SEQS = [3]  # Arbitrary values for testing
NUM_HEADS = [(40, 40), (64, 8)]  # Arbitrary values for testing

# FlashAttention forward only supports head dimension at most 128
# https://github.com/ROCmSoftwarePlatform/flash-attention/blob/3d2b6f5d037782cc2c906909a46fb7e2e1b48b25/csrc/flash_attn_rocm/flash_api.cpp#L62
HEAD_SIZES = [64, 80, 96, 112, 128, 192, 256
              ] if not is_hip() else [64, 80, 96, 112, 128]

BLOCK_SIZES = [16, 32]
USE_ALIBI = [False, True]
KV_CACHE_DTYPE = ["auto", "fp8"]
SEEDS = [0]
CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)
]

def generate_page_mapping(
    max_num_pages: int,
    num_seqs: int,
    max_context_len: int,
    page_size: int,
):
    unique_page_mapping = [i for i in range(max_num_pages)]
    max_num_pages_per_seq = (max_context_len + page_size - 1) // page_size
    random.shuffle(unique_page_mapping)
    page_table = []
    for i in range(num_seqs):
        page_table.append(unique_page_mapping[i * max_num_pages_per_seq:(i + 1) * max_num_pages_per_seq])
    assert len(page_table[-1]) == max_num_pages_per_seq, "alloc more pages to allow generating unique page mapping"
    return page_table


def run_paged_attention(
    kv_cache_factory,
    version: str,
    num_seqs: int,
    seqlen,
    num_heads: Tuple[int, int],
    head_size: int,
    use_alibi: bool,
    block_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    seed: int,
    device: str,
) -> None:
    random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_default_device(device)
    scale = float(1.0 / (head_size**0.5))
    num_query_heads, num_kv_heads = num_heads
    query = torch.empty(num_seqs, num_query_heads, head_size, dtype=dtype)
    query.uniform_(-scale, scale)

    assert num_query_heads % num_kv_heads == 0
    num_queries_per_kv = num_query_heads // num_kv_heads
    alibi_slopes = None
    if use_alibi:
        alibi_slopes = torch.randn(num_query_heads, dtype=torch.float)

    if seqlen is None:
        seq_lens = [random.randint(1, MAX_SEQ_LEN) for _ in range(num_seqs)]
        seq_lens[-1] = MAX_SEQ_LEN
    else:
        seq_lens = [seqlen] * num_seqs
    max_seq_len = max(seq_lens)
    seq_lens = torch.tensor(seq_lens, dtype=torch.int)

    # Create the block tables.
    # max_num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    # block_tables = []
    # flatten_block_tables = []
    # for _ in range(num_seqs):
    #     block_table = [
    #         random.randint(0, NUM_BLOCKS - 1)
    #         for _ in range(max_num_blocks_per_seq)
    #     ]
    #     block_tables.append(block_table)
    #     flatten_block_tables.extend(block_table)
    block_tables = generate_page_mapping(NUM_BLOCKS, num_seqs, max_seq_len, block_size)
    flatten_block_tables = []
    for block_table in block_tables:
        flatten_block_tables.extend(block_table)
    print("num unique pages:", len(set(flatten_block_tables)))
    block_tables = torch.tensor(block_tables, dtype=torch.int)

    # Create the KV caches.
    key_caches, value_caches = kv_cache_factory(NUM_BLOCKS, block_size, 1,
                                                num_kv_heads, head_size,
                                                kv_cache_dtype, dtype, seed,
                                                device)
    key_cache, value_cache = key_caches[0], value_caches[0]

    # Using default kv_scale
    kv_scale = 1.0

    # Call the paged attention kernel.
    output = torch.empty_like(query)
    if version == "v1":
        ops.paged_attention_v1(
            output,
            query,
            key_cache,
            value_cache,
            num_kv_heads,
            scale,
            block_tables,
            seq_lens,
            block_size,
            max_seq_len,
            alibi_slopes,
            kv_cache_dtype,
            kv_scale,
        )
    elif version == "v2":
        num_partitions = ((max_seq_len + PARTITION_SIZE - 1) // PARTITION_SIZE)
        assert PARTITION_SIZE % block_size == 0
        num_seqs, num_heads, head_size = output.shape
        tmp_output = torch.empty(
            size=(num_seqs, num_heads, num_partitions, head_size),
            dtype=output.dtype,
        )
        exp_sums = torch.empty(
            size=(num_seqs, num_heads, num_partitions),
            dtype=torch.float32,
        )
        max_logits = torch.empty_like(exp_sums)
        ops.paged_attention_v2(
            output,
            exp_sums,
            max_logits,
            tmp_output,
            query,
            key_cache,
            value_cache,
            num_kv_heads,
            scale,
            block_tables,
            seq_lens,
            block_size,
            max_seq_len,
            alibi_slopes,
            kv_cache_dtype,
            kv_scale,
        )
    else:
        raise AssertionError(f"Unknown version: {version}")


if __name__ == "__main__":
    from conftest import create_kv_caches_with_random as create_kv_caches
    run_paged_attention(
        create_kv_caches,
        version = "v1",
        num_seqs= 64,
        seqlen=2048,
        num_heads=(32,32),
        head_size=96,
        use_alibi=False,
        block_size=16,
        dtype=torch.float16,
        kv_cache_dtype="fp8",
        seed=0,
        device="cuda",
    )
