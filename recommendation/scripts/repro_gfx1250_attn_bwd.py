#!/usr/bin/env python3
"""Standalone gfx1250 reproducer for the ``_hstu_attn_bwd`` memory-access fault.

This is defect 6 in docs/mi450.md reduced to a single Triton kernel. The
original ``BLOCK_N=128`` config faults on Triton main @ 7ff97e3109::

    AMDGCN_USE_BUFFER_OPS=0 AMD_SERIALIZE_KERNEL=3 \
      python scripts/repro_gfx1250_attn_bwd.py --part bwd --layout contiguous \
      --bw-config m=32,n=128,warps=4,stages=1,nonkdim=16,waves=0

    Memory access fault by GPU node-2 ... Reason: Page not present.

On gfx1250 with Triton 3.8 or newer, the production default uses
``BLOCK_N=64`` and this script is its randomized stress test.

No dataset, no TBE, no TorchRec, no dataloader: just the HSTU jagged attention
backward over a fresh random sequence layout each iteration. What the loop adds
over the existing single-shot op tests is the *variation* -- every step brings
different jagged extents, the way the real dataloader feeds the trainer -- and
enough steps for the fault to land (tens to a few hundred).

How the blame was narrowed (each claim is a run in docs/mi450.md defect [6]):

  * ``--part fwd`` is clean for 2000 iterations; ``--part bwd`` faults. So the
    forward kernel is not implicated, the backward one is.
  * ``--layout strided`` vs ``contiguous`` both fault, so the packed-uvqk
    stride the fused call site passes is not the trigger.
  * ``--contextual 0 --no-targets`` still faults, and fastest, so neither the
    contextual-prefix nor the multiple-targets masking path is required.
  * ``--layout guarded`` over-allocates dq/dk/dv and checks a sentinel tail:
    the guard is never clobbered before the fault, and the fault address sits
    far outside the heap range, so this is a wild pointer rather than a small
    overrun past the end of the gradient buffers.

Needs the repo on PYTHONPATH. One process per run.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import triton

from generative_recommenders.ops.triton.triton_hstu_attention import (
    _bwd_pre_hook,
    _hstu_attn_bwd,
    triton_hstu_attention_bwd,
    triton_hstu_attention_fwd,
)

# gin-default yambda-5b HSTU ranker (see DlrmHSTUConfig in the e2e log).
BATCH = 8
HEADS = 4
ATTN_DIM = 128
HIDDEN_DIM = 128
MAX_SEQ_LEN = 4096
CONTEXTUAL_SEQ_LEN = 8
MAX_TARGETS = 1
MIN_HISTORY = 64
HISTORY_LENGTH = 4086
DTYPE = torch.bfloat16

# Packed uvqk width the fused path allocates: u, v (hidden) then q, k (attn).
UVQK_WIDTH = 2 * HEADS * (HIDDEN_DIM + ATTN_DIM)


def _print_env() -> None:
    props = torch.cuda.get_device_properties(0)
    print(f"torch {torch.__version__} | hip {torch.version.hip}", flush=True)
    print(f"triton {triton.__version__} | arch {props.gcnArchName}", flush=True)
    print(
        f"AMDGCN_USE_BUFFER_OPS={os.environ.get('AMDGCN_USE_BUFFER_OPS', '<unset>')} "
        f"AMD_SERIALIZE_KERNEL={os.environ.get('AMD_SERIALIZE_KERNEL', '<unset>')} "
        f"PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '<unset>')}",
        flush=True,
    )
    print(
        "backward configs: "
        + ", ".join(str(config) for config in _hstu_attn_bwd.configs),
        flush=True,
    )


def _layout(
    device: torch.device,
    rng: torch.Generator,
    fixed=None,
    contextual: int = CONTEXTUAL_SEQ_LEN,
):
    """One yambda-5b-shaped batch: per-row history plus a single target."""
    if fixed is not None:
        lengths = torch.tensor(fixed, dtype=torch.int64, device=device)
    else:
        uih = torch.randint(
            MIN_HISTORY, HISTORY_LENGTH, (BATCH,), generator=rng, dtype=torch.int64
        ).to(device)
        lengths = uih + MAX_TARGETS + contextual
    offsets = torch.zeros((lengths.numel() + 1,), dtype=torch.int64, device=device)
    offsets[1:] = torch.cumsum(lengths, dim=0)
    num_targets = torch.full(
        (lengths.numel(),), MAX_TARGETS, dtype=torch.int64, device=device
    )
    return (
        offsets,
        num_targets,
        int(lengths.max().item()),
        int(offsets[-1].item()),
        lengths,
    )


def _strided_qkv(total: int, device: torch.device):
    """q/k/v as views into one packed uvqk buffer, exactly like the fused path."""
    uvqk = torch.empty((total, UVQK_WIDTH), device=device, dtype=DTYPE).uniform_(
        -0.1, 0.1
    )
    duvqk = torch.empty_like(uvqk)
    widths = [
        HIDDEN_DIM * HEADS,
        HIDDEN_DIM * HEADS,
        ATTN_DIM * HEADS,
        ATTN_DIM * HEADS,
    ]
    _, v, q, k = uvqk.split(widths, dim=1)
    _, dv, dq, dk = duvqk.split(widths, dim=1)
    q = q.view(-1, HEADS, ATTN_DIM)
    k = k.view(-1, HEADS, ATTN_DIM)
    v = v.view(-1, HEADS, HIDDEN_DIM)
    dq = dq.view(-1, HEADS, ATTN_DIM)
    dk = dk.view(-1, HEADS, ATTN_DIM)
    dv = dv.view(-1, HEADS, HIDDEN_DIM)
    return (q, k, v, dq, dk, dv), (uvqk, duvqk)


GUARD_ELEMS = 1 << 16  # 128 KiB of bf16 sentinel after each output buffer
GUARD_FILL = -12345.0


def _guarded_qkv(total: int, device: torch.device):
    """Contiguous q/k/v, but dq/dk/dv sit at the front of over-allocated buffers.

    The tail of each buffer is filled with a sentinel and checked after the
    kernel: if the backward kernel writes past the logical end of dq/dk/dv the
    sentinel changes, which pins an out-of-bounds write without depending on
    the overflow happening to land on an unmapped page.
    """

    def mk(dim):
        return torch.empty((total, HEADS, dim), device=device, dtype=DTYPE).uniform_(
            -0.1, 0.1
        )

    def mk_guarded(dim):
        n = total * HEADS * dim
        buf = torch.full((n + GUARD_ELEMS,), GUARD_FILL, device=device, dtype=DTYPE)
        return buf[:n].view(total, HEADS, dim), buf

    q, k, v = mk(ATTN_DIM), mk(ATTN_DIM), mk(HIDDEN_DIM)
    dq, dq_buf = mk_guarded(ATTN_DIM)
    dk, dk_buf = mk_guarded(ATTN_DIM)
    dv, dv_buf = mk_guarded(HIDDEN_DIM)
    guards = [
        ("dq", dq_buf, total * HEADS * ATTN_DIM),
        ("dk", dk_buf, total * HEADS * ATTN_DIM),
        ("dv", dv_buf, total * HEADS * HIDDEN_DIM),
    ]
    return (q, k, v, dq, dk, dv), guards


def _check_guards(guards, step: int) -> bool:
    for name, buf, n in guards:
        tail = buf[n:]
        bad = (tail != GUARD_FILL).nonzero()
        if bad.numel():
            first = int(bad[0].item())
            print(
                f"!! OOB WRITE step={step} buffer={name} "
                f"{bad.numel()} of {tail.numel()} guard elements clobbered, "
                f"first at +{first} elements ({first * 2} bytes) past the end "
                f"(logical size {n} elements)",
                flush=True,
            )
            return False
    return True


def _contiguous_qkv(total: int, device: torch.device):
    def mk(dim):
        return torch.empty((total, HEADS, dim), device=device, dtype=DTYPE).uniform_(
            -0.1, 0.1
        )

    q, k, v = mk(ATTN_DIM), mk(ATTN_DIM), mk(HIDDEN_DIM)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    return (q, k, v, dq, dk, dv), None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--layout", choices=("strided", "contiguous", "guarded"), default="strided"
    )
    parser.add_argument("--part", choices=("both", "fwd", "bwd"), default="both")
    parser.add_argument("--iters", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument(
        "--bw-config",
        default="",
        help="override the backward config, for example "
        "'m=32,n=64,warps=4,stages=1,nonkdim=16,waves=0,sp=0,unroll=1'",
    )
    parser.add_argument(
        "--contextual",
        type=int,
        default=CONTEXTUAL_SEQ_LEN,
        help="contextual_seq_len; 0 disables the contextual-prefix masking path",
    )
    parser.add_argument(
        "--no-targets",
        action="store_true",
        help="pass num_targets=None, disabling the multiple-targets path",
    )
    parser.add_argument(
        "--lengths",
        default="",
        help="comma-separated per-row sequence lengths (incl. target and "
        "contextual tokens); replays one fixed layout instead of sampling",
    )
    args = parser.parse_args()

    if os.environ.get("AMDGCN_USE_BUFFER_OPS") != "0":
        raise RuntimeError("set AMDGCN_USE_BUFFER_OPS=0 before running this repro")

    if args.bw_config:
        spec = dict(item.split("=") for item in args.bw_config.split(","))
        config = triton.Config(
            {
                "BLOCK_M": int(spec.get("m", 32)),
                "BLOCK_N": int(spec.get("n", 128)),
                "matrix_instr_nonkdim": int(spec.get("nonkdim", 16)),
                "waves_per_eu": int(spec.get("waves", 0)),
                "SEQUENCE_PARALLEL": bool(int(spec.get("sp", 0))),
                "UNROLL": int(spec.get("unroll", 1)),
            },
            num_stages=int(spec.get("stages", 1)),
            num_warps=int(spec.get("warps", 4)),
            pre_hook=_bwd_pre_hook,
        )
        _hstu_attn_bwd.configs = [config]
        print(f"-- backward config override: {config}", flush=True)

    _print_env()
    device = torch.device("cuda")
    rng = torch.Generator().manual_seed(args.seed)
    make = {
        "strided": _strided_qkv,
        "contiguous": _contiguous_qkv,
        "guarded": _guarded_qkv,
    }[args.layout]
    print(
        f"-- {args.layout}/{args.part}: {args.iters} iterations of "
        f"triton_hstu_attention, fresh jagged layout per step",
        flush=True,
    )
    tag = f"{args.layout}/{args.part}"
    fixed = (
        [int(t) for t in args.lengths.split(",") if t.strip()] if args.lengths else None
    )
    for step in range(args.iters):
        offsets, num_targets, max_seq, total, lengths = _layout(
            device, rng, fixed, contextual=args.contextual
        )
        if args.no_targets:
            num_targets = None
        (q, k, v, dq, dk, dv), extra = make(total, device)
        if step == 0:
            print(
                f"-- q.stride={q.stride()} contiguous={q.is_contiguous()}",
                flush=True,
            )
        # Logged before the kernels run, so after an async fault the last line
        # in the log is the layout that faulted rather than the one before it.
        if args.print_every and step % args.print_every == 0:
            torch.cuda.synchronize()
            print(
                f"-- {tag}: step={step} N={total} max_seq={max_seq} "
                f"lengths={lengths.tolist()}",
                flush=True,
            )
        if args.part in ("both", "fwd"):
            out = triton_hstu_attention_fwd(
                N=max_seq,
                alpha=1.0 / (ATTN_DIM**0.5),
                q=q,
                k=k,
                v=v,
                seq_offsets=offsets,
                num_targets=num_targets,
                max_attn_len=0,
                contextual_seq_len=args.contextual,
                sort_by_length_indices=None,
                enable_tma=False,
                num_softmax_heads=0,
            )
        else:
            # bwd only: stand in for the forward result without running it.
            out = torch.empty_like(v)
        if args.part in ("both", "bwd"):
            triton_hstu_attention_bwd(
                dout=torch.randn_like(out) * 0.1,
                q=q,
                k=k,
                v=v,
                dq=dq,
                dk=dk,
                dv=dv,
                seq_offsets=offsets,
                num_targets=num_targets,
                N=max_seq,
                alpha=1.0 / (ATTN_DIM**0.5),
                max_attn_len=0,
                contextual_seq_len=args.contextual,
                sort_by_length_indices=None,
                enable_tma=False,
                num_softmax_heads=0,
            )
        if args.layout == "guarded" and not _check_guards(extra, step):
            return 1
    torch.cuda.synchronize()
    print(f"-- {tag}: PASS ({args.iters} iterations)", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"EXCEPTION {type(exc).__name__}: {exc}", flush=True)
        sys.exit(1)
