# gfx1250 WMMA / extended-VGPR reproducer

This directory isolates the LLVM machine-code condition behind the MI450 A0
HSTU backward fault. It is a minimized, checked LLVM MIR reproducer for the
hazardous allocation shape. The runtime reproducer remains
`../../scripts/repro_gfx1250_attn_bwd.py` because deleting the kernel's register
pressure also deletes the intermittent hardware symptom.

Validated on MI450 A0 (`gfx1250`, device revision `0x00`) with Triton
`7ff97e3109` and LLVM `ce3529423abda3fc4ad0b542daa50af21e539a29`. All runtime
tests set `AMDGCN_USE_BUFFER_OPS=0`; buffer operations are a separate gfx1250
hang and are not part of this fault.

## What is confirmed

The runtime root-cause class is confirmed: an address-critical value held in
an extended VGPR across the WMMA region is not reliably preserved/selected on
the tested A0. It is not yet known whether the final defect is A0 silicon or a
missing LLVM erratum rule. This MIR test reproduces the LLVM-generated machine
shape; by design it does not claim that two isolated WMMAs are sufficient to
fault. The full kernel's register pressure and instruction stream are required
to observe the intermittent corruption.

Two independent changes remove the fault:

1. Recompute the same lane offset after the final WMMA, where LLVM allocates it
   to a short-lived low VGPR.
2. Compile the identical original LLVM IR with extended VGPR addressing
   disabled.

Neither changes the address formula. Both stop the final address from relying
on the old `v257` value. Conversely, delays and waits that retain `v257` do not
reliably help. That is why the conclusion is extended-VGPR value survival or
selection, rather than store lowering or a generic WMMA spacing problem.

## Intuition

gfx1250 instruction encodings contain only the low eight bits of a VGPR index.
For example, `v1` and `v257` have the same encoded register number. The
stateful `S_SET_VGPR_MSB` instruction supplies the missing high bits separately
for source and destination operands. LLVM therefore emits this sequence:

1. Select extended destination bank 1 and write the lane offset to logical
   `v257`.
2. Reprogram the operand banks repeatedly for hundreds of WMMAs.
3. Select extended source bank 1 and consume logical `v257` in a global-address
   calculation.

In LLVM's physical-register model, the value is still live and no intervening
WMMA explicitly overlaps it, so ordinary liveness and hazard checks consider
the sequence valid. At runtime, lane 0 instead contains an accumulator-like
FP32 bit pattern. Lanes 1--31 retain the expected affine offsets. The next
integer multiply amplifies that one corrupt word into an arbitrary global
address, and the first final-DV `global_store_b128` raises the memory fault.
The store is the observer, not the producer, of the corruption.

## Input

`gfx1250-wmma-live-extended-vgpr.mir` keeps physical `v257` live across two
independent BF16 WMMAs, matching the important property of the failing
`_hstu_attn_bwd` allocation. Run it with LLVM `ce352942`:

```bash
git -C llvm-project checkout ce3529423abda3fc4ad0b542daa50af21e539a29
cmake -G Ninja -S llvm-project/llvm -B llvm-project/build \
  -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD=AMDGPU
ninja -C llvm-project/build llc FileCheck

LLVM_BIN="$PWD/llvm-project/build/bin"
"$LLVM_BIN/llc" -mtriple=amdgpu12.50 \
  -run-pass=post-RA-hazard-rec,amdgpu-lower-vgpr-encoding \
  -o /tmp/gfx1250-wmma-live-extended-vgpr.out \
  recommendation/reproducers/llvm/gfx1250-wmma-live-extended-vgpr.mir
"$LLVM_BIN/FileCheck" \
  recommendation/reproducers/llvm/gfx1250-wmma-live-extended-vgpr.mir \
  < /tmp/gfx1250-wmma-live-extended-vgpr.out
```

`FileCheck` exits zero and the output contains the exact mode transitions shown
below. A prebuilt LLVM at the same commit can be used by setting `LLVM_BIN` and
skipping the configure/build commands.

For the runtime symptom, use Triton `7ff97e3109` linked to that LLVM, run from
the `recommendation` directory on an MI450, and keep buffer operations off:

```bash
AMDGCN_USE_BUFFER_OPS=0 AMD_SERIALIZE_KERNEL=3 \
PYTHONPATH=/path/to/triton/python:. \
python scripts/repro_gfx1250_attn_bwd.py \
  --part bwd --layout contiguous --contextual 0 --no-targets \
  --iters 4000 --print-every 500 --seed 4321 \
  --bw-config m=32,n=128,warps=4,stages=1,nonkdim=16,waves=0
```

On the tested A0, the extended-VGPR binary faults at step 0. The same emitted
LLVM IR passed all 4,000 iterations after recompilation with:

```bash
"$LLVM_BIN/llc" -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1250 \
  -mattr=-1024-addressable-vgprs -O3 -filetype=asm \
  -o hstu-attn-bwd-no-extended-vgpr.s hstu-attn-bwd.ll
```

The Triton dump/override mechanism was used to substitute that assembly without
changing the Python kernel, launch arguments, or LLVM IR.

## Current output and hazard

The post-RA hazard pass leaves `v257` live through the WMMAs because it does
not overlap any explicit WMMA operand. The later VGPR-encoding pass then adds
the mutable `S_SET_VGPR_MSB` state needed to select registers above `v255`.
In the full kernel this produces hundreds of mode transitions while `v257`
remains live. On the tested gfx1250 A0 (`Device Rev 0x00`), lane 0 of the value
used from that register is corrupted before the first final-DV store. The
store itself is correctly lowered; it only makes the earlier corruption
observable as a real-address memory fault.

Schematically, current code generation permits:

```asm
s_set_vgpr_msb <select v257 as destination>
v_lshrrev_b32_e32 v1 /* v257 */, 1, v16
...
s_set_vgpr_msb <WMMA operand banks; previous mode in high byte>
v_wmma_f32_16x16x32_bf16 ...
... many WMMAs and mode changes ...
s_set_vgpr_msb <select v257 as source>
v_add_nc_u32_e32 v14, v10, v1 /* v257 */
global_store_b128 v[14:15], v[10:13], off
```

The MIR test checks the exact current output, including the four mode changes:

```text
S_SET_VGPR_MSB 64
$vgpr257 = V_LSHRREV_B32_e32 ...
S_SET_VGPR_MSB 16513
V_WMMA_F32_16X16X32_BF16_w32_threeaddr ...
S_SET_VGPR_MSB 33154
V_WMMA_F32_16X16X32_BF16_w32_threeaddr ...
S_SET_VGPR_MSB 33284
$vgpr0 = V_ADD_U32_e32 0, $vgpr257
```

This output is architecturally intended to preserve `v257`, but it is
hazardous on the tested A0: a value that later contributes to a global address
is allowed to depend on the correctness of mutable extended-VGPR selection
across a long sequence of WMMA mode changes. A single-lane corruption therefore
becomes an arbitrary-address store rather than merely a bad numerical result.

LLVM IR, TTIR, and TTGIR contain the correct address expression. LLVM AMDGPU
machine scheduling/register allocation is the first stage that creates this
long live range in an extended VGPR.

## Desired output

The desired output does not consume the pre-WMMA extended-VGPR value in the
address calculation. It rematerializes the lane offset after the last WMMA and
uses a short-lived low VGPR:

```asm
... final WMMA ...
v_mbcnt_lo_u32_b32 v64, -1, 0
v_and_b32_e32 v64, 16, v64
v_lshrrev_b32_e32 v64, 1, v64
...
v_add_nc_u32_e32 v14, v10, v64
global_store_b128 v[14:15], v[10:13], off
```

The clean LLIR override with this change passes 4,000 iterations twice. Fixed
cycle padding, `V_NOP`, and `s_wait_alu` experiments are not reliable fixes.

The complementary whole-function control compiles the identical original
LLIR with `-mattr=-1024-addressable-vgprs`. It emits no `S_SET_VGPR_MSB`, uses
256 VGPRs and 2,468 bytes of scratch, and passes 4,000 seed-4321 iterations.
The extended-VGPR build from the same LLIR faults at step 0. This is strong
evidence for the extended-VGPR mechanism, although the extra spilling means it
does not by itself identify which mode transition is defective.

## Fix recommendation

Treat this first as a gfx1250-A0 backend erratum, gated by a subtarget feature
until native B0 testing establishes the affected stepping range. In LLVM's
AMDGPU backend, teach register allocation/rematerialization (or a focused
post-RA erratum pass) not to carry cheap, address-critical work-item-derived
values in extended VGPRs across WMMA regions. For this kernel, rematerializing
the inexpensive lane expression after the final WMMA is both correct and
substantially cheaper than disabling extended VGPRs for the whole function.

A conservative correctness fallback is compiling with
`-mattr=-1024-addressable-vgprs`; it removes all `S_SET_VGPR_MSB` use at the
cost of heavy spilling and passed the 4,000-iteration runtime control. Do not
use the previously tested `s_nop 7` padding as a correctness fix: a
4,000-iteration run passed once but an identical repeat faulted at step 0.

## Validation matrix

| Variant | Result |
| --- | --- |
| Original N128, extended VGPRs, seed 4321 | Fault at step 0 |
| Identical original LLIR, `-mattr=-1024-addressable-vgprs` | PASS, 4,000 iterations |
| Late lane-offset rematerialization, clean LLIR replay | PASS, 4,000 iterations twice |
| N64 application workaround | PASS, reduced and full, 4,000 iterations |
| `s_nop 7` after every WMMA, identical repeat | Fault at step 0 |
| NOPs around every `S_SET_VGPR_MSB`, `V_NOP`, or `s_wait_alu` drains | Fault at step 0 |

Native cold-boot B0 validation is still required to distinguish an A0 silicon
erratum from a backend requirement that also applies to later steppings.
