# AOT ID: ['1_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import (
    grid,
    split_scan_grid,
    grid_combo_kernels,
    start_graph,
    end_graph,
    cooperative_reduction_grid,
)
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /home/ljp/llm/Diffulex/tmp/torchinductor_ljp/7b/c7bxzj3ifpibqo5zrqc7d6vfuw7j3ecwmowaowfw5xdpj7re257j.py
# Topologically Sorted Source Nodes: [cat, to_1], Original ATen: [aten.cat, aten._to_copy]
# Source node to ATen node mapping:
#   cat => cat
#   to_1 => convert_element_type_1
# Graph fragment:
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%sub, %add], -1), kwargs = {})
#   %convert_element_type_1 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%cat, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_cat_0 = async_compile.triton('triton_poi_fused__to_copy_cat_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_cat_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'E859E07B6B62AFA163C40CD3F96C46CB1FA6DF68D13FA0DF3E136DE6944A067E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_cat_0(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 229376
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 128)
    x4 = xindex // 128
    x2 = xindex // 3584
    x3 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (128*x4 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tl.load(in_ptr1 + (x2), tmp4, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.full([XBLOCK], 32768, tl.int32)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp7 < 0
    tmp11 = tl.where(tmp10, tmp9, tmp7)
    tl.device_assert(((0 <= tl.broadcast_to(tmp11, [XBLOCK])) & (tl.broadcast_to(tmp11, [XBLOCK]) < 32768)) | ~(tmp4), "index out of bounds: 0 <= tl.broadcast_to(tmp11, [XBLOCK]) < 32768")
    tmp13 = tl.load(in_ptr2 + (128*tmp11 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp14 = tmp6 * tmp13
    tmp15 = tl.load(in_ptr0 + (64 + 128*x4 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tl.load(in_ptr2 + (64 + 128*tmp11 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp16 * tmp17
    tmp19 = tmp14 - tmp18
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp4, tmp19, tmp20)
    tmp22 = tmp0 >= tmp3
    tmp23 = tl.full([1], 128, tl.int64)
    tmp24 = tmp0 < tmp23
    tmp25 = tl.load(in_ptr0 + (64 + 128*x4 + ((-64) + x0)), tmp22, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tl.load(in_ptr1 + (x2), tmp22, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.full([XBLOCK], 32768, tl.int32)
    tmp29 = tmp27 + tmp28
    tmp30 = tmp27 < 0
    tmp31 = tl.where(tmp30, tmp29, tmp27)
    tl.device_assert(((0 <= tl.broadcast_to(tmp31, [XBLOCK])) & (tl.broadcast_to(tmp31, [XBLOCK]) < 32768)) | ~(tmp22), "index out of bounds: 0 <= tl.broadcast_to(tmp31, [XBLOCK]) < 32768")
    tmp33 = tl.load(in_ptr2 + (128*tmp31 + ((-64) + x0)), tmp22, eviction_policy='evict_last', other=0.0)
    tmp34 = tmp26 * tmp33
    tmp35 = tl.load(in_ptr0 + (128*x4 + ((-64) + x0)), tmp22, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp36 = tmp35.to(tl.float32)
    tmp37 = tl.load(in_ptr2 + (64 + 128*tmp31 + ((-64) + x0)), tmp22, eviction_policy='evict_last', other=0.0)
    tmp38 = tmp36 * tmp37
    tmp39 = tmp34 + tmp38
    tmp40 = tl.full(tmp39.shape, 0.0, tmp39.dtype)
    tmp41 = tl.where(tmp22, tmp39, tmp40)
    tmp42 = tl.where(tmp4, tmp21, tmp41)
    tmp43 = tmp42.to(tl.float32)
    tl.store(out_ptr1 + (x3), tmp43, None)
''', device_str='cuda')


# kernel path: /home/ljp/llm/Diffulex/tmp/torchinductor_ljp/lz/clzptcyhundzqpgatwu5o26p52pvsrrnbkeuxelxilyppndu6kix.py
# Topologically Sorted Source Nodes: [cat_1, to_3], Original ATen: [aten.cat, aten._to_copy]
# Source node to ATen node mapping:
#   cat_1 => cat_1
#   to_3 => convert_element_type_3
# Graph fragment:
#   %cat_1 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%sub_1, %add_1], -1), kwargs = {})
#   %convert_element_type_3 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%cat_1, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_cat_1 = async_compile.triton('triton_poi_fused__to_copy_cat_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_cat_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'E859E07B6B62AFA163C40CD3F96C46CB1FA6DF68D13FA0DF3E136DE6944A067E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_cat_1(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 128)
    x4 = xindex // 128
    x2 = xindex // 512
    x3 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (128*x4 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tl.load(in_ptr1 + (x2), tmp4, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.full([XBLOCK], 32768, tl.int32)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp7 < 0
    tmp11 = tl.where(tmp10, tmp9, tmp7)
    tl.device_assert(((0 <= tl.broadcast_to(tmp11, [XBLOCK])) & (tl.broadcast_to(tmp11, [XBLOCK]) < 32768)) | ~(tmp4), "index out of bounds: 0 <= tl.broadcast_to(tmp11, [XBLOCK]) < 32768")
    tmp13 = tl.load(in_ptr2 + (128*tmp11 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp14 = tmp6 * tmp13
    tmp15 = tl.load(in_ptr0 + (64 + 128*x4 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tl.load(in_ptr2 + (64 + 128*tmp11 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp16 * tmp17
    tmp19 = tmp14 - tmp18
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp4, tmp19, tmp20)
    tmp22 = tmp0 >= tmp3
    tmp23 = tl.full([1], 128, tl.int64)
    tmp24 = tmp0 < tmp23
    tmp25 = tl.load(in_ptr0 + (64 + 128*x4 + ((-64) + x0)), tmp22, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tl.load(in_ptr1 + (x2), tmp22, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.full([XBLOCK], 32768, tl.int32)
    tmp29 = tmp27 + tmp28
    tmp30 = tmp27 < 0
    tmp31 = tl.where(tmp30, tmp29, tmp27)
    tl.device_assert(((0 <= tl.broadcast_to(tmp31, [XBLOCK])) & (tl.broadcast_to(tmp31, [XBLOCK]) < 32768)) | ~(tmp22), "index out of bounds: 0 <= tl.broadcast_to(tmp31, [XBLOCK]) < 32768")
    tmp33 = tl.load(in_ptr2 + (128*tmp31 + ((-64) + x0)), tmp22, eviction_policy='evict_last', other=0.0)
    tmp34 = tmp26 * tmp33
    tmp35 = tl.load(in_ptr0 + (128*x4 + ((-64) + x0)), tmp22, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp36 = tmp35.to(tl.float32)
    tmp37 = tl.load(in_ptr2 + (64 + 128*tmp31 + ((-64) + x0)), tmp22, eviction_policy='evict_last', other=0.0)
    tmp38 = tmp36 * tmp37
    tmp39 = tmp34 + tmp38
    tmp40 = tl.full(tmp39.shape, 0.0, tmp39.dtype)
    tmp41 = tl.where(tmp22, tmp39, tmp40)
    tmp42 = tl.where(tmp4, tmp21, tmp41)
    tmp43 = tmp42.to(tl.float32)
    tl.store(out_ptr1 + (x3), tmp43, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    assert_size_stride(arg0_1, (64, 3584), (3584, 1))
    assert_size_stride(arg1_1, (64, 512), (512, 1))
    assert_size_stride(arg2_1, (32768, 128), (128, 1))
    assert_size_stride(arg3_1, (64, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf1 = empty_strided_cuda((64, 28, 128), (3584, 128, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [cat, to_1], Original ATen: [aten.cat, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_cat_0.run(arg0_1, arg3_1, arg2_1, buf1, 229376, grid=grid(229376), stream=stream0)
        del arg0_1
        buf3 = empty_strided_cuda((64, 4, 128), (512, 128, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [cat_1, to_3], Original ATen: [aten.cat, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_cat_1.run(arg1_1, arg3_1, arg2_1, buf3, 32768, grid=grid(32768), stream=stream0)
        del arg1_1
        del arg2_1
        del arg3_1
    return (reinterpret_tensor(buf1, (64, 3584), (3584, 1), 0), reinterpret_tensor(buf3, (64, 512), (512, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((64, 3584), (3584, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1_1 = rand_strided((64, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg2_1 = rand_strided((32768, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.int64)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
