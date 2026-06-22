# L1 SHM 大页（THP）启用方案

> 适用：LMCache MP 模式下 server 进程的 L1 pinned SHM 池。
> 范围：Linux 5.15 / tmpfs / `/dev/shm` / POSIX `shm_open` + `mmap(MAP_SHARED)`。
> 不在范围：hugetlbfs / `MAP_HUGETLB` 路径，以及匿名 `MAP_PRIVATE | MAP_ANONYMOUS` 路径。

## 1. 背景与目标

XPU offload v2 设计要求 L1 pool 是 hugepage + pinned host memory（见
`xpu_offload_v2_design.md` §3.7、§10）。关键事实：

- L1 pool 必须**跨进程共享**（lmcache server 与 vLLM worker 同时 attach），
  因此底座只能是 POSIX SHM（`shm_open` + `mmap(MAP_SHARED, fd)`），不能用
  匿名 mmap。
- D2H/H2D 在 4 KiB 页 + pageable 上带宽极低；走 2 MiB 大页 + `host_register`
  是带宽下限的硬性条件。
- `MAP_HUGETLB` 改造跨进程协议（要 fd-passing）改动太大，**首选**让
  tmpfs/shmem THP 生效。

## 2. 关键差别：tmpfs THP vs 匿名 THP

`bench_huge_page.py`（不要参考）走的是**匿名 mmap**，匿名 THP 在 5.15 上稳定，
首触一次就直接给 2 MiB 页。但跨进程必须走 SHM，所以这条路不能直接搬过来。

LMCache 这边需要的是：`shm_open + ftruncate + mmap(MAP_SHARED, fd)`
路径下让 tmpfs/shmem 走 PMD 映射。

## 3. 离线验证方法

脚本：`/ssd2/zhaoguochun/ds4/probe_shm_thp_variants.py`

构造 5 个 variant，**全部通过 `multiprocessing.shared_memory.SharedMemory` 创建**
（这就是 `python_ops_fallback.alloc_shm_pinned_ptr` 用的同一原语），区别仅在：

| Variant | `prctl(PR_SET_THP_DISABLE, 0)` | `madvise(MADV_HUGEPAGE)` | first-touch 方式 | 结果（512 MiB） |
|--------|:------------------------------:|:------------------------:|------------------|------------------|
| A baseline | 否 | 是 | 不写 | `Rss=0, ShmemPmdMapped=0` |
| B 当前 LMCache fallback | 否 | 是 | 2 MiB stride 1 字节 | `Rss=1024 KiB, ShmemPmdMapped=0` ❌ |
| C memset 无 prctl | 否 | 是 | `memset(addr, 0, size)` | `Rss=size, ShmemPmdMapped=0` ❌ |
| D **prctl + memset** | 是 | 是 | `memset(addr, 0, size)` | `Rss=size, ShmemPmdMapped=size` ✅ |
| E prctl + stride | 是 | 是 | 2 MiB stride 1 字节 | `Rss=size, ShmemPmdMapped=size` ✅ |

判定指纹：**`ShmemPmdMapped == Rss == Size`**。

### 关键 take-away

1. **必须 `prctl(PR_SET_THP_DISABLE, 0)`**：进程可能从 systemd / docker / 父
   shell 继承 `THP_enabled=0`（看 `/proc/<pid>/status` 里的 `THP_enabled`），
   此时 `madvise(MADV_HUGEPAGE)` 表面上返回 0、`VmFlags` 也带上 `hg`，但
   fault 路径仍走 4 KiB。**这是过去观察到的失效根因**：lmcache server 段
   `ShmemPmdMapped=0` 而 bench_shm_hugepage_register.py 同进程能拿到大页，
   差别就在 bench 显式调用了 prctl。
2. **stride 1 字节**在 tmpfs/shmem 路径上**也能拿到大页**，前提是 prctl 已开
   （variant E）。但 stride 模式延迟更难控（每页一次 minor fault），生产路径
   首选 D（一次 `memset` 走完）。
3. madvise 必须有，仅 prctl 不够：5.15 tmpfs 默认走"懒路径"，
   `madvise(MADV_HUGEPAGE)` 是把 vma 标 `hg` 让 fault 走同步 huge 分配的
   触发条件。

## 4. 系统前置条件

启动 lmcache server 之前必须满足全部：

```bash
# 4.1 内核 THP 总开关（mode 至少是 madvise；always 也行）
cat /sys/kernel/mm/transparent_hugepage/enabled
# 期望：always 或 [madvise]

# 4.2 tmpfs THP 总开关
cat /sys/kernel/mm/transparent_hugepage/shmem_enabled
# 期望：[always] 或包含 advise

# 4.3 /dev/shm mount 选项
mount | grep '/dev/shm'
# 期望含 huge=always

# 4.4 进程级 THP（运行时检查；启动脚本无法事先校验，由代码兜底）
cat /proc/<lmcache-server-pid>/status | grep THP_enabled
# 期望：1（代码会主动调 prctl 把它修成 1）
```

启动脚本里用的 `setup_lmcache_shm_hugepages`（DSV4 server.sh）做了
`echo always > .../shmem_enabled` 和 `mount -o remount,huge=always /dev/shm`，
对应 4.1（仅 shmem 部分）和 4.3。**4.4 必须由进程内代码 prctl 修，不能在脚本里
解决**——`PR_SET_THP_DISABLE` 不会跨 exec 反向继承。

## 5. 代码侧实现

入口：`lmcache/python_ops_fallback.py:_advise_thp_and_first_touch`

```python
def _advise_thp_and_first_touch(ptr: int, size: int) -> bool:
    libc = _get_libc()
    if libc is None or size == 0:
        return False
    # 1) 进程级反向解禁 THP（关键：抵消父进程继承的 THP_DISABLE=1）
    libc.prctl(_PR_SET_THP_DISABLE, 0, 0, 0, 0)
    # 2) vma 级 THP 偏好
    rc = libc.madvise(ptr, size, _MADV_HUGEPAGE)
    if rc != 0:
        warnings.warn(...)
        return False
    # 3) 一次性全段 memset：让 tmpfs fault 路径同步分配 2 MiB 页
    ctypes.memset(ptr, 0, size)
    return True
```

调用链：
```
lmcache server CLI:  --l1-use-hugepages
   ↓
L1MemoryManagerConfig.use_hugepages = True
   ↓
create_memory_allocator → MixedMemoryAllocator(..., use_hugepages=True)
   ↓
_resolve_pinned_alloc_free → alloc_shm_pinned_ptr(size, name, use_hugepages=True)
   ↓ (XPU 走 fallback；CUDA 仍走 c_ops 2-arg ABI)
python_ops_fallback.alloc_shm_pinned_ptr → _advise_thp_and_first_touch
```

CUDA C++ pybind 仍是 2-arg `(size, shm_name)`，`_resolve_pinned_alloc_free`
里只在 `use_hugepages=True` 时才把第三个参数透传出去，保住旧 ABI。

## 6. 启动时验证

### 6.1 快速一行判断

```bash
# 找到 lmcache server PID
pid=$(pgrep -f "lmcache server" | head -1)

# SHM 是否用了大页——ShmemPmdMapped > 0 即生效
grep -A20 "/dev/shm/lmcache_l1_pool_xpu_" /proc/$pid/smaps \
  | grep -E 'ShmemPmdMapped|THPeligible'
```

期望输出：
```
ShmemPmdMapped: 134217728 kB     # ★ 等于 Size 即全部 PMD 映射
THPeligible:    1                 # VMA 符合 THP 条件
```

### 6.2 完整 smaps 段查看

```bash
grep -A25 "/dev/shm/lmcache_l1_pool_xpu_" /proc/$pid/smaps
```

期望：
```
Size:           134217728 kB     # = 128 GiB
KernelPageSize:        4 kB      # tmpfs 此字段恒 4 kB，不影响物理页判断
Rss:            134217728 kB     # 全段 fault-in
ShmemPmdMapped: 134217728 kB     # ★ 等于 Size 即全部走 2 MiB 大页
THPeligible:             1      # VMA 符合 THP 条件
VmFlags: rd wr sh mr mw me ms sd hg
```

`ShmemPmdMapped == Size` 是大页生效的硬证据。如果 `ShmemPmdMapped == 0` 但
`hg` 标志在，去查 `/proc/<pid>/status` 的 `THP_enabled`：若是 0 说明 prctl 路径
没跑（代码版本不对，或者 `_get_libc()` 没拿到）。

### 6.3 各查看方式对比

不同工具能看到的大页信息不同，容易误判：

| 查看方式 | 能看到大页指标？ | 说明 |
|----------|:---:|------|
| `top` / `htop` | 否 | 只看到 SHR（= RssShmem），与是否大页无关 |
| `/proc/<pid>/statm` | 否 | 只有页面计数，无大页信息 |
| `/proc/<pid>/status` | 间接 | `VmPTE` 很小（几 MB）→ 大页生效；`HugetlbPages` 只统计静态大页 |
| `/proc/<pid>/smaps_rollup` | 是 | 可看 `ShmemPmdMapped`、`AnonHugePages`（进程汇总） |
| **`/proc/<pid>/smaps`** | **是** | **唯一能看每个 VMA 的 ShmemPmdMapped + THPeligible 的方式** |
| `/proc/meminfo` | 是（系统级） | `ShmemHugePages` / `ShmemPmdMapped` 是系统汇总，非进程级 |

### 6.4 大页指标含义与常见误判

**PMD = Page Middle Directory**，Linux 三级页表的中间层。4 KiB 小页需要 PMD →
PTE → 4 KiB page，2 MiB 大页直接 PMD → 2 MiB page（跳过 PTE 层）。

`ShmemPmdMapped` = 共享内存中通过 PMD 直接映射为 2 MiB 大页的量。

| 指标 | 值 = 0 是否说明没大页？ | 说明 |
|------|:---:|------|
| `AnonHugePages` | **否** | 只统计匿名页（heap/mmap），SHM 大页不计入此项 |
| `HugetlbPages` (status) | **否** | 只统计 hugetlbfs 静态大页（`MAP_HUGETLB`），THP 不计入 |
| `Shared/Private_Hugetlb` (smaps) | **否** | 同上，hugetlbfs 专用，THP 走 ShmemPmdMapped |
| `KernelPageSize: 4 kB` | **否** | tmpfs VMA 此字段恒 4 kB，是虚拟页面大小，不影响物理页 |
| **`ShmemPmdMapped`** | **是** | ★ SHM 透明大页的唯一可靠指标 |
| **`THPeligible: 1`** | **否**（=0 才说明不符合） | VMA 是否符合 THP 条件 |

三种大页机制对应不同指标：

| 大页类型 | 对应 smaps 指标 | 本方案是否使用 |
|----------|----------------|---------------|
| THP 透明大页（tmpfs/shmem） | `ShmemPmdMapped` | **是，128 GB** |
| THP 透明大页（匿名页） | `AnonHugePages` | 否（L1 pool 是 SHM 不是匿名页） |
| hugetlbfs 静态大页 | `Private/Shared_Hugetlb` | 否（需要 `MAP_HUGETLB`，跨进程改动大） |

### 6.5 间接验证：VmPTE

```bash
grep VmPTE /proc/$pid/status
```

- 大页生效：`VmPTE` 仅几 MB（128 GB 只需 ~6.5 万个 PMD 条目）
- 4 KiB 小页：`VmPTE` 达数百 MB（128 GB 需要 ~3300 万个 PTE 条目）

## 7. 失败回退语义

`_advise_thp_and_first_touch` 返回 False 不再 raise。三条失败路径全部回退到
4 KiB 页，服务可用：

| 触发条件 | 行为 |
|----------|------|
| libc 加载失败 | 直接 return False，不 madvise、不 prctl、不 memset |
| madvise 返回非 0 | 发 RuntimeWarning（含 errno），return False |
| size == 0 | return False |

回退时上层 `xpu_transfer._ensure_shm_registered` 仍会按 4 GiB 段做
`cudaHostRegister/xpu_host_register`，只是 pin 的是 4 KiB 页，TLB 压力大，
带宽显著下降。这种情况要靠监控（`ShmemPmdMapped`）告警，代码本身不会因为
"没拿到大页"就停止启动。

## 8. 已知限制 / 后续可选项

- **本方案只对 tmpfs THP**。如果要 100% 确定性（不依赖 shmem 路径所有勾选条件），
  改 `MAP_HUGETLB` 是终极方案：跨进程协议要从 "命名 SHM + shm_open" 换成
  "fd-passing"（unix domain socket SCM_RIGHTS / pidfd_getfd）。改动量大于
  本次。
- **kernel ≥ 6.1** 起 tmpfs/shmem 的 PMD 路径更稳定，prctl + memset 不变；
  迁移升级时不用动这块代码。
- **memset 全段时长**：128 GiB 段 memset 大概 10s 量级；此时段在启动期，与
  原本就要做的 `host_register` 串行，**不影响热路径**。

## 9. 相关文件 / 命令索引

| 资源 | 路径 |
|------|------|
| fallback 实现 | `lmcache/python_ops_fallback.py:_advise_thp_and_first_touch` |
| 配置字段 | `lmcache/v1/distributed/config.py:L1MemoryManagerConfig.use_hugepages` |
| CLI 透传 | `lmcache server --l1-use-hugepages` |
| ABI 兼容 | `lmcache/v1/memory_management.py:_resolve_pinned_alloc_free` |
| 启动脚本 | `xvllm/.../DeepSeek-V4-Flash-INT8_MTP_fast_attn_TP8_server.sh` |
| 离线验证脚本 | `/ssd2/zhaoguochun/ds4/probe_shm_thp_variants.py` |
| 现成的同进程对照 | `/ssd2/zhaoguochun/ds4/bench_shm_hugepage_register.py` |
