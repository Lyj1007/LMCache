# XPU KV Offload v2 设计（精简版）

> 本文档替代 `xpu_offloaded_d2d_d2h_design.md`。基线代码状态为 commit
> `4dce383a`（DSV4 per-group + connector-side SWA-suffix trim 完成，但
> `transfer_context/xpu_*.py` / `modules/xpu_transfer.py` / `xpu_cuda_compat/`
> 尚未引入）。`4dce383a` 之后的 19k 行实现仅作问题分析参考。

## 1. 目标与非目标

### 1.1 目标

- 让 XPU 上 KV cache 的 store / retrieve **可与推理重叠**，且**不阻塞**推理主循环
  超过 1ms。
- 把当前实现 **代码量缩减 ~70%**（从 ~9000 行到 ~2800 行）、**跨进程同步原语**
  从 4 类（设备 READY、设备 DONE、SHM 状态机、跨进程 flock）压到 1 类（普通
  SHM MQ + ACK）。
- 在 MLA 模型上，**TP 间冗余 KV 只走一次 D2H/H2D**，剩余 rank 通过 XCCL
  broadcast 拿到数据。
- D2H/H2D 路径全程使用 hugepage + pinned host memory。

### 1.2 非目标

- 不追求"forward 主线程零开销 submit + 跨进程 device flag 0 同步"。当前
  实现为这个目标付出了 ring buffer + per-slot READY/DONE + ABA/generation
  + circuit breaker + watchdog 的复杂度，新设计接受 forward 线程几微秒
  的 `event.record()` + `queue.put()` 开销，把 `event.synchronize()` 推到
  **worker 进程内**的 BG 线程上完成。
- 不复用 CUDA `HandleTransferContext` 的 IPC handle 路径。XPU 平坦物理地址
  空间直接使用 `data_ptr()`。
- 不支持非 MLA、非 hybrid（DSV4-Flash）以外的模型变体新特性。
- 不引入新的 C++ 算子。仅复用 xvllm 已有的
  `gather_multi_layer_block_kv_transfer` / `scatter_multi_layer_block_kv_transfer`
  / XCCL TP broadcast。

## 2. 硬件与负载约束

新设计的所有取舍都是从下列**硬件 + 负载**约束推出来的，先列在前面，后面
任何复杂度都要回到这张表上自证必要性：

| 约束 | 含义 | 对设计的影响 |
|---|---|---|
| XPU 平坦物理地址空间 | 进程 B 知道进程 A 的 `data_ptr()` 即可在 kernel 内跨进程读写 A 的设备内存 | 不需要 IPC handle / `_share_cuda_()`；跨进程 D2D 直接复用现有 gather/scatter 算子 |
| 设备 event 不能跨进程 | `xpuIpcEventGetHandle` 不可用 | 跨进程同步只能选 (a) SHM CPU 标志 (b) 设备内存标志 + kernel 读。新设计选 SHM MQ + ACK 这一类，**不再用 (b)** |
| `memcpy` / `.item()` / `.cpu()` 不能操作跨进程设备地址 | 跨进程必须经 kernel | 决定 server 端两阶段：跨进程 D2D gather kernel → 本地 D2H copy_ |
| H2D / D2H 在 pageable memory 上带宽极低 | 无 hugepage pinned 即不可用 | L1 pool **必须**是 hugepage + `cudaHostRegister`；控制面 SHM 不需要 |
| 同卡 TP 间互联（XCCL）带宽 ≫ 跨进程 PCIe D2D | XCCL 直接走 NVLink/类似互联 | MLA 多 rank 重复数据**只在 1~2 个 rank 上 H2D**，其它 rank 通过 XCCL `broadcast` 拿；自造跨进程 device-ptr 复制语义比 collective 弱 |
| MLA group 在 TP 间 KV 完全冗余 | TP-replicated；非 MLA group TP-sharded | store 时只让 TP0 发；retrieve 时只让 TP0（或 TP0/TP7 轮转）做 H2D + scatter，其余 rank 走 broadcast |
| forward 主线程同时驱动 attention kernel launch + connector hook | connector hook 在 forward 主线程上调 | 任何 `stream.synchronize()` 直接放主线程 = 推迟下一 step 的 kernel launch；必须挪到 BG 线程 |
| vLLM v1 connector 已经把 store / retrieve 设计为 future-based | future 可以在主线程 attempt 阶段 register、step 末尾 wait | submit 阶段不要 sync，可以"submit 立即返回 future"，只要在 wait 入口正确等到 |

## 3. 设计思路（为什么是这套方案）

按"问题 → 取舍 → 决策"的顺序梳理一遍——每条都对应到 §2 的具体约束。

### 3.1 不让 forward 主线程做 sync

vLLM v1 connector hook 与 attention kernel launch **同线程**。上一版当前实现
为了避免在主线程上 `stream.synchronize()`，引入了"跨进程 device READY flag +
B 端 kernel 轮询"——但**进程内**的 `event.synchronize()` 完全可以绕开这套
复杂度：

```
forward 主线程 (亚毫秒):
  event = Event(); event.record(compute_stream);
  queue.put((event, key, block_ids, future)); return future

worker BG 线程 (一条或多条, 跟 forward 解耦):
  event.synchronize();   ← 进程内 PyTorch 一等 API, 稳, 不需要跨进程协议
  mq.send_recv(...);
  future.set_result(...)
```

这个改动一举去掉 ring buffer + READY flag + ABA/generation 三大块（~1700 行）。

### 3.2 store 的"future 立即返回"是异步语义

`save_kv_layer` / `wait_for_save` 是 vLLM v1 连接器原生 future 模型。submit 时
立即返回 future，下一 step 入口或 `wait_for_save` 才统一 wait——BG 线程在这
段间隔里早把 MQ ACK 收完了。所以**不需要**自造"零阻塞 submit"特殊路径，
只需要让 future resolution 走 BG 线程而不是 forward 主线程。

### 3.3 retrieve 在高并发下也走同一异步队列

retrieve 必须等 KV 就绪才能继续推理，看起来"必然阻塞"。但**阻塞点是
forward 主线程在 `future.result()` 那一刻**，不是 submit 那一刻：

- 多个 retrieve 请求可以**同时**入 BG 队列 → BG 线程并行（多条）跟 server
  握手 → server 端单 stream pipeline 串行 H2D + scatter → 多个 future 在
  forward 主线程入口被一次 wait 收掉。
- worker 端 BG 线程数 N 可调（默认 2~4），server 端单 retrieve_stream 即可
  保证 `local_device_buffer` 的覆盖顺序。
- 这样高并发场景下 retrieve 跟 storage 查找、跨实例请求自然 pipeline，
  forward 主线程仍然只在 step 入口 wait 一次。

### 3.4 跨进程同步只保留一种：SHM MQ + ACK

把所有跨进程同步压到"普通 SHM MQ 上的 request / ACK"这一类后：

- 所有"标志位"消失：READY / DONE / slot_state / generation / error_code
  都不再需要跨进程语义。
- 所有"轮询 + 兜底"消失：DONE 轮询、watchdog、circuit breaker、ring lock
  超时全部退化为 MQ recv timeout 这一种失败模式。
- server 端单 stream FIFO 由 stream 自身保证，不再靠跨进程 generation
  防 ABA。

代价是 worker 必须接受"提交一次 store ~一次 MQ recv 等 ACK 才能复用 block"
的延迟——这个延迟本来就被 BG 线程吞掉了，forward 主线程感受不到。

### 3.5 MLA TP 路由 = 一次 D2H/H2D + XCCL broadcast

MLA group 在 TP 间 KV 完全相同。让所有 TP rank 都各自做一遍 D2H/H2D 浪费 N×
H2D 带宽。两个观察：

- **store**：只 TP0 发 store 即可，其余 rank 完全不参与 LMCache 路径。
- **retrieve**：只 TP0（或 TP0/TP7 chunk 奇偶轮转）做 server H2D + scatter，
  其余 rank 在 vLLM XCCL TP group 上 `broadcast(src=0)` 接收。XCCL 卡间带宽
  远高于 PCIe，且**已经是 vLLM 的标准依赖**——不需要自造跨进程 compact
  device buffer + publish-wait + cleanup 的山寨 broadcast。

这一条决定了 §7 的路由表：MLA / 非 MLA group 走完全不同的参与策略。

### 3.6 server 端：单 stream FIFO + 异步 commit/ACK

server 端不再每条请求自带 generation / DONE flag——同一条 store_stream 上
的 FIFO 顺序就是同步语义。提交线程只在 stream 上 enqueue（不阻塞），独立
commit 线程 wait `event` 后再调 storage_manager + 回 ACK。retrieve 同理：
独立 retrieve_stream + 独立 ack 线程。

如果未来确实要 throughput，可以**只**加 store_stream/retrieve_stream 数量
（按 instance 分片），不引入跨进程同步。

### 3.7 hugepage pinned SHM 必须，但只对 L1 pool

H2D/D2H pageable 带宽极低，所以 L1 pool **必须**是 hugepage + 4 GiB 段
`cudaHostRegister`（已验证可行，已有实现可继续用）。控制面 SHM（MQ、
metadata）不申请 THP / host-register，避免吃 pinned 预算。

### 3.8 复用现有算子，不写新 kernel

跨进程 D2D 通过 `__cuda_array_interface__` 把 `data_ptr()` 包成 PyTorch
tensor，直接传入 xvllm 已有的
`gather_multi_layer_block_kv_transfer` / `scatter_multi_layer_block_kv_transfer`。
TP broadcast 直接调 `vllm.distributed.parallel_state.get_tp_group().broadcast()`。
**整个新方案不引入新的 C++ 算子**——这是把代码量压回 ~2800 行的关键。

## 4. 设计原则（精简）

由 §3 的思路抽出可执行检查清单：

1. **TP 路由先于传输优化**：MLA group 只在 TP0（可选 TP0/TP7 分担）做
   D2H/H2D，其余 rank 走 XCCL broadcast。
2. **Sync 全部在 worker 进程内的 BG 线程做，forward 主线程不阻塞**。store /
   retrieve 共用一份 BG 线程 + future 模型；不使用跨进程 device flag。
3. **跨卡通信用 XCCL**：retrieve broadcast 复用 vLLM 已有 TP collective，
   不自造。
4. **D2H/H2D 只走 hugepage pinned SHM**；控制面 SHM 不做 host-register。
5. **算子复用**：跨进程 D2D 复用 xvllm gather/scatter；不新增 kernel。
6. **跨进程同步只有一种**：SHM MQ 上的 request / ACK 报文。任何新增"标志
   位 / 轮询 / 看门狗 / 熔断"都需要回到 §2 约束表证明必要性。

## 5. 当前方案的问题（要砍掉的复杂度）

§3.1–§3.5 解释了为什么这些机制不需要；下表把它们对应到具体代码：

| 项 | 现状 | 在新方案里被什么替代 |
|---|---|---|
| Per-slot 设备 READY flag | A 计算流上 `fill_(generation)`；B 用 `torch.add(remote_flag,0)` kernel 跨进程读 | worker BG 线程 `event.synchronize()`（§3.1） |
| Per-slot 设备 / SHM DONE flag | B 写 SHM 或 device kernel；A 轮询 + watchdog + circuit breaker | future + MQ ACK（§3.2 / §3.4） |
| 跨进程 flock + ring 状态机 | EMPTY → FILLING → READY → IN_PROGRESS → DONE/ERROR + 跨进程原子 | 不存在；server 端单 stream FIFO 自带顺序（§3.6） |
| Rank0 → peers 自定义 broadcast | compact device buffer + 跨进程 device-ptr 复制 + publish-wait + cleanup | XCCL `broadcast(src=0)`（§3.5） |
| async D2H/H2D pipeline + stream-ordered DONE | 单 stream FIFO + pinned SHM 8B token DONE | server 端 commit / ack 线程 wait `event` 后回 ACK（§3.6） |

合并起来：**跨进程同步原语数量**是真正的复杂度来源。新设计只保留**一类**——
普通 SHM MQ 上的 request / ACK 报文。

## 6. 架构总览

```
                推理进程 worker（每个 TP rank 一个）
                ┌─────────────────────────────────────┐
                │  vLLM forward                        │
                │     │                                │
                │     ├─ MLA group? ───YES───>┐        │
                │     │                       │        │
                │     │   tp_rank == 0?       │        │
                │     │     ├─ YES ─> store / retrieve+broadcast(src=0)
                │     │     └─ NO  ─> store: skip      │
                │     │              retrieve: wait on broadcast
                │     │                                │
                │     └─ non-MLA group ─> per-rank store / retrieve（不变）
                └─────────────────────────────────────┘
                                │
                       (worker 仅在参与时与 server 交互)
                                │
                                ▼
                ┌─────────────────────────────────────┐
                │     LMCache server （单进程）         │
                │  ┌──────────────────────────────┐   │
                │  │ XpuTransferModule             │   │
                │  │  - per (instance_id) 注册     │   │
                │  │  - D2D gather (跨进程 kernel) │   │
                │  │  - D2H 到 hugepage pinned SHM │   │
                │  │  - 反向 retrieve 路径相同     │   │
                │  └──────────────────────────────┘   │
                └─────────────────────────────────────┘
```

不存在独立"transfer helper 进程"——LMCache server 单进程就足够；store 期间的
跨进程 D2D 是 server 进程通过 worker 暴露的 `data_ptr()` + 自身 stream 直接发起。

## 7. TP 路由策略

### 7.1 注册阶段（worker → server）

每个 worker 注册时把 `(tp_rank, tp_size, group_views, mla_per_group)` 上报给
server：

```python
class RegisterXpuContextPayload:
    instance_id: int
    tp_rank: int
    tp_size: int
    groups: list[GroupRegister]   # per engine group

class GroupRegister:
    engine_group_id: int
    is_mla: bool
    layer_indices: list[int]
    layer_data_ptrs: list[int]
    layer_shape: tuple[int, ...]   # 单层 shape
    layer_dtype: str
    block_size: int
    storage_blocks_per_chunk: int  # connector 算好的
```

### 7.2 路由表（connector 决定哪些 rank 发请求）

| group 类型 | store 发起 rank | retrieve 发起 rank | 其它 rank |
|---|---|---|---|
| MLA（KV TP-replicated） | 仅 TP0 | TP0；可选 TP7 接管偶数/奇数 chunk | 等 XCCL broadcast 把 KV scatter 后的目标段广播到本 rank |
| 非 MLA（KV TP-sharded） | 每个 rank 各自发 | 每个 rank 各自发 | 不参与 broadcast |

实现位置：`lmcache/integration/vllm/lmcache_mp_connector.py` 的 `store` /
`retrieve` 入口，按 `group_views[*].is_mla` 和 `tp_rank` 短路。
非参与 rank 仍要保留 connector 级别的 future 占位，以便 vLLM scheduler 端
统一等待。

### 7.3 TP0/TP7 分担（可选第二阶段）

第一阶段所有 MLA chunk 都让 TP0 出。第二阶段在 prompt 长度超过阈值时把
**chunk** 按奇偶轮转给 TP0 / TP7：

```
chunk 0 -> TP0 H2D + scatter，broadcast(src=0)
chunk 1 -> TP7 H2D + scatter，broadcast(src=7)
chunk 2 -> TP0 ...
```

理由：按 chunk 切粒度自然，不需要在 layer 维度切（layer 维切会让两个 H2D
共用同一段 hugepage SHM，需要额外协调）。门槛由 prompt token 数（或 chunk
数 ≥ 4）触发，单 chunk 请求继续走 TP0-only 避免 broadcast src 切换。

## 8. Worker → Server 协议

### 8.1 线程模型：forward 主线程 + worker BG 线程 + MQ

vLLM v1 connector hook（`save_kv_layer` / `wait_for_save` / retrieve hook）
和 attention kernel launch 在同一 worker **forward 主线程**上调用。在主线程
上 `compute_stream.synchronize()` 会让下一 step 的 kernel launch 推迟，等同于
推理阻塞。新方案让 forward 线程**只做亚毫秒动作**，sync 推到 worker 进程内的
专用 BG 线程。

```
forward 主线程（不阻塞）:
  1. event = torch_dev.Event(); event.record(compute_stream)   # 微秒级 enqueue
  2. submit_queue.put((op, event, key, block_ids, future))    # 进程内 queue
  3. return future                                              # 立即返回

worker BG 线程（store/retrieve 各 1 条；高并发可调大）:
  1. op, event, key, block_ids, future = submit_queue.get()
  2. event.synchronize()                                        # ← sync 在 BG，
                                                                #   不阻塞 forward
  3. resp = mq_client.send_recv(op, key, block_ids, ...)        # 同步 MQ
  4. future.set_result(resp.ok)
```

要点：

- `event.synchronize()` 是**进程内** PyTorch API，不需要跨进程 device flag。
- store / retrieve **共用一份**异步队列 + future 模型，差别只在调用方何时
  `future.result()`：
  - **store**：connector 不等待，下一 step 入口才统一 `wait_for_save()`，
    BG 线程在此期间已经把请求发出 + ACK 收回。
  - **retrieve**：connector 在 forward 真正需要 KV 之前 wait——高并发下
    多条请求可同时排队在 BG 线程 / server 上 pipeline，wait 只是收尾。
- 队列消费严格 FIFO；如果实测 retrieve 单 BG 线程吃不下高并发，可启动 N 条
  retrieve BG 线程（每条只跟 server 串行交互），不影响协议。
- BG 线程中 MQ recv 超时 → `future.set_exception()`，调用方按现有路径回退。

### 8.2 worker 端 `XPUDevicePtrTransferContext` 接口

只暴露四个方法，每个都**不阻塞 forward 线程**：

1. `register()`：发 `REGISTER_XPU_KV_CACHE`（§7.1 payload），等 ACK 拿到
   l1_shm_name + pool_size，attach pinned hugepage SHM。同步阻塞 OK
   （注册只在启动时一次）。
2. `submit_store(key, instance_id, block_ids)`：
   - `event.record(compute_stream)`，入 store 队列，立即返回 future。
   - BG 线程负责 sync + 发 MQ + 收 ACK + resolve future。
3. `submit_retrieve(key, instance_id, block_ids, skip_blocks_per_group)`：
   - `event.record(compute_stream)`（保证目标 KV 区域之前的写入已 enqueue），
     入 retrieve 队列，立即返回 future。
   - BG 线程负责 sync + 发 MQ + 收 ACK + resolve future。
   - **broadcast 由调用方在 future resolve 后单独触发**（§9.3），broadcast
     本身在 forward 主线程上调（XCCL 是流式 collective，本身不阻塞 host），
     不再放进 BG 线程，避免再做一次跨线程 stream 切换。
4. `close()`：drain 两条队列、stop BG 线程、解绑 SHM、发 `UNREGISTER`。

### 8.3 删除的内容（相对当前实现）

- `XpuRingBuffer` 整个类（1235 行）
- `XpuReadyFlag` + `preallocate_read_buffers`（482 行）
- `_pending_futures` / `_request_slots` / `_slot_submit_time` / `_slot_generation`
  / `_slot_request_id`（worker 端 ABA / generation 协议）
- `_store_consecutive_failures` / `_store_cb_open_until`（circuit breaker）
- `_RING_LOCK_TIMEOUT` / `_DONE_POLL_INTERVAL` / `_STORE_DONE_TIMEOUT`
  / `_STORE_CB_*`（共 7 个 env-tunable）
- server 端的 `_RetrieveBroadcastChunk` / `_retrieve_broadcast_chunks` /
  `_retrieve_broadcast_cleanup_timer` / `_copy_or_receive_retrieve_broadcast`
  / `_release_retrieve_broadcast_chunk`（~600 行）

### 8.4 同步语义对照表

| 阶段　　　　　　　　　　 | 当前　　　　　　　　　　　　　　　　　　　　　　　　| 新方案　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |
| --------------------------| -----------------------------------------------------| ------------------------------------------------------------------------------------------------|
| store 之前 KV 写入完成　 | 跨进程 device READY flag + B kernel 轮询　　　　　　| worker BG 线程 `event.synchronize()`（进程内，不阻塞 forward）　　　　　　　　　　　　　　　　 |
| store 完成可以复用 block | A 轮询跨进程 DONE flag + watchdog + CB　　　　　　　| future `.result()` 等 MQ ACK　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |
| retrieve KV 就绪　　　　 | 跨进程 SHM DONE token + 单 stream pipeline　　　　　| future `.result()` 等 MQ ACK；高并发下多条 BG 线程排队　　　　　　　　　　　　　　　　　　　　 |
| 高并发 retrieve　　　　　| server 端 single-stream rolling window + DONE token | worker 端 N 条 BG 线程 + future；server 端可继续保留 single-stream rolling window 作为内部优化 |
| MLA TP 间数据一致　　　　| rank0 写 compact device buffer + peers 跨进程拷贝　 | XCCL `broadcast(src=0)`　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|

## 9. Server 端 XpuTransferModule

### 9.1 register

- 收到 `REGISTER_XPU_KV_CACHE`，按 `(instance_id, tp_rank)` 建表。
- 用 `__cuda_array_interface__` 包出每个 layer 的远端 tensor 视图（`data_ptr()`
  + shape + dtype），缓存在 `XpuInstanceEntry`。
- 启动时一次性把 L1 SHM pool 按 4 GiB 段 `cudaHostRegister`（hugepage
  pinned），失败回滚 + 服务退出。
- 预分配 server 本地 `local_device_buffer`（按 peak chunk × max blocks_per_chunk
  × max group hidden_dim）。

### 9.2 store 路径（异步 pipeline）

```
server 内部线程模型：
  - MQ 接收线程：解析请求 → 入 store_queue
  - 主提交线程：在单条 store_stream 上顺序提交所有 D2D gather + D2H
  - commit 线程：消费 store_done_queue，调 storage_manager 落 L1，回 ACK

server 处理一次 STORE 请求：
  1. (MQ 线程) 接到请求 → 申请 hugepage pinned SHM slot → 入 store_queue
  2. (主提交线程) 取 queue 头：
       - gather_multi_layer_block_kv_transfer(remote_layer_tensors,
              server_local_device_buffer, block_ids, ...)   on store_stream
       - server_local_device_buffer.copy_(pinned_shm_slot)    on store_stream
       - 在 store_stream 上 record event_done
       - 把 (event_done, key, slot_id, future_ack) 入 store_done_queue
  3. (commit 线程) event_done.synchronize() → storage_manager.commit
       → MQ 回 ACK to worker
```

要点：

- **单条 store_stream + FIFO**：所有 store 请求按到达顺序提交在同一 stream
  上，stream 自身保证 D2H 顺序、`local_device_buffer` 不会被下一次 gather 覆盖
  即将 D2H 的数据。无需 ring buffer / per-slot READY/DONE。
- **server 提交线程不在 stream 上同步**：commit 线程独立 wait `event_done`，
  不阻塞下一个请求的 gather/H2D enqueue。这是 server 端 throughput 的关键。
- 多 instance（多 vLLM）混部时，可按 instance 拆 store_stream，依然不引入
  跨进程 flag。

### 9.3 retrieve 路径（异步 pipeline，高并发）

```
server 处理一次 RETRIEVE 请求 (TP0 only for MLA)：
  1. (MQ 线程) 接到请求 → 入 retrieve_queue
  2. (lookup 线程池) 并行 storage_manager.read_prefetched_results(obj_keys)
       命中后入 retrieve_h2d_queue（按 request_id 携带原始 future）
  3. (h2d 提交线程) 取 queue 头，在单条 retrieve-stream 上：
       for each chunk:
         pinned_shm.copy_(server_local_device_buffer)        # H2D
         scatter_multi_layer_block_kv_transfer(
             remote_layer_tensors,                          # 跨进程写 TP0 KV
             server_local_device_buffer,
             block_ids, skip_blocks_per_group, ...)
       record event_scatter_done
       入 retrieve_done_queue
  4. (ack 线程) event_scatter_done.synchronize() → MQ ACK to worker (TP0)

worker (TP0) BG 线程收到 ACK：
  - future.set_result(True)（forward 主线程稍后 wait）
    注意：future = "TP0 KV 就绪"；非 TP0 ranks 必须独立在同一 hook 点
    调 broadcast，不依赖此 future。

worker forward 主线程在真正需要 KV 之前（§9.4 详述）：
  TP0:
    retrieve_future.result()
    gather_multi_layer_block_kv_transfer(
        TP0 KV cache → TP0 broadcast_buffer, block_ids)   # 进程内
    broadcast(broadcast_buffer, src=0)
    # TP0 KV 已写好，不再 scatter
  非 TP0:
    broadcast(broadcast_buffer, src=0)                    # 接收
    scatter_multi_layer_block_kv_transfer(
        own broadcast_buffer → own KV cache, block_ids)   # 进程内
```

要点：

- server 端 retrieve 走**独立 stream**，与 store_stream 并行；H2D / scatter 串
  在同 stream 内保证 `server_local_device_buffer` 不被覆盖。
- **lookup 与 H2D 解耦**：lookup（storage 命中查找）走线程池并发，H2D 走单条
  stream 串行。命中快的请求可以先到 retrieve_h2d_queue，命中慢的不阻塞它。
- **worker 高并发**：worker 端 retrieve BG 线程数 N 可调（默认 2~4），N 条线程
  各自走 MQ send_recv，不强求 FIFO（每个 future 自带 request_id）。
- 高并发下 N 个 retrieve future 在 forward 主线程入口被一次性 wait → 合并
  triggered broadcast 到下游 attn collective 路径。

### 9.4 broadcast 机制（XCCL 落地细节）

XCCL `broadcast()` 要求所有 rank 上同 shape 的**连续** tensor，而 vLLM KV
cache 是 paged 离散 block。所以**不能**直接对 KV 做 broadcast，必须借助
per-rank 的连续 broadcast_buffer：

```
每个 worker rank 预分配 broadcast_buffer = 1 个 peak chunk 大小（连续 device 内存）
server side 不动它（broadcast_buffer 是 worker process-local）

server (TP0 only):
  scatter chunk → TP0 KV cache（现有 cross-process scatter op）
  MQ ACK → TP0

TP0 worker (forward 主线程):
  retrieve_future.result()                                      # 等 server
  gather TP0 KV cache → broadcast_buffer (process-local op)     # +1× gather
  broadcast(broadcast_buffer, src=0)                            # XCCL collective

非 TP0 worker (forward 主线程, 同一 hook 点):
  broadcast(broadcast_buffer, src=0)                            # 接收
  scatter broadcast_buffer → own KV cache (process-local op)
```

收益核算（以 8 卡 TP，1 个 chunk 为单位）：

| 路径 | H2D 次数 | scatter 次数 | gather 次数 | broadcast |
|---|---:|---:|---:|---:|
| 当前实现（每 rank 各自 H2D） | 8 | 8 | 0 | 0 |
| 新实现 | 1 | 8 | 1 | 1 |

新实现把 H2D 减到 1×（瓶颈带宽 5 GB/s），多了 1× process-local gather +
1× XCCL broadcast（卡间 50 GB/s+），净收益显著。

**broadcast_buffer 尺寸**：等于 1 个 peak chunk 的 (num_blocks_per_chunk *
block_size * hidden_dim * num_layers * dtype_bytes)。在 DSV4-Flash MLA group
配置下约几 MB ~ 几十 MB 量级，可在 register 时按 `per_layer_storage_blocks_per_chunk`
最大值预分配。

**collective 调用时序**：所有 rank 必须在 connector 的同一 hook 点、按相同
chunk 顺序调 `broadcast()`。connector 由 vLLM scheduler 在所有 rank 上分发
相同的 retrieve plan（block_ids），保证调用次数和顺序一致。

**TP0/TP7 分担时**：`src` 按 chunk 奇偶切换。所有 rank 仍要按相同顺序参与
每一个 broadcast 调用——非 src rank 此 chunk 也要 broadcast(buffer, src=N)。

### 9.5 实现要点（broadcast）

- 复用 `vllm.distributed.parallel_state.get_tp_group().broadcast()`。
- `broadcast_buffer` 在 worker 注册时一次性分配，shape 按所有 group 中
  最大 chunk 上界（MLA group 主导）。
- TP0 worker 的 gather 与非 TP0 worker 的 scatter 都使用现有
  `gather_multi_layer_block_kv_transfer` /
  `scatter_multi_layer_block_kv_transfer` op，不引入新 kernel。
- 非 MLA group（SWA / state group）**完全不走** broadcast 路径——每个 rank
  独立 store / retrieve（路由表 §7.2 第 2 行）。

### 9.6 删除的内容（相对当前实现）

- 4239 行 → 目标 ~1200 行
- 不再有 ring buffer 轮询线程、READY flag preallocate、broadcast cleanup timer、
  per-slot watchdog、circuit breaker

## 10. Hugepage Pinned SHM

完全沿用当前实现里**已经验证过的**部分（`xpu_cuda_compat/mem_alloc.py` 中的
4 GiB 段 host-register）：

1. L1 pool 创建为 POSIX SHM + `mmap(MAP_SHARED)`
2. `madvise(MADV_HUGEPAGE)` + 范围预 fault
3. 按 4 GiB 段 `cudaHostRegister`，初始化期间一次性完成
4. 失败回滚已注册段 + 启动失败

控制面 SHM（MQ buffer 等）**不**做 hugepage / host-register。

## 11. 失败处理

| 失败类型 | 旧方案 | 新方案 |
|---|---|---|
| server 死锁 / OOM | worker 端 watchdog + circuit breaker，避免 worker 跟着 hang | server 死则 MQ recv 超时；worker `submit_store` 收到 timeout → 抛出，由上层 connector 兜底回退到 sync 路径或失败 |
| ring lock 长时间持有 | flock + `_RING_LOCK_TIMEOUT=200ms` | 不存在 ring lock |
| ABA / 旧标志读到 | per-slot generation 协议 | 不存在跨进程 flag |
| broadcast 目标 rank 没注册 | publish-wait timeout + cleanup | XCCL 在 init 阶段已经握手；运行时 broadcast 不会找不到 peer |

worker 上 `submit_store` 不再需要 `_STORE_CB_FAILURE_THRESHOLD` 之类参数：
所有失败都是 MQ 同步语义下的"recv timeout"或"server 返回 error"。

## 12. 与已有代码 / 文档的关系

### 12.1 保留并继续使用

- `lmcache.v1.gpu_connector.utils`：`compute_kv_layout` / `is_mla` /
  `get_group_data_ptrs` / `LayoutHints`（不动）。
- `lmcache.v1.multiprocess.transfer_context.base.NonGpuContext` 抽象：
  CPU-only 路径继续用，XPU 路径不再继承它的 prepare/commit 两阶段（XPU 走
  自己的 `XPUDevicePtrTransferContext`）。
- `xvllm` 的 gather/scatter ops + XCCL TP group。
- `xpu_cuda_compat/mem_alloc.py` 中 hugepage host-register 部分。
- DSV4 per-group / SWA-suffix trim 协议（`per_layer_storage_blocks_per_chunk`
  / `skip_blocks_per_group`），完全保留。

### 12.2 删除

整个 `transfer_context/xpu_ring_buffer.py`、`transfer_context/xpu_flags.py`、
`modules/xpu_transfer.py` 中的 ring/flag/broadcast/cleanup/CB 部分。
保留并大幅瘦身 `transfer_context/xpu_device_ptr.py` 和
`modules/xpu_transfer.py` 主路径。

### 12.3 docs

- 本文档替代 `xpu_offloaded_d2d_d2h_design.md`（迁移完成后将旧文档移到
  `docs/design/_archive/v1/multiprocess/`，保留作历史参考）。
- 协议变更（`REGISTER_XPU_KV_CACHE` payload 加 `tp_rank/tp_size/groups[*]`）
  在 `protocols/engine.py` 注释里说明。

## 13. 落地步骤（建议拆分 PR）

| PR | 范围 | 行数预估 | 验证 |
|---|---|---:|---|
| 1 | 删除 ring/flag/broadcast 协议代码，把 XPU 路径短接到 NonGpuContext SHM 同步路径，只保留 hugepage register | -8000 | 已有同步路径回归不挂 |
| 2 | 引入新 `XPUDevicePtrTransferContext`（仅 store/retrieve 主路径，TP0-only MLA store） | +600 | DSV4-Flash MLA store 精度对齐 |
| 3 | 引入 server 端 D2D gather + D2H 路径 | +500 | 端到端 store 性能基线 |
| 4 | XCCL retrieve broadcast | +200 | request_long_question.sh 精度 |
| 5 | 可选：TP0/TP7 chunk 分担 | +150 | 长 prompt 性能 |
| 6 | 可选：server 端 prefetch ring（容量 2） | +100 | 高并发 store throughput |

每个 PR 单独可回归（用项目自带的
`/ssd2/zhaoguochun/ds4/request_long_question.sh` 做端到端验证），不强依赖
后续 PR；PR 1 到 PR 4 是必备路径。

## 14. 待确认事项

1. **TP7 是否真有 H2D 带宽收益？** 当前推测：单卡 H2D ~5 GB/s，TP0 一卡承担
   就饱和；XCCL 跨卡 ~50 GB/s 以上。所以 TP0/TP7 分担收益主要在让 TP7 的卡间
   带宽不闲置。需要在落地 PR 1-4 后实测决定是否做 PR 5。
2. **MLA group 在 DSV4-Flash 之外的模型上是否成立？** 当前实现 `is_mla`
   只看 `gpu_kv_format`；新设计沿用同一判定，不引入额外判断。
3. **server 在多 instance（多 vLLM 实例）下是否需要并发 D2H？** 当前
   `_LOCAL_DEVICE_BUFFER_PEAK_BLOCKS` 假设单 in-flight。如果未来多实例混部
   再引入 PR 6 的 prefetch ring。
