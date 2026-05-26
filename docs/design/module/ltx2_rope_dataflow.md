# LTX-2.3 3D RoPE Data Flow

本文档梳理 LTX-2.3 video transformer 中 3D RoPE 从输入坐标到最终作用在 Q/K 上的完整数据流,包含每一步的 input / output tensor 及 shape 变化。

相关代码:
- `prepare_video_coords`: [ltx2_transformer.py:1196](../../../vllm_omni/diffusion/models/ltx2/ltx2_transformer.py#L1196)
- `LTX2AudioVideoRotaryPosEmbed.forward`: [ltx2_transformer.py:1326](../../../vllm_omni/diffusion/models/ltx2/ltx2_transformer.py#L1326)
- `apply_interleaved_rotary_emb`: [ltx2_transformer.py:68](../../../vllm_omni/diffusion/models/ltx2/ltx2_transformer.py#L68)
- `_slice_rope_for_tp`: [ltx2_transformer.py:411](../../../vllm_omni/diffusion/models/ltx2/ltx2_transformer.py#L411)
- Pipeline 调用点: [pipeline_ltx2_3.py:828-840](../../../vllm_omni/diffusion/models/ltx2/pipeline_ltx2_3.py#L828-L840)

---

## 0. 前置假设(贯穿全文)

为方便追踪 shape,本文档使用以下具体参数(对应 LTX-2.3 默认配置):

| 参数 | 值 | 含义 |
|---|---|---|
| `B` | `1` | batch size |
| `num_frames, height, width` | `4, 16, 16` (latent) | latent 视频形状 |
| `N_patches` | `4 * 16 * 16 = 1024` | 摊平后的 token 数 |
| `num_attention_heads` | `32` | head 数 |
| `head_dim` | `128` | 每 head 维度 |
| `dim` | `4096` | `= num_heads * head_dim` |
| `num_pos_dims` | `3` | video 有 3 个时空轴 |
| `num_rope_elems` | `6` | `= num_pos_dims * 2`(每轴 cos+sin) |
| `scale_factors` | `(8, 32, 32)` | VAE (T, H, W) 下采样率 |
| `base_positions` | `(20s, 2048px, 2048px)` | 归一化基准 |
| `theta` | `10000` | RoPE 频率底 |
| `fps` | `24` | 视频帧率 |

---

## Stage A: `prepare_video_coords` —— 物理坐标生成

**目的**:为每个 latent token 计算它在**原视频**中的 `[start, end)` 时空区间,时间用**秒**,空间用**像素**。

### 数据流

| 步骤 | 操作 | shape | 含义 |
|---|---|---|---|
| A1 | `grid_f, grid_h, grid_w = arange(...)` | `[4]`, `[16]`, `[16]` | 三个轴的 latent 索引 |
| A2 | `meshgrid` + `stack` | `[3, 4, 16, 16]` | 每点存 (f, h, w) 起始 latent 坐标 |
| A3 | `stack([grid, grid+1], -1)` | `[3, 4, 16, 16, 2]` | 每点的 `[start, end)` 边界 |
| A4 | `flatten(1, 3)` + `unsqueeze(0).repeat(B,...)` | `[1, 3, 1024, 2]` | 摊平成 token 序列 |
| A5 | `× (8, 32, 32)` (per-axis) | `[1, 3, 1024, 2]` | latent → pixel 坐标 |
| A6 | time 轴 `+causal_offset - scale_factors[0]` + `clamp(min=0)` | `[1, 3, 1024, 2]` | causal VAE 首帧修正 |
| A7 | `pixel_coords[:, 0] /= fps` | `[1, 3, 1024, 2]` | time 轴变成**秒** |

### 输出

```
video_coords.shape = [B, 3, N_patches, 2]   = [1, 3, 1024, 2]
```

四个轴的含义:

| 轴 | 大小 | 含义 |
|---|---|---|
| 0 | `B` | batch index |
| 1 | `3` | `0=frame(秒), 1=height(像素), 2=width(像素)` |
| 2 | `N_patches` | token index |
| 3 | `2` | `[start, end)` |

> Audio coords 由 `prepare_audio_coords` 类似生成,shape 为 `[B, 1, N_audio_patches, 2]`,只有时间维(也是秒)。

---

## Stage B: `LTX2AudioVideoRotaryPosEmbed.forward` —— 坐标 → (cos, sin)

**目的**:把每个 token 的物理坐标转成 RoPE 旋转矩阵的 `(cos, sin)` 张量,形状与 Q/K 对齐。

### B1. 区间取中点

```python
coords_start, coords_end = coords.chunk(2, dim=-1)  # 各 [B, 3, N, 1]
coords = ((coords_start + coords_end) / 2).squeeze(-1)
```

| 输入 | shape |
|---|---|
| `coords` | `[1, 3, 1024, 2]` |

| 输出 | shape | 含义 |
|---|---|---|
| `coords` | `[1, 3, 1024]` | 每 token 在 (t秒, y像素, x像素) 上的单点位置 |

### B2. 归一化 + 维度重排

```python
grid = torch.stack(
    [coords[:, i] / max_positions[i] for i in range(num_pos_dims)],
    dim=-1
)
```

| 输出 | shape | 含义 |
|---|---|---|
| `grid` | `[1, 1024, 3]` | 每 token 在 3 个轴上的归一化位置 `[0, 1]` |

- 除以 `(20, 2048, 2048)` 让 RoPE **对分辨率/时长解耦**。
- `stack(..., dim=-1)` 把 "轴维度" 从 axis 1 挪到最后,方便后续广播频率向量。

### B3. 频率序列

```python
pow_indices = theta ** linspace(0, 1, steps=dim // num_rope_elems)  # [682]
freqs_base  = pow_indices * pi / 2                                  # [682]
```

| 输出 | shape | 含义 |
|---|---|---|
| `freqs_base` | `[4096 // 6 = 682]` | log-spaced 的 682 个频率(~1.57 → ~15708) |

- `dim // 6`:head 通道切成 6 份(3 轴 × cos/sin 占一对)。
- log-spaced 让低频(全局)和高频(局部)都被均匀覆盖。

### B4. 位置 × 频率(外积)

```python
freqs = (grid.unsqueeze(-1) * 2 - 1) * freqs_base
```

| 输入 | shape |
|---|---|
| `grid.unsqueeze(-1)` | `[1, 1024, 3, 1]` |
| `freqs_base` | `[682]` |

| 输出 | shape | 含义 |
|---|---|---|
| `freqs` | `[1, 1024, 3, 682]` | 每 token、每轴、每频率对应一个相位 |

`(grid * 2 - 1)` 把 `[0, 1]` 拉到 `[-1, 1]`,让相位关于画面中心对称。

### B5. 三轴交错拼接

```python
freqs = freqs.transpose(-1, -2).flatten(2)
```

| 输出 | shape | 含义 |
|---|---|---|
| `freqs` | `[1, 1024, 3 * 682 = 2046]` | 通道排布 `[t_f0, h_f0, w_f0, t_f1, h_f1, w_f1, ...]` |

三轴**交错**而不是分块拼接,让每个 head 内的通道都同时编码 time / height / width 三种位置信息。

注意 `2046 < dim / 2 = 2048`,差 2 个通道由后面的 padding 补。

### B6. cos / sin + padding + repeat_interleave(`interleaved` 模式)

```python
cos_freqs = freqs.cos().repeat_interleave(2, dim=-1)  # [1, 1024, 4092]
sin_freqs = freqs.sin().repeat_interleave(2, dim=-1)

# 余数通道 padding:cos=1, sin=0(不旋转)
if dim % num_rope_elems != 0:
    cos_padding = ones_like(cos_freqs[..., : dim % num_rope_elems])    # [1, 1024, 4]
    sin_padding = zeros_like(cos_freqs[..., : dim % num_rope_elems])
    cos_freqs = cat([cos_padding, cos_freqs], -1)                       # [1, 1024, 4096]
    sin_freqs = cat([sin_padding, sin_freqs], -1)
```

| 输出 | shape | 含义 |
|---|---|---|
| `cos_freqs`, `sin_freqs` | `[1, 1024, 4096]` | 与 Q/K 最后一维对齐 |

`repeat_interleave(2, -1)` 把每个频率值复制成相邻两个通道,让"每对相邻通道共享同一个 θ"在内存里自动对齐(对应 Stage D 的复数旋转布局)。

### `split` 模式(可选,见 [ltx2_transformer.py:1377](../../../vllm_omni/diffusion/models/ltx2/ltx2_transformer.py#L1377))

不做 `repeat_interleave`,而是 reshape 成 `[B, H, N, dim / (2H)]`,布局是 "前 r 通道一组,后 r 通道一组",与 `apply_split_rotary_emb` 对应。

---

## Stage C: TP 切分(可选)

`_slice_rope_for_tp` ([ltx2_transformer.py:411](../../../vllm_omni/diffusion/models/ltx2/ltx2_transformer.py#L411))

| RoPE 类型 | 输入 shape | 输出 shape(TP=`tp_size`) | 切分维 |
|---|---|---|---|
| `interleaved` | `[B, N, dim]` | `[B, N, dim / tp_size]` | 最后一维 |
| `split` | `[B, H, N, r]` | `[B, H / tp_size, N, r]` | head 维 |

SP 切分由 `_build_sp_plan` 配置([ltx2_transformer.py:1442](../../../vllm_omni/diffusion/models/ltx2/ltx2_transformer.py#L1442)),沿 token 维(N)切。

---

## Stage D: 应用到 Q / K

### D1. Q / K 的形状(self-attention)

来自 `to_qkv` 投影 + split + RMSNorm,在 [ltx2_transformer.py:478-479](../../../vllm_omni/diffusion/models/ltx2/ltx2_transformer.py#L478-L479):

| Tensor | shape (TP=1) |
|---|---|
| `query` | `[1, 1024, 4096]` |
| `key` | `[1, 1024, 4096]` |

### D2. `apply_interleaved_rotary_emb` ([ltx2_transformer.py:68](../../../vllm_omni/diffusion/models/ltx2/ltx2_transformer.py#L68))

```python
x_real, x_imag = x.unflatten(2, (-1, 2)).unbind(-1)
x_rotated      = stack([-x_imag, x_real], -1).flatten(2)
out            = x * cos + x_rotated * sin
```

| 步骤 | tensor | shape | 含义 |
|---|---|---|---|
| 拆对 | `x.unflatten(2, (-1, 2))` | `[1, 1024, 2048, 2]` | 把 4096 看成 2048 对,每对 2 元素 |
| 拆对 | `x_real, x_imag` | 各 `[1, 1024, 2048]` | 实部 / 虚部 |
| 旋转伙伴 | `x_rotated` | `[1, 1024, 4096]` | `[-b_0, a_0, -b_1, a_1, ...]` |
| 输出 | `out` | `[1, 1024, 4096]` | 旋转后的 Q 或 K |

**数学等价**:每对通道 `(a, b)` 视为复数 `z = a + bi`,乘上 `e^{iθ} = cos θ + i sin θ`,得到:
```
a' = a cos θ - b sin θ
b' = a sin θ + b cos θ
```

逐通道验证(前 4 个通道):

| 通道 | x | cos | x_rotated | sin | out |
|---|---|---|---|---|---|
| 0 | a_0 | c_0 | -b_0 | s_0 | a_0·c_0 - b_0·s_0 |
| 1 | b_0 | c_0 |  a_0 | s_0 | b_0·c_0 + a_0·s_0 |
| 2 | a_1 | c_1 | -b_1 | s_1 | a_1·c_1 - b_1·s_1 |
| 3 | b_1 | c_1 |  a_1 | s_1 | b_1·c_1 + a_1·s_1 |

与复数乘法公式完全一致。

### D3. reshape 进 attention backend

在 `__call__` 中([ltx2_transformer.py:494](../../../vllm_omni/diffusion/models/ltx2/ltx2_transformer.py#L494)):

```python
query = query.unflatten(2, (heads, -1))  # [B, N, H, D]
key   = key.unflatten(2, (heads, -1))
value = value.unflatten(2, (heads, -1))
hidden_states = attn(query, key, value, ...)
```

| Tensor | shape |
|---|---|
| `query`, `key`, `value` | `[1, 1024, 32, 128]` |

进入 attention backend 做 `softmax(QK^T) V`。

---

## 总数据流图

```
prepare_video_coords
  latent grid (3 axes)            [3, 4, 16, 16]
  → patch [start, end)             [3, 4, 16, 16, 2]
  → flatten + batch                [B, 3, 1024, 2]
  → ×scale + causal + /fps         [B, 3, 1024, 2]    ── 物理坐标 (秒, 像素, 像素)
                                      │
                                      ▼
rope.forward
  midpoint                         [B, 3, 1024]
  / max + stack(-1)                [B, 1024, 3]       ── 归一化 grid
  freqs = θ ^ linspace · π/2       [682]              ── log-spaced 频率
  grid · freqs (外积)               [B, 1024, 3, 682]  ── 相位
  transpose + flatten              [B, 1024, 2046]    ── 三轴交错
  cos/sin + pad + interleave       [B, 1024, 4096]    ── 最终 (cos, sin)
                                      │
                                      ▼
attention
  query / key                      [B, 1024, 4096]
  apply_interleaved_rotary         [B, 1024, 4096]    ── Q/K 被旋转
  unflatten heads                  [B, 1024, 32, 128]
  scaled-dot-product attn          [B, 1024, 32, 128]
```

---

## Shape 速查表

| 阶段 | tensor | shape | 备注 |
|---|---|---|---|
| pipeline → transformer | `video_coords` | `[B, 3, N, 2]` | `prepare_video_coords` 输出 |
| rope 取中点 | `coords` | `[B, 3, N]` | `(start + end) / 2` |
| 归一化 | `grid` | `[B, N, 3]` | `/ max + stack(-1)` |
| 频率基 | `freqs_base` | `[dim / 6]` | `θ ^ linspace · π/2` |
| 相位 | `freqs` | `[B, N, 3, dim / 6]` | 外积 |
| 三轴拼接 | `freqs` | `[B, N, 3 * (dim / 6)]` | `transpose(-1, -2).flatten(2)` |
| RoPE 输出(interleaved) | `(cos, sin)` | `[B, N, dim]` | pad + interleave |
| RoPE 输出(split) | `(cos, sin)` | `[B, H, N, dim / (2H)]` | reshape |
| Q / K 投影后 | `query`, `key` | `[B, N, dim]` | `to_qkv` + split |
| RoPE 应用后 reshape | `query`, `key` | `[B, N, H, head_dim]` | `unflatten(2, (H, -1))` |

---

## 设计要点

1. **物理坐标(秒/像素)而非 token 索引**:让 RoPE 对分辨率、帧率、时长解耦,模型可以推理任意尺寸的视频。
2. **首帧 causal 修正**:`+ causal_offset - scale_factors[0]` + `clamp(min=0)` 处理 causal VAE 首帧 stride=1 的特性。
3. **三轴交错拼接**:让单个 head 同时感知 time / height / width,而不是某些 head 只看时间维。
4. **`repeat_interleave(2, -1)`** 是 interleaved 布局的关键 —— 让每对相邻通道共享同一个 `(cos, sin)`,正好对应"复数旋转"语义。
5. **Audio 共享秒级时间轴**:`prepare_audio_coords` 把 mel 帧通过 `hop_length / sampling_rate` 也换算成秒,使 a2v / v2a cross-attention 中 video 和 audio token 能按"同一时刻"对齐。

---

## 一句话总结

> RoPE 把 `[B, 3, N, 2]` 的物理坐标,经过 **"中点 → 归一化 → 外积频率 → 三轴交错 → cos/sin"** 五步,变成 `[B, N, dim]` 的 (cos, sin) 张量,与 Q/K 同形,通过复数旋转嵌入位置信息,让 attention 的内积只依赖**相对时空距离**。
