# 数据

## 1. 一条回合里有什么

```python
Episode(
  global_rgb (T,H,W,3) uint8   # 船头全局相机
  wrist_rgb  (T,H,W,3) uint8   # 腕部相机
  q  (T,7)  dq (T,7)           # 关节角/角速度（本体感觉）
  tcp (T,3)                    # 甲板系 TCP 位置
  grip (T,)                    # 归一化开口 0=闭 1=开
  action (T,4)                 # [dx,dy,dz,g]，前三维 ×action_scale 得米
  obj_track (T,N,3)            # 物体在甲板系的位置（特权，只用于评测/可视化）
  tokens (L,) int16            # 指令 token
  instruction: str, success: bool, target_index: int, seed: int, phase: [str]
  meta: dict                   # 环境参数快照（action_scale / dt / 海况 / bin 位置…）
)
```

写入 `<out>/ep0007.npz`（`np.savez_compressed`），另有一份 `index.jsonl` 便于快速筛选。
字段语义与版本号在 `vla_vehicle/data/schema.py`，改格式时**必须**同时改 `FORMAT_VERSION`。

体积参考：160×160 双相机、约 70 帧 → **2.1 MB/条**（压缩后）。100 条 ≈ 210 MB。

## 2. 采集

```bash
# 默认：3 件垃圾、sea scale=0.35、中文指令
python scripts/record_demos.py --episodes 50 --out data/demos

# 拆成几个域，方便做"跨域泛化"：
python scripts/record_demos.py --episodes 30 --out data/calm  --config configs/sea_calm.yaml
python scripts/record_demos.py --episodes 30 --out data/rough --config configs/sea_rough.yaml
python scripts/record_demos.py --episodes 30 --out data/en    --lang en
```

`seed` 连续递增，**训练/验证/测试必须用不重叠的 seed 段**：
同一 seed 的物体摆放完全一样，混用会让成功率虚高。

失败回合默认丢弃（`--save-failed` 可保留）。所有回合都会写进 `index.jsonl`，
包含 `success` 字段，方便统计专家本身的成功率。

## 3. 训练

```bash
python scripts/train_bc.py --data data/demos --out runs/bc --epochs 30
```

* 按**回合**划分训练/验证（`val_ratio`），不按帧 —— 相邻帧几乎相同，按帧划分等于泄漏。
* 图像在训练时下采样到 `image_size`（默认 96），节省显存；`BCPolicy` 里做同样的下采样，
  保证训练/推理预处理一致。
* 损失 = `L1(位置) + 0.5·BCE(夹爪)`；夹爪是二值，所以单独用分类损失。
* 归一化统计（q/dq/tcp 的均值方差）存在 ckpt 里，推理时自动加载 —— 不要手工算第二遍。

## 4. 评测

```bash
python scripts/eval_policy.py --ckpt runs/bc/policy.pt --episodes 20 --seed0 1000
python scripts/eval_policy.py --policy expert --episodes 20 --seed0 1000   # 上界参照
python scripts/eval_policy.py --policy random --episodes 10                # 下界参照
```

成功判据（`PickPlaceEnv.evaluate`）：**目标物体**中心落在垃圾舱内腔（各边内缩 5 mm）
且 z < −0.02（低于甲板面），并且此刻不在夹爪里。

判据只认"目标物体"，所以抓错物体、抓起来又掉在外面、丢到舱外都算失败。

## 5. 想扩大数据集时

* 现在一条回合只有一个目标物。多目标（"把三件都收进舱"）需要扩展专家为多回合状态机，
  数据格式不用改（`target_index` 改成序列即可）。
* 想加相机噪声/运动模糊，改 `scene.py` 的渲染参数或在 `Recorder.add` 里加增广 ——
  建议**离线增广**而不是在线，保证同一份数据可复现。
* 想做动作块（action chunk）训练：回合里已是完整序列，`EpisodeDataset` 换一个
  `_frames` 取 `t..t+k` 即可，不用重录。
