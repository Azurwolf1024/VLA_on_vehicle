# 内环：位置伺服 vs 上游 TSID 力矩环

两个内环共享同一套上层接口（10 Hz 决策给出 TCP 目标），区别只在"怎么让 TCP 到位"。

| | `pos`（默认） | `tsid` |
|---|---|---|
| 执行器 | `<position>` 关节伺服 kp=2400 kv=60 | `<motor>` 纯力矩 |
| 频率 | 100 Hz（关节目标线性插值） | 500 Hz（每个物理步解一次 QP） |
| 动力学模型 | 无（只做逆解） | 上游 `Robot`：`M`、`H`、`J`、`J_B`、`τ_base` |
| 基座运动 | 目标在甲板系，随船走，不需要补偿 | 可给世界系目标，用 `η = J̇q̇ + J_B V̇_B + J̇_B V_B` 前馈 |
| 代价 | 2.7–2.9 s / 回合（CPU, sea 0.35） | 9.0–9.1 s / 回合（CPU） |
| 抓取偏差（sea 0.35） | 1.3 mm（deck）/ 0.6 mm（world） | 0.8 mm（两种系一致） |

实测：`python scripts/bench_control.py --episodes 3 --tsid-episodes 2`
（原始数据 `results/bench/control.json`，seed 200+，舷外任务）。
`tsid` 在两种参考系下抓取偏差相同（0.8 mm）—— 因为它本来就在**主动补偿基座运动**，
目标钉不钉在世界系对它影响不大；`pos` 则从 1.3 mm 掉到 0.6 mm，靠的是换参考系。

## 为什么要有两个

**采集/训练默认走 `pos`**：它便宜 4 倍，而且只要目标语义在甲板系里，
"把 TCP 送到那个点"这件事它直接就能做。

**`tsid` 存在的理由是基座运动前馈**。任务里有一段是世界系的 ——
垃圾漂在水面上，船一横摇它就相对甲板移动。`pos` 只能事后追误差，
`tsid` 用 `τ_base`（基座加速度）做前馈，提前给出反向力矩。

参考系本身是另一层开关（`env.frame`），和"用哪个内环"正交：

| `env.frame` | 零动作时目标 | 实测抓取偏差（sea 1.0） |
|---|---|---|
| `deck` | 随甲板动 | 2.6 mm |
| `world` | 钉在世界系 | **1.0 mm** |

完整的对照（含步数代价）由 `scripts/bench_frame.py` 产出，
数据在 `results/bench/frame_compare.json`。

## 船摇抑制：怎么量、量到多少

做法：保持 3 s 零动作，量 TCP 在**世界系**里的游走。
`pos/deck` 与 `tsid/deck` 测的是"目标跟着船走时 TCP 走多少"——这两者应当一致，
它们构成了"不补偿"的基线；`tsid/world` 是"目标钉在世界上"。

| 内环 / 参考系 | 平均游走 | 最大游走 |
|---|---|---|
| `pos` / deck | 12.3 mm | 26.1 mm |
| `tsid` / deck | 12.4 mm | 26.4 mm |
| `tsid` / world | **0.7 mm** | **1.0 mm** |

海况 `vehicle_scale=1.0`（与上游论文同口径，峰值线速度约 0.32 m/s）。

**读数注意**：这不是"TSID 比位置伺服好 18 倍"。`pos/deck` 与 `tsid/deck` 一样差，
是因为它们的**目标本身**在动 —— 换了参考系才有 18 倍。真正被验证的是：
上游的基座运动前馈在 MuJoCo 里确实成立，能把船摇对末端的影响压到毫米级。

## 复现

```bash
python scripts/bench_control.py --episodes 4              # 两张表都跑
python scripts/bench_control.py --skip-task               # 只跑船摇抑制
python scripts/view_scene.py --control-mode tsid --control-frame world   # 肉眼看
```

## 已知边界

1. **`world` 系跑完整任务是可行的，但更慢**。目标不随船走意味着机械臂要主动抵消船摇，
   sea 1.0 下单回合从 89 步涨到 113 步（平均）。默认仍用 `deck`。
   注：这条推翻了 v0.1 的说法（"world 系跑不了完整抓放"）—— 那时垃圾在甲板上随船动，
   目标不随船就必然漂走；现在垃圾在水面上，`world` 反而是**正确**的参考系。
2. **mocap 平台没有基座速度**（见 `architecture.md` §2.4）。上游模型里的
   `J_B·V_B` 项在仿真里对应不上，量级约毫米，比"基座自由落体"小两个数量级。
3. **末端估计器还没接**。仿真里直接把真值喂给控制器（`R_WB`、`p_B`、`ω`、`v`、`α`、`a`）。
   真机上游用的是 ESKF，噪声与滞后都会进来 —— 这是接真机前必须补的一环。
4. `Kp/Kd` 是重新调的（2000/80），不是论文 Table I 的 500/40，理由见 `architecture.md` §2.5。
