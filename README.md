# VLA on vehicle —— 载具上的视觉-语言-动作抓取

把"捡垃圾"这件事放到**会晃的甲板**上：一艘作业艇（或任何移动基座）载着一条 7 轴机械臂，
船头一支全局相机、腕部一支相机，模型看图 + 读指令，把指定颜色的垃圾抓起来丢进甲板上的
内置垃圾舱。

控制与机械臂模型**直接复用** [`robotic_arm_on_vehicle`](https://github.com/Azurwolf1024/robotic_arm_on_vehicle)
的 `veh_arm`（NERO 厂商 URDF、FK/雅可比、载具 6-DOF 运动引擎），本项目只新增
**仿真场景 / 双相机数据管线 / 策略接口**。

```
        ┌─ 决策层 10 Hz ──────────────────────────────────────┐
        │ 全局图(船头) + 腕部图 + q,dq + 指令 "把红色易拉罐…"   │
        │            ↓  VLA / BC / 脚本专家                    │
        │ 动作 = [dx,dy,dz,g]  甲板系 TCP 增量 + 夹爪           │
        └─────────────────────────┬──────────────────────────┘
                                  ↓
        ┌─ 内环 100 Hz ─ DLS 逆解（复用 veh_arm 的 J/FK）────────┐
                                  ↓
        ┌─ 物理 500 Hz ─ MuJoCo + 上游载具 6-DOF 运动写入 ───────┐
```

---

## 0. 当前状态（v0.1）

| 项 | 现状 |
|---|---|
| 场景 | MuJoCo：0.96×0.72 m 甲板 + 0.15×0.13×0.12 m 内置垃圾舱 + NERO 7-DOF + 舷墙 |
| 双相机 | 船头全局 160×160 fovy 46°，腕部侧装内倾 18° 160×160 fovy 80° |
| 垃圾 | 4 类（易拉罐 / 方块 / 纸箱 / 水瓶），每回合随机 3 件，形状与颜色互不相同 |
| 语言 | 中文 4 种句式 + 英文 3 种句式（字符级 token，48 长） |
| 动作 | 4 维：甲板系 TCP 增量（±2 cm/步）+ 夹爪 0/1 |
| 专家（上界） | **静止 8/8、sea 0.35 8/8、sea 1.0 8/8**（见 `results/bench/`） |
| 数据 | 160² 双相机 / 约 70 帧 → **2.1 MB 一条**（压缩 npz） |
| 策略 | 行为克隆基线（双相机 + 指令 + 本体感觉，~1.5 M 参数，CPU 可训）；20 条示范下闭环 **0/6**，见 §5 |
| 预训练 VLA | **只有接入位**（`policies/vla_adapter.py`），未接权重 —— 见 `docs/vla_interface.md` |
| 测试 | 13 条，含"MuJoCo FK 与上游 FK 逐点一致（1e-7）"和"完整抓放跑通" |
| 最大缺口 | 没有真机；单步 BC 在 20 条示范上过拟合到"时间规律"，闭环跑不通（§5） |

> **一个必须说清楚的结论**：v0.1 里**专家对海况不敏感**（三档都是 100%）。
> 原因是任务写在甲板系、内环位置伺服很硬，船的运动会通过关节目标自动抵消 ——
> 现阶段"载具"主要影响**感知**（画面在晃、物体在动），而不是控制。
> 要让动力学补偿真正起作用，需要把上游的 TSID 力矩环接进来并把任务改到世界系，
> 这是 v0.3 的目标（`docs/roadmap.md`）。

---

## 1. 快速开始

```bash
# 0) 依赖：需要上游控制库（本项目不复制它的代码）
git clone https://github.com/Azurwolf1024/robotic_arm_on_vehicle.git ../robotic_arm_on_vehicle
#   或: export VLA_VEHICLE_UPSTREAM=/path/to/robotic_arm_on_vehicle
#   或: python scripts/fetch_upstream.py      # 拉到 third_party/

pip install -r requirements.txt

# 1) 看一眼场景（无显示环境用 --headless 导出 PNG）
python scripts/view_scene.py
python scripts/view_scene.py --headless --out runs/scene.png

# 2) 采集示范
python scripts/record_demos.py --episodes 20 --out data/demos

# 3) 训练行为克隆
python scripts/train_bc.py --data data/demos --out runs/bc --epochs 30

# 4) 闭环评测（上界 / 下界 / 学到的策略）
python scripts/eval_policy.py --ckpt runs/bc/policy.pt --episodes 20 --seed0 1000
python scripts/eval_policy.py --policy expert --episodes 20 --seed0 1000
python scripts/eval_policy.py --policy random --episodes 10

# 5) 测试
python -m unittest discover -s tests -t .
```

三档工况（配置文件在 `configs/`）：

```bash
python scripts/eval_policy.py --policy expert --config configs/sea_calm.yaml   # 载具不动
python scripts/eval_policy.py --policy expert                                  # sea 0.35（默认）
python scripts/eval_policy.py --policy expert --config configs/sea_rough.yaml  # sea 1.0
```

---

## 2. 任务与场景

| 项 | 值 |
|---|---|
| 载具 | 上游 `VehicleMotion`：`sea` / `road` / `air` / `rail` / `static`，`scale` 调幅值 |
| 默认工况 | `sea, scale=0.35`（有义波高 ≈0.12 m，峰值线速度 ≈0.11 m/s） |
| 机械臂 | NERO 7-DOF，工具沿法兰法向 0.105 m，夹爪最大开口 84 mm |
| 垃圾舱 | 甲板开孔 + 船体开孔，真凹坑 0.12 m 深；**不是**摆在甲板上的箱子 |
| 到达判据 | 目标物中心进入舱内腔（各边内缩 5 mm）且 z < −0.02 |

垃圾撒布在**环形**可达域（r ∈ [0.31, 0.39] m）：工具必须竖直朝下时，太靠近基座的点
因肘关节限位解不出来（可达域表见 `docs/sim.md` §2，`tests/test_env.py` 会替你把关）。

---

## 3. 双相机

| | 全局（船头） | 腕部（掌侧） |
|---|---|---|
| 位置 | `(0.52, 0, 0.76)` 甲板系 | 工具系 `(0, −0.042, 0.008)`，掌部侧面 |
| 视场 | fovy 46°，覆盖臂 + 前方垃圾 + 右舷垃圾舱 | fovy 80°，朝工具轴内倾 18° |
| 语义 | 挑哪一件、往哪放 | 最后 2 cm 的对位与夹持 |
| 随动 | 固连船体 → **随船摇** | 固连腕部 → 随臂动 |

两支相机都是刚体相机，没有做任何稳定化 —— 画面里的地平线会随船倾斜，这是**数据分布的一部分**。

腕部相机最初装在两指正中间（同轴）。问题是：一旦夹住物体，物体距镜头只有 2 cm，
画面会被它糊满，策略丢掉"接下来往哪走"的视野。改成掌侧外挂 + 内倾后，
夹爪固定在画面下缘、前方始终可见。

![rollout](docs/assets/rollout.png)

*左：船头全局；右：腕部。自上而下为接近、下探、夹持、运输、投放。*

---

## 4. 数据和策略

```bash
python scripts/record_demos.py --episodes 50 --out data/demos
python scripts/train_bc.py --data data/demos --out runs/bc --epochs 30
```

一条回合存 `(全局图, 腕部图, q, dq, tcp, grip) → 动作(4)` 的 10 Hz 序列，
外加指令 token、成功标记、专家阶段名和一份环境参数快照。格式与体积见 `docs/data.md`。

策略只要满足 `policy.act(obs) -> (4,)` 就能闭环跑（`vla_vehicle/policies/base.py`）。
仓库里给了三个：脚本专家（上界）、行为克隆（可训基线）、随机（下界）。
预训练 VLA 的接入位是 `policies/vla_adapter.py`，`load()/infer()` **故意**抛
`NotImplementedError` —— 不做"看起来能跑其实没权重"的假实现。

---

## 5. 基线现状（先把话说清楚）

| 配置 | 训练侧 | 闭环成功率 |
|---|---|---|
| 专家（特权位姿，上界） | — | **24/24**（静止 / 0.35 / 1.0 各 8 条） |
| 行为克隆（20 条示范，15 epoch，96² 输入） | val L1 = 0.17，夹爪准确率 96% | **0/6** |

BC 的失败模式很具体：它**确实朝目标物体移动**，但在离物体 7.7 cm 处就闭合夹爪，
之后 76 步不再张开 —— 它拟合的是示范里的**时间规律**（大约第 14 步闭爪），
而不是"对准之后再闭爪"的视觉伺服关系。这是单步行为克隆的典型协变量漂移：
训练时只在专家轨迹上见过状态，一旦早了 7 cm，就进入了训练分布之外。

对应的修法（写进 v0.2）：action chunk、更大的视觉输入、DAgger / 更多示范。
**在做出 50%+ 的闭环成功率之前，任何"VLA 效果好"的说法都不成立** ——
所以这一版先交付链路，而不是分数。

---

## 6. 目录结构

```
VLA_on_vehicle/
├── vla_vehicle/
│   ├── deps.py              上游 veh_arm 的解析（安装/兄弟目录/third_party 三级回退）
│   ├── config.py            YAML → dataclass（写错键名直接报错）
│   ├── robot/
│   │   ├── arm.py           NERO 复用 + TCP/夹爪参数 + READY 位形
│   │   ├── mjcf.py          ArmSpec → MJCF（与上游同源，FK 逐点一致）
│   │   └── urdf_fix.py      ★ 修补上游把 URDF 惯量读成 0 的缺陷
│   ├── sim/
│   │   ├── scene.py         甲板/舱/船体/相机/垃圾装配
│   │   ├── objects.py       垃圾库 + 指令模板 + 字符级分词
│   │   └── env.py           载具驱动 + 三级时序 + 双相机观测 + 成功判据
│   ├── control/
│   │   ├── ik.py            DLS 逆解（基座系，带零空间偏好）
│   │   └── expert.py        脚本专家（分阶段 + 位移限幅）
│   ├── data/                回合结构 / 录制器 / 训练 Dataset
│   └── policies/            策略接口 / BC 基线 / 预训练 VLA 接入位
├── scripts/                 record_demos · train_bc · eval_policy · view_scene · fetch_upstream
├── configs/                 default · sea_calm · sea_rough · train_bc
├── tests/                   13 条：运动学一致性 / 环境契约 / 端到端 / 数据
├── results/bench/           专家成功率的三档基准（含每条回合明细）
└── docs/                    architecture · sim · data · vla_interface · roadmap
```

---

## 7. 上游依赖与一个上游缺陷

本项目**不复制** `veh_arm` 的代码，运行期从上游导入（`vla_vehicle/deps.py`）。
这样机械臂模型只有一个真相源，改限位/换 URDF 不会两边漂移。

顺带发现并修了一个上游缺陷：`veh_arm.robot.urdf_io._inertia_from_urdf` 用
`el.find("ixx")` 按**子元素**读惯量，而标准 URDF 把六个分量写成**属性**，
于是 NERO 七个连杆的转动惯量全部被读成 0（不报错）。
运动学完全不受影响，所以上游的跟踪精度结论依然成立；但任何依赖 `M(q)` 的动力学量、
以及 MuJoCo 的编译都会出问题（`inertia must have positive eigenvalues`）。
`vla_vehicle/robot/urdf_fix.py` 重新按属性读一遍补上，并有单测守着。

---

## 8. 已知限制

1. **抓取用运动学吸附**（`grasp_mode="attach"`）：闭合到位即把物体钉在工具上。
   纯接触摩擦在这个质量比下会持续滑移（实测 40–180 mm），诊断过程写在 `docs/sim.md` §5.2。
   释放后的下落、落点、碰撞仍是真物理。
2. **没有旋转动作**，抓取姿态恒为朝下。
3. **任务都在甲板系**，不做世界系定点（那才需要抵抗船摇，见 v0.3）。
4. **没有真机**；行为克隆的样本量还远不够给出有意义的成功率，现在只证明链路可通。
5. 垃圾初始朝向与夹爪开合方向对齐（±10°），否则超过 84 mm 开口就夹不住。

---

## 9. 许可

MIT，见 [LICENSE](LICENSE)。上游 `robotic_arm_on_vehicle` 的许可证见其仓库；
NERO 的 URDF 由上游自带，本项目不重新分发。
