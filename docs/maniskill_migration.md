# ManiSkill → UniVTAC 任务迁移

## 概览

从 [ManiSkill](https://github.com/haosulab/ManiSkill) (Apache 2.0) 迁移了 7 个 tabletop gripper 任务，补充 UniVTAC 原有的 9 个任务，总计 16 个任务。

| # | UniVTAC 任务 | ManiSkill 来源 | 技能类型 | 抓取/操作模式 | 资产 | step_lim | 代码行数 |
|---|---|---|---|---|---|---|---|
| 1 | `stack_cube` | StackCube-v1 | 精密堆叠 | top-down grasp → lift → place on cubeB → release | Cube_Red + Cube_Green | 50 | 253 |
| 2 | `push_cube` | PushCube-v1 | 非抓取式前推 | partial-close(0.3) → behind cube → push +x | Cube_Red | 50 | 117 |
| 3 | `pull_cube` | PullCube-v1 | 非抓取式后拉 | partial-close(0.3) → far side → pull -x | Cube_Red | 50 | 113 |
| 4 | `poke_cube` | PokeCube-v1 | 工具中介推动 | grasp peg → peg head behind cube → poke +x | Cube_Red + Peg | 50 | 160 |
| 5 | `lift_peg_upright` | LiftPegUpright-v1 | 手内重定向 | grasp flat → gravity_rotate upright → lower → release | Peg | 50 | 143 |
| 6 | `roll_ball` | RollBall-v1 | 动态滚动 | partial-close → behind ball → push along target vector | Ball (r=35mm) | 80 | 122 |
| 7 | `place_sphere` | PlaceSphere-v1 | 容器放置 | grasp sphere → lift → move above bin → place → release | Sphere (r=20mm) + Bin | 50 | 183 |

**总代码**: 1,091 行

## 技能覆盖

```
UniVTAC 原有 9 个任务:
  grasp_classify      — 触觉材质分类 (unique, no ManiSkill equivalent)
  insert_hole         — 垂直 peg-in-hole
  insert_tube         — 倾斜管插入
  insert_HDMI         — 连接器插入
  lift_can            — 抓取 + 手内重定向
  lift_bottle         — 抓取 + 受限重定向
  put_bottle_in_shelf — 抓取 + 倾斜放置
  pull_out_key        — 提取 + 旋转
  collect             — 通用数据采集

新增 7 个任务 (来自 ManiSkill):
  stack_cube          — 精密堆叠
  push_cube           — 非抓取式前推
  pull_cube           — 非抓取式后拉
  poke_cube           — 工具中介推动
  lift_peg_upright    — 手内重定向（平→竖）
  roll_ball           — 动态滚动接触
  place_sphere        — 容器放置

─────────────────────────────────────────────
总计: 16 个任务
```

## 新增文件

### 资产 (`assets/objects/`)

| 文件 | 几何体 | 尺寸 | 用途 |
|---|---|---|---|
| `Cube_Red.usda` | 8-vert box mesh | half-size 0.02m (4cm) | stack_cube, push_cube, pull_cube, poke_cube |
| `Cube_Green.usda` | 8-vert box mesh | half-size 0.02m (4cm) | stack_cube |
| `Peg.usda` | 8-vert box mesh | 0.24m × 0.05m × 0.05m | poke_cube, lift_peg_upright |
| `Ball.usda` | USD Sphere | radius 0.035m | roll_ball |
| `Sphere.usda` | USD Sphere | radius 0.02m | place_sphere |
| `Bin.usda` | 8-vert box mesh | 0.05m × 0.05m × 0.01m | place_sphere |

### 任务定义 (`envs/`)

| 文件 | 行数 |
|---|---|
| `stack_cube.py` | 253 |
| `push_cube.py` | 117 |
| `pull_cube.py` | 113 |
| `poke_cube.py` | 160 |
| `lift_peg_upright.py` | 143 |
| `roll_ball.py` | 122 |
| `place_sphere.py` | 183 |

### Instructions (`instructions/`)

7 个 JSON 文件，每个包含 `seen` 和 `unseen` 指令文本（用于 language-conditioned policy）。

## 修改文件

| 文件 | 改动 |
|---|---|
| `envs/_base_task.py` | 添加 `_restore_primvar_color` static method（tet-mesh 生成后恢复物体颜色） |
| `policy/task_settings.json` | 添加 7 个新任务的 camera_type/downsample 条目（17 个任务） |
| `third_party/MANISKILL_ATTRIBUTION.md` | 记录所有 7 个派生任务的 Apache 2.0 来源 |

## 技术对齐

### 抓取模式

所有 prehensile 任务使用统一模式（对齐 `insert_hole`）：

```python
cpose = self._robot_manager.build_grasp_pose(
    target_pose.p,
    approach_direction=np.array([0, 0, 1]),    # top-down
    object_x_axis=np.array([1, 0, 0]),
)
self.move(self.atom.grasp_actor(
    actor, contact_point_id=cid,
    pre_dis=0.0, dis=0.0, is_close=True,       # adaptive grasp during approach
))
self.origin_inhand_pose = self._robot_manager.get_inhand_pose(actor)
```

- `build_grasp_pose` 自动处理 ZX hand 的高度补偿和开口方向
- `is_close=True` 在 approach 过程中触觉自适应夹紧
- `origin_inhand_pose` 在夹紧后立即记录

非抓取式任务（push_cube, pull_cube, roll_ball）使用 partial-close 形成推面：

```python
self.move(self.atom.close_gripper(0.3))  # 30% close = ~5.5cm gap for 4cm cube
```

### 坐标框架

所有 ManiSkill 坐标通过 `plate_x_offset` 映射到 UniVTAC 世界帧：

- ManiSkill table origin (0, 0, 0) → UniVTAC plate (0.5, 0, 0)
- `_target_xy` 公式包含此 offset
- Expert demo waypoints 使用 cube/sphere 实际世界坐标，不重复加 offset

### Success Criteria

完全对齐 ManiSkill `evaluate()` 逻辑：

| 元素 | ManiSkill 原始 | UniVTAC 实现 |
|---|---|---|
| stack_cube on-top xy | `norm(offset[:2]) <= sqrt(2)*H + 0.005` | 同 |
| stack_cube static | `is_static(lin=0.01, ang=0.5)` | frame-diff + 3-frame hysteresis |
| stack_cube not grasped | `~agent.is_grasping(cubeA)` | `gripper_open \|\| inhand_dis > 0.03` |
| push/pull/roll distance | `norm(cube_xy - target_xy) < goal_radius` | 同 |
| lift_peg_upright | `euler[:,2] vs pi/2` | `dot(body_x, world_z) > 0.99` (更鲁棒) |
| place_sphere on-bin | xy within 5mm, z at radius+half_z | 同 |

### 触觉集成

所有任务包含 `check_early_stop()` 的 `tactile_overpressed()` 检查，防止 gel pad 过压损伤。

## 合规性

- ManiSkill: Apache License 2.0
- UniVTAC: Apache License 2.0
- 所有派生逻辑记录在 `third_party/MANISKILL_ATTRIBUTION.md`
- 未复制任何 ManiSkill 源代码 — 全部基于 UniVTAC `BaseTask` / `Atom` / `TactileManager` 重写
