# 任务成功判定标准

| 任务 | 位置 | 姿态 | 其他 |
|------|------|------|------|
| grasp_classify | 相对目标 pad：\|x\|<2cm，\|y\|<2cm | z 轴竖直（≈15°内） | — |
| insert_hole | 相对目标孔：\|x\|、\|y\|<1cm | z 轴竖直（≈8°内） | 未从夹爪滑落 |
| insert_tube | 相对孔位：x、y < 5mm | z 轴竖直（≈15°内） | 未从夹爪滑落 |
| insert_HDMI | 相对 slot：y、z < 5mm | z 轴竖直（≈15°内） | 末端已抬起 |
| lift_can | can 贴桌面 | x 轴竖直（侧立） | — |
| lift_bottle | 相对墙前：x > -2cm，\|y\|<10cm | x 轴竖直（侧立） | — |
| put_bottle_in_shelf | 相对放置点：\|x\|<2cm，\|y\|<10cm | z 轴竖直（≈15°内） | — |
| pull_out_key | key 已拔出 | key z 轴竖直 | slot 未损坏；手仍抓着 key |
| collect | — | — | 恒为成功 |

**说明**
- Eval：每步调用 `check_success()`，成功则 `eval_success = True`
- 采集：还需 `plan_success` 且 `check_early_stop()` 为假
