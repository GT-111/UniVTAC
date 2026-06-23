# ManiSkill Attribution

The following task designs are derived from [ManiSkill](https://github.com/haosulab/ManiSkill):

| UniVTAC Task | Derived From | Category |
|---|---|---|
| `stack_cube` | `StackCube-v1` | Object stacking |
| `push_cube` | `PushCube-v1` | Non-prehensile pushing |
| `pull_cube` | `PullCube-v1` | Non-prehensile pulling |
| `lift_peg_upright` | `LiftPegUpright-v1` | In-hand reorientation |
| `poke_cube` | `PokeCube-v1` | Tool-mediated pushing |
| `roll_ball` | `RollBall-v1` | Dynamic rolling contact |
| `place_sphere` | `PlaceSphere-v1` | Bin placement |

**License**: ManiSkill is licensed under the Apache License, Version 2.0.
Full text: https://github.com/haosulab/ManiSkill/blob/main/LICENSE

**What was derived**: Task semantics (objective, success criteria, randomization strategy).
**What was reimplemented**: All code is original, written against the UniVTAC/Isaac Sim
task framework (`BaseTask`), UIPC physics pipeline, and `Atom` manipulation primitives.
No ManiSkill source code was copied.

**Attribution note**: If you redistribute this project, you must retain this notice
and a copy of the Apache 2.0 license text (see `../LICENSE`) as required by
Section 4 of the Apache License, Version 2.0.
