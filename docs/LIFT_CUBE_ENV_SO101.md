# SO-101 Lift-Cube: среда, награды, таски и скрипты

Документ описывает **только Lift-Cube** для **SO-101** в расширении `isaac_so_arm101` (репозиторий `manipulation/`). Другие таски (Reach, SO-100, Target-Cube) здесь не разбираются.

---

## 1. Архитектура конфигов


| Файл                              | Назначение                                                                                                                               |
| --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `tasks/lift/lift_env_cfg.py`      | Базовый **LiftEnvCfg**: сцена (стол, свет, камеры), команды, действия, **наблюдения**, **награды** по умолчанию, терминации, куррикулум. |
| `tasks/lift/joint_pos_env_cfg.py` | Конкретные таски SO-101: робот **SO_ARM101**, joint-position действия, куб, `ee_frame`, варианты **Dense / Sparse / Play**.              |
| `tasks/lift/guided_env_cfg.py`    | Варианты **Guided** (trajectory / discriminator) поверх joint-pos конфигов.                                                              |
| `tasks/lift/mdp/rewards.py`       | Реализации наград, в т.ч. trajectory / discriminator guidance.                                                                           |
| `tasks/lift/__init__.py`          | Регистрация Gym ID → `env_cfg_entry_point` + `rsl_rl_cfg_entry_point`.                                                                   |


При `gym.make("Isaac-…-v0")` Isaac Lab по строке `env_cfg_entry_point` **импортирует класс** конфига и строит `ManagerBasedRLEnv` по этому `cfg`.

---

## 2. Сцена и время

- **Робот**: SO-101 (`SO_ARM101_CFG`), articulation `…/Robot`.
- **Объект**: куб DexCube (Nucleus USD), начальная позиция задаётся в конфиге.
- **Стол / пол / свет**: `ObjectTableSceneCfg` в `lift_env_cfg.py`.
- **EE**: `FrameTransformer` от `base_link` к `gripper_link` с offset (конец эффектора для наград/наблюдений).
- **Симуляция**: `sim.dt = 0.01` (100 Гц физики), **`decimation = 2`** → шаг среды **20 Гц** (`0.02` с на шаг RL).
- **Эпизод**: `episode_length_s = 5.0` → при 20 Гц это **до 125 шагов** на эпизод (если нет раннего done).
- **Команда `object_pose`**: целевая поза для куба (Uniform pose command к `gripper_link`), ресэмплинг каждые **5 с**; диапазоны поз в `CommandsCfg` (`lift_env_cfg.py`).

---

## 3. Действия (actions)

Для SO-101 в `SoArm101LiftCubeEnvCfg`:

- **`arm_action`**: `JointPositionActionCfg` — суставы `shoulder_.*`, `elbow_flex`, `wrist_.*`, `scale=0.5`, `use_default_offset=True`.
- **`gripper_action`**: `BinaryJointPositionActionCfg` — сустав `gripper`, открыто/закрыто через выражения команд.

Размер вектора действий задаётся этими двумя компонентами действия.

---

## 4. Наблюдения (observations)

### 4.1. Группа `policy` (вектор состояния)

Все **компоненты** вектора наблюдения **склеиваются** в один тензор (`concatenate_terms = True`), при необходимости с **добавлением шума к наблюдениям** (в Isaac Lab — *observation corruption*), если `enable_corruption = True` (в Play часто отключают).


| Ключ (логически)         | Функция MDP                                         | Содержание                                       |
| ------------------------ | --------------------------------------------------- | ------------------------------------------------ |
| `joint_pos`              | `joint_pos_rel`                                     | Позиции суставов относительно дефолта робота.    |
| `joint_vel`              | `joint_vel_rel`                                     | Скорости суставов относительно дефолта.          |
| `object_position`        | `object_position_in_robot_root_frame`               | Позиция объекта в корневом фрейме робота (3D).   |
| `target_object_position` | `generated_commands` (`command_name="object_pose"`) | Команда цели (поза цели в пространстве команды). |
| `actions`                | `last_action`                                       | Предыдущее действие агента.                      |


Источники: `ObservationsCfg.PolicyCfg` в `lift_env_cfg.py`, часть функций — из `isaaclab.envs.mdp`, `object_position_in_robot_root_frame` — локально в `tasks/lift/mdp/observations.py`.

### 4.2. Группа `observation` (изображения)

Включена структура **`ImagesCfg`**: камеры `camera_top`, `camera_wrist` (Tiled RGB, 640×480). Для совместимости со скриптами есть **алиасы** `images_side` (= top) и `images_up` (= wrist).

**Важно для обучения без зрения:** при флаге **`disable_task_cameras = True`** в конфиге камеры **не спавнятся**, компоненты наблюдения с изображениями **обнуляются** (`lift_env_cfg.__post_init__`). В скриптах RSL-RL это делается через CLI **`--disable_task_cameras`**.

Чтобы получить RGB в рантайме, нужен запуск с **`--enable_cameras`** у `AppLauncher` / `isaaclab.sh` (как в `run_smolvla_isaac.py` / `save_env_cameras.py`).

---

## 5. Награды: общая форма

На каждом шаге среды Isaac Lab формирует **скалярную награду** как **сумму по компонентам награды**:

$$
R_t = \sum_i w_i \, r_{i,t}
$$

где $w_i$ — **вес** из `RewTerm(..., weight=...)`, $r_{i,t}$ — выход функции награды (часто уже в $[0,1]$ или штраф).

Ниже — **базовые** функции из `tasks/lift/mdp/rewards.py` (и стандартный MDP Isaac Lab для штрафов).

---

## 6. Dense Lift-Cube (`Isaac-SO-ARM101-Lift-Cube-v0`)

Базовый класс **`RewardsCfg`** в `lift_env_cfg.py` + SO-101 в `SoArm101LiftCubeEnvCfg`.

### 6.1. `reaching_object` — подвести EE к объекту

$$
r_{\text{reach}} = 1 - \tanh\left(\frac{\|p_{\text{obj}} - p_{\text{ee}}\|}{\sigma_{\text{reach}}}\right), \quad \sigma_{\text{reach}} = 0.05
$$

- $p_{\text{obj}}$, $p_{\text{ee}}$ — позиции объекта и EE (world), EE из `ee_frame`.
- **Вес** $w_{\text{reach}} = 1.0$.

### 6.2. `lifting_object` — поднять выше порога

$$
r_{\text{lift}} = \mathbb{1}\{ z_{\text{obj}} > h_{\min} \}, \quad h_{\min} = 0.025\,\text{м}
$$

- **Вес** $w_{\text{lift}} = 15.0$.

### 6.3. `object_goal_tracking` — грубое ведение к цели

Пусть $p_{\text{goal}}$ — целевая позиция команды `object_pose` в world (из команд робота и `des_pos_b`), $p_{\text{obj}}$ — позиция объекта.

$$
r_{\text{goal, coarse}} = \mathbb{1}\{ z_{\text{obj}} > h_{\min} \} \cdot \left(1 - \tanh\left(\frac{\|p_{\text{goal}} - p_{\text{obj}}\|}{\sigma_{\text{coarse}}}\right)\right)
$$

- $\sigma_{\text{coarse}} = 0.3$, $h_{\min} = 0.025$.
- **Вес** $16.0$.

### 6.4. `object_goal_tracking_fine_grained` — то же с мелким ядром

Та же формула, но $\sigma_{\text{fine}} = 0.05$, **вес** $5.0$.

### 6.5. `action_rate` и `joint_vel`

Стандартные компоненты награды в Isaac Lab:

- **`action_rate_l2`**: штраф за **изменение действий** между шагами (L2; знак минус через вес).
- **`joint_vel_l2`**: штраф за **скорости суставов** робота.

В Dense конфиге: **веса** $-10^{-4}$ каждый.

### 6.6. Куррикулум (Dense)

В `CurriculumCfg` для компонентов `action_rate` и `joint_vel` вызывается **`modify_reward_weight`**: за **10000** шагов веса меняются к **$-0.1$** (более сильный штраф после разгона). Это влияет только на Dense вариант, где эти штрафы включены.

---

## 7. Sparse Lift-Cube (`Isaac-SO-ARM101-Lift-Cube-Sparse-v0`)

Класс **`SoArm101LiftCubeSparseEnvCfg`**:

- Плотные шейпы **`reaching_object`**, **`object_goal_tracking`**, **`object_goal_tracking_fine_grained`** — **вес 0** (отключены).
- Остаётся **`lifting_object`** с весом **1.0** и $h_{\min}=0.025$.
- Лёгкие штрафы **`action_rate`** / **`joint_vel`** остаются с $-10^{-4}$ (как в базовом Sparse описании в коде).

Итого в типичном шаге доминирует **бинарный лифт** + маленькие регуляризаторы.

---

## 8. Guided (trajectory) поверх Dense / Sparse

Классы **`GuidedRewardsCfg`** добавляют:

### 8.1. `trajectory_guidance`

После матчинга траектории к раскладке (см. `TrajectoryStore`) доступны режимы **`guidance_mode`** в `guided_env_cfg.py`:

1. **`path_progress`** (по умолчанию): к полилинии учительской траектории в 3D строится **проекция** текущего EE; $s \in [0,1]$ — доля **длины пути** от старта до точки проекции. Награда за **прогресс вперёд** за шаг (независимо от того, на каком по счёту шаге симуляции находится учитель в записи):

$$
\Delta s_t = s_t - s_{t-1}, \quad r_{\text{traj}} = \lambda_p \tanh\left(\frac{\operatorname{relu}(\Delta s_t)}{\sigma_{\Delta}}\right) - \lambda_\ell \tanh\left(\frac{\ell_t}{\sigma_{\ell}}\right)
$$

где $\ell_t$ — **латеральное** расстояние до полилинии (второй член опционален, `lateral_penalty_weight`; в дефолтном `GuidedRewardsCfg` **выключен**, $\lambda_\ell=0$).

   Опционально **`progress_lateral_gate`**: если задан и $\ell_t$ выше порога, **приращение** $\Delta s_t$ обнуляется (в дефолте **`None`**, гейт выключен).

2. **`path_progress_milestones`**: путь делится на **`num_path_milestones`** интервалов по длине дуги; награда только за **последовательное** продвижение по «вехам» (не больше **`max_milestone_jump`** за шаг), при **`milestone_lateral_gate`** на латеральную ошибку.

3. **`time_sync`** (legacy): тот же индекс времени, что и у записи — $d_t = \|p_{\text{ee}}^{\text{student}} - p_{\text{ee}}^{\text{teacher}}(t)\|$:

$$
r_{\text{traj}} = 1 - \tanh\left(\frac{d_t}{\sigma}\right)
$$

**Вес** основного терма $5.0$ (по умолчанию). Параметры см. `GuidedRewardsCfg` (`path_progress_delta_std`, `path_progress_scale`, опционально `progress_lateral_gate` / `lateral_penalty_weight`, milestone-поля, …).

Путь к файлу траекторий: **`ISAAC_SO_ARM101_TRAJECTORY_FILE`** или дефолт  
`isaac_so_arm101/logs/rsl_rl/teacher_trajectories/so101_lift_cube_teacher.pt` (относительно `cwd`).

Датасет от **`collect_trajectories.py`** содержит **`gripper_trajectories`** (позиция сустава `gripper` на каждом шаге, рад; 0 — сжато, 0.5 — открыто в терминах `BinaryJointPositionActionCfg`). Старые `.pt` без этого ключа по-прежнему загружаются (нули).

**Какой ряд датасета выбрать для env:** по умолчанию **`match_mode: object_only`** — по положению куба (и при необходимости цели), **не** по EE. После сброса событие **`align_ee_to_teacher_trajectory_start`** (guided-конфиги) подводит EE к **первой точке** выбранной траектории через **differential IK** по суставам руки — это не replay `joint_pos` из записи учителя, а геометрическое «дотянуться» до той же мировой точки tool-frame (переносимо на реального робота с тем же IK).

### 8.1b. `teacher_gripper_alignment`

Сопоставление сустава захвата студента с учителем по **индексу времени, выровненному по пути**: $t = \mathrm{round}(s \cdot (T-1))$. Награда $1 - \tanh(|\Delta g|/\sigma)$; если в файле нет `gripper_trajectories`, терм даёт ноль.

### 8.1c. Наблюдения `GuidedObservationsCfg`

В policy-вектор добавляется один терм **`teacher_gripper_cmd_and_close_hint`** (форма `N×2`): колонка 0 — целевое положение захвата учителя по пути, колонка 1 — бинарный признак «закрыть» (ниже порога `close_threshold`). Расчёт использует **общий кэш** проекции EE на полилинию с наградами траектории (одна проекция на шаг).

### 8.2. `trajectory_guidance_debug_distance_over_std`

**Вес** $10^{-3}$ (слабо влияет на обучение). В режимах `path_progress` / `path_progress_milestones` в лог уходит **латеральная** ошибка $\ell_t / \sigma_{\ell}$; в режиме `time_sync` — $d_t / \sigma$, как раньше.

### 8.3. `discriminator_guidance`

В **`GuidedRewardsCfg`** **вес 0** — дискриминатор не грузится.

### Guided Sparse (`Isaac-SO-ARM101-Guided-Lift-Cube-Sparse-v0`)

Наследует Sparse: плотные task-шейпы отключены, **`lifting_object`** вес **1.0**. Лёгкие **`action_rate`** / **`joint_vel`** и куррикулум к ним **как в обычном Sparse** ($-10^{-4}$ → $-0.1$ по шагам), чтобы снизить дрожание; плотный сигнал идёт от **trajectory_guidance** (+ слабый отладочный компонент награды для логов).

---

## 9. Guided Sparse + Discriminator (`Isaac-SO-ARM101-Guided-Lift-Cube-Sparse-Discriminator-v0`)

Используется **`GuidedDiscriminatorRewardsCfg`**: компоненты **trajectory guidance** с **весом 0**, активен **`discriminator_guidance`** (обучаемый лог $D$ по переходам EE; детали и сглаживание — в docstring `discriminator_guidance_reward` в `mdp/rewards.py`). Нужен файл дискриминатора:

- **`ISAAC_SO_ARM101_TRAJECTORY_DISCRIMINATOR_FILE`** или дефолт  
`isaac_so_arm101/logs/rsl_rl/teacher_trajectories/trajectory_discriminator_lift_cube.pt`.

Плюс событие **`DiscriminatorDiagnosticsEventCfg`** — сброс метрик в `extras['log']` для TensorBoard.

---

## 10. Терминации

Из `TerminationsCfg` (`lift_env_cfg.py`):

- **`time_out`**: окончание по времени эпизода.
- **`object_dropping`**: высота объекта ниже **-0.05** м (`root_height_below_minimum`).

---

## 11. Зарегистрированные Gym ID (Lift-Cube, SO-101)


| ID                                                         | Конфиг          | Примечание                                  |
| ---------------------------------------------------------- | --------------- | ------------------------------------------- |
| `Isaac-SO-ARM101-Lift-Cube-v0`                             | Dense           | Основной dense RL.                          |
| `Isaac-SO-ARM101-Lift-Cube-Play-v0`                        | Play            | Меньше envs, без corruption.                |
| `Isaac-SO-ARM101-Lift-Cube-Sparse-v0`                      | Sparse          | Только lift + лёгкие штрафы.                |
| `Isaac-SO-ARM101-Lift-Cube-Sparse-Play-v0`                 | Sparse Play     |                                             |
| `Isaac-SO-ARM101-Guided-Lift-Cube-v0`                      | Guided Dense    | Нужен `.pt` траекторий учителя.             |
| `Isaac-SO-ARM101-Guided-Lift-Cube-Sparse-v0`               | Guided Sparse   |                                             |
| `Isaac-SO-ARM101-Guided-Lift-Cube-Sparse-Discriminator-v0` | + Discriminator | Нужны траектории + обученный дискриминатор. |
| `…-Play-v0` варианты                                       | Play            | Для визуализации / отладки.                 |
| `Isaac-SO-ARM101-FixedLayout-Lift-Cube-v0`                 | Dense, фикс. расклад | Куб без джиттера сброса + фиксированная цель (`fixed_layout_env_cfg.py`). |
| `Isaac-SO-ARM101-FixedLayout-Guided-Lift-Cube-Sparse-v0`   | Guided Sparse   | Тот же расклад; учитель и `collect_trajectories` — с **тем же** task ID. |
| `…-FixedLayout-…-Play-v0`                                  | Play            | Мало env, без corruption.                 |

Подробный пайплайн (teacher → `.pt` → guided): **[FIXED_LAYOUT_LIFT_EXPERIMENT.md](FIXED_LAYOUT_LIFT_EXPERIMENT.md)**.

---

## 12. Скрипты: обучение и инференс (RSL-RL)

Запуск из каталога проекта в контейнере обычно **`/workspace/isaac-bridge`**, через **`isaaclab`** (или `./isaaclab.sh` из корня Isaac Lab).

Логи по умолчанию: **`isaac_so_arm101/logs/rsl_rl/<experiment_name>/`** (см. `scripts/rsl_rl/log_paths.py`). Переменная **`ISAAC_SO_ARM101_RSL_RL_LOG_ROOT`** переопределяет корень. Имена: **`lift`** / **`guided_lift_cube`** (рандомизированная сцена), **`lift_fixed_layout`** / **`guided_lift_cube_fixed_layout`** (fixed-layout таски — см. `FIXED_LAYOUT_LIFT_EXPERIMENT.md`).

### 12.1. Обучение PPO (в т.ч. Sparse / Guided)

```bash
isaaclab -p -m isaac_so_arm101.scripts.rsl_rl.train \
  --task Isaac-SO-ARM101-Lift-Cube-Sparse-v0 \
  --disable_task_cameras \
  --headless
```

Подставьте нужный `--task`. Для Guided заранее положите файл траекторий или задайте `ISAAC_SO_ARM101_TRAJECTORY_FILE`.

**Wall-clock (долгие итерации / ETA много часов):** базовый `LiftEnvCfg` исторически тянет **`num_envs=4096`**; для SO-101 + куб это даёт очень тяжёлую симуляцию на шаг. В `SoArm101LiftCubeEnvCfg` по умолчанию выставлено **`2048`** сред — при необходимости масштабируйте вручную: `--num_envs 1024` (быстрее итерация, меньше данных за rollout) или `--num_envs 4096` (если хватает VRAM и нужно больше параллелизма). Для коротких прогонов: `--max_iterations 300` (и т.д.). Hydra позволяет уменьшить нагрузку на PPO-апдейт, напр. `runner.algorithm.num_learning_epochs=4`.

### 12.2. Инференс / проигрывание чекпоинта

```bash
isaaclab -p -m isaac_so_arm101.scripts.rsl_rl.play \
  --task Isaac-SO-ARM101-Lift-Cube-Play-v0 \
  --checkpoint /path/to/model_XXXX.pt
```

Или без явного checkpoint — поиск в каталоге эксперимента (`load_run`, `load_checkpoint` в agent cfg).

---

## 13. Артефакты для Guided: как получить

### 13.1. Чекпоинт «учителя» (dense lift-cube)

Обучите **`Isaac-SO-ARM101-Lift-Cube-v0`** (или другой совместимый teacher-task), возьмите `model_*.pt` из `isaac_so_arm101/logs/rsl_rl/.../`.

### 13.2. Датасет траекторий EE (`so101_lift_cube_teacher.pt`)

```bash
isaaclab -p -m isaac_so_arm101.scripts.rsl_rl.collect_trajectories \
  --task Isaac-SO-ARM101-Lift-Cube-v0 \
  --checkpoint /path/to/teacher_model.pt \
  --num_episodes 500 \
  --disable_task_cameras \
  --headless
```

По умолчанию выход:  
`isaac_so_arm101/logs/rsl_rl/teacher_trajectories/so101_lift_cube_teacher.pt`.

### 13.3. Дискриминатор (только для `…-Discriminator-v0`)

```bash
isaaclab -p -m isaac_so_arm101.scripts.train_trajectory_discriminator \
  --teacher_file .../so101_lift_cube_teacher.pt \
  --output .../trajectory_discriminator_lift_cube.pt
```

(параметры по умолчанию см. в скрипте).

### 13.4. Обучение Guided после артефактов

```bash
export ISAAC_SO_ARM101_TRAJECTORY_FILE=/workspace/isaac-bridge/isaac_so_arm101/logs/rsl_rl/teacher_trajectories/so101_lift_cube_teacher.pt   # если не дефолт

isaaclab -p -m isaac_so_arm101.scripts.rsl_rl.train \
  --task Isaac-SO-ARM101-Guided-Lift-Cube-Sparse-v0 \
  --disable_task_cameras \
  --headless
```

---

## 14. Скрипт VLA-инференса (не RSL-RL)

Для **SmolVLA** на тех же тасках:

```bash
isaaclab -p scripts/run_smolvla_isaac.py \
  --task Isaac-SO-ARM101-Lift-Cube-v0 \
  --enable_cameras
```

Камеры/JSON: см. `README.md` в `manipulation/` (`--camera_json`, `--camera_usd`).

---

## 15. Отладка камер (PNG)

```bash
isaaclab -p scripts/save_env_cameras.py --task Isaac-SO-ARM101-Lift-Cube-v0
```

---

*Исходники: `manipulation/isaac_so_arm101/src/isaac_so_arm101/tasks/lift/`.*