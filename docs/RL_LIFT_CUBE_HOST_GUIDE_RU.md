# Обучение RL-политики (MLP, PPO) в Isaac Lab и запись LeRobot-датасета
## на примере SO-ARM101 Lift Cube



## 1) Что нужно заранее

- Установлены Isaac Sim и Isaac Lab.
- Запускаем через `./isaaclab.sh -p ...`
- В Python-окружении есть зависимости для LeRobot (для скрипта сбора датасета).

---

## 2) Команды



Используются скрипты:

- `isaac_so_arm101/src/isaac_so_arm101/scripts/rsl_rl/train.py`
- `isaac_so_arm101/src/isaac_so_arm101/scripts/rsl_rl/play.py`
- `isaac_so_arm101/src/isaac_so_arm101/scripts/rsl_rl/collect_lerobot_dataset.py`

---

## 3) Дефолт таска для проверки


- Для обучения: `Isaac-SO-ARM101-Lift-Cube-v0`
- Для эвала/сбора датасета: `Isaac-SO-ARM101-Lift-Cube-Play-v0`

---

## 4) Обучение PPO MLP



```bash
./isaaclab.sh -p path/to/isaac_so_arm101/src/isaac_so_arm101/scripts/rsl_rl/train.py \
  --task Isaac-SO-ARM101-Lift-Cube-v0 \
  --disable_task_cameras \
  --headless
```

### Полезные дополнительные флаги

```bash
--num_envs 1024
--max_iterations 1500
--seed 42
```

Примечания:

- `--disable_task_cameras` ускоряет обучение MLP (без рендера RGB-камер).
- Архитектура PPO MLP по умолчанию берется из `rsl_rl_ppo_cfg.py` (скрытые слои actor/critic `[256, 128, 64]`).

---

## 5) Где искать чекпоинт

Логи и модели сохраняются в:

- `isaac_so_arm101/logs/rsl_rl/lift/<timestamp>/`

Обычно финальный чекпоинт:

- `model_1499.pt`

---

## 6) Проверка policy (play)

```bash
isaaclab -p isaac_so_arm101/src/isaac_so_arm101/scripts/rsl_rl/play.py \
  --task Isaac-SO-ARM101-Lift-Cube-Play-v0 \
  --checkpoint isaac_so_arm101/logs/rsl_rl/lift/<timestamp>/model_1499.pt
```

Если политика стабильно поднимает куб, можно переходить к записи датасета.

---

## 7) Запись LeRobot датасета обученной MLP-политикой

```bash
./isaaclab.sh -p /path/to/isaac_so_arm101/src/isaac_so_arm101/scripts/rsl_rl/collect_lerobot_dataset.py \
  --task Isaac-SO-ARM101-Lift-Cube-Play-v0 \
  --checkpoint isaac_so_arm101/logs/rsl_rl/lift/<timestamp>/model_1499.pt \
  --num_episodes 50 \
  --output_dir output/datasets/so101-lift-cube \
  --repo_id <HF_USER>/so101-lift-cube \
  --fps 30 \
  --success_only \
  --headless
  --push_to_hub
```

### Что сохраняется в датасет

- `observation.images.top`
- `observation.images.wrist`
- `observation.images.side`
- `observation.state` (joint positions в градусах)
- `action` (действие policy)

### Важно

- Если `--output_dir` уже существует, по умолчанию он перезаписывается.
- Чтобы дописывать в существующий датасет, добавьте `--resume-dataset`.
- `--fps 30` - можете поменять, обратите внимание, что герцовка не меняется - ее менять нужно отдельно через sim.dt и decimation.

---

## 8) (Опционально) загрузка датасета на Hugging Face

Добавьте к команде сбора:

```bash
--push_to_hub
```

И убедитесь, что в текущем окружении настроена HF-аутентификация.

---

## 9) Частые проблемы

1. Не найден checkpoint:
   - проверьте путь относительно `manipulation/` или используйте абсолютный путь.
2. Слишком медленно обучается:
   - уменьшите `--num_envs`, оставьте `--disable_task_cameras`.
3. Проблемы с GUI / display:
   - используйте `--headless`.
4. Папка датасета уже существует:
   - удалите вручную или добавьте `--resume-dataset`.
5. Не грузится на HF Hub:
   - проверьте токен и логин в том же env, где запускается `isaaclab`.

---

## 10) Как добавлять свои таски?

Ниже — типичный путь для **расширения `isaac_so_arm101`** (как сделано для lift-cube). Другие варианты (чистый Isaac Lab без этого расширения, LeIsaac) здесь не расписываются.

### 1. Конфиг среды

- Добавьте класс конфига (например, наследник `ManagerBasedRLEnvCfg` или существующего lift/reach конфига).
- Исходники lift-cube и регистрация ID: `isaac_so_arm101/src/isaac_so_arm101/tasks/lift/`.
- Подробно про устройство lift-сцены, награды и зарегистрированные ID: `LIFT_CUBE_ENV_SO101.md`.

### 2. Регистрация Gym ID

- В `tasks/lift/__init__.py` (или в `__init__.py` нового пакета задач) вызовите `gymnasium.gym.register(...)` с полями:
  - `id` — строка вида `Isaac-SO-ARM101-My-Task-v0`;
  - `entry_point="isaaclab.envs:ManagerBasedRLEnv"` (или другой подходящий env);
  - `kwargs`: `env_cfg_entry_point` (модуль:класс конфига) и `rsl_rl_cfg_entry_point` (модуль:класс runner, например PPO из `tasks/lift/agents/`).

### 3. Импорт при запуске скриптов

- Скрипты `train.py` / `play.py` / `collect_lerobot_dataset.py` уже делают `import isaac_so_arm101.tasks.lift` — при **новом пакете** задач добавьте аналогичный `import`, иначе `gym.make("--task ...")` не увидит ваш ID.

### 4. Обучение и сбор данных

- В командах из разделов 4–7 подставьте свой `--task` и при необходимости свой `experiment_name` в runner-конфиге (куда пишутся логи под `isaac_so_arm101/logs/rsl_rl/`).

### 5. Сбор LeRobot и скрипт `collect_lerobot_dataset.py`

Скрипт ожидает в сцене сенсоры с именами **`camera_top`**, **`camera_wrist`**, **`camera_side`** (как в базовом lift-конфиге). Ключи в датасете задаются флагом **`--dataset-cameras`** (`top`, `wrist`, `side` — подмножество этих трёх).

Если у вашей задачи **другие имена камер** или другой состав наблюдений, нужно либо:

- назвать сенсоры так же в конфиге сцены, либо
- доработать `collect_lerobot_dataset.py` под ваши имена / дополнительные поля.

### 6. Документация Isaac Lab (общий случай)

- Официальные туториалы по кастомным manager-based env: [Isaac Lab — tutorials](https://isaac-sim.github.io/IsaacLab/main/index.html) (раздел про создание/регистрацию env и задач).
