# VLA-hack

## Введение

Ваша задача — научить манипулятор **SO-101** класть объекты в тарелку. Конкретно: робот должен уметь справляться с тремя заданиями:

| # | Задание |
|---|---------|
| 🧊 | Положить **кубик** в тарелку |
| 🎾 | Положить **теннисный мячик** в тарелку |
| ❓ | Положить **секретный объект** в тарелку *(объект будет раскрыт на финале)* |


### Мы советуем следующие подходы:

1. **Записать масштабный датасет в симуляции** — используйте MuJoCo, это быстро и не требует железа.
2. **Дополнить данными из реальности** — у нас есть сетап с leader/follower-руками и двумя камерами; реальные данные сильно улучшают качество.
3. **Задуматься о Sim2Real** — потому что симуляционные данные далеко не всегда совпадают с реальными.
4. **Дообучить VLA-модель** — советуем начать со [SmolVLA](https://huggingface.co/blog/smolvla), она лёгкая и хорошо подходит для старта.

Также у нас есть уже собранный датасет, можете рассмотреть его: [data](https://drive.google.com/file/d/17pKcWhjH6h93pfkWoLvjW5kTLeCzqAog/view?usp=sharing)

## Сбор датасета

### Что такое MuJoCo и зачем он здесь

**MuJoCo** (Multi-Joint dynamics with Contact) — физический симулятор для робототехники. Он реалистично считает динамику суставов, контакты и трение, при этом работает очень быстро (десятки тысяч шагов симуляции в секунду) и умеет рендерить изображения с виртуальных камер.

В этом проекте MuJoCo симулирует роборуку **SO-101**: вы управляете ею в виртуальной среде и собираете датасет демонстраций (видео + команды), который потом идёт на обучение VLA-модели.

**Ключевые понятия, которые встретятся в коде:**

| Термин | Что это |
|--------|---------|
| `mjModel` | Статическое описание сцены — тела, суставы, геометрия. Загружается из XML. |
| `mjData` | Текущее состояние симуляции — позиции, скорости, силы. Меняется на каждом шаге. |
| `qpos` | Углы суставов (что сейчас «куда повёрнуто»). |
| `ctrl` | Вектор команд двигателям — то, что вы отправляете руке. |
| `actuator` | Виртуальный двигатель, управляющий суставом. |

Сцена описана в XML-файлах (`asset/`). Каждый объект — отдельный файл, подключаемый через `<include>`. Подробнее — в [MUJOCO.md](./MUJOCO.md).

### Запуск MuJoCo и запись демонстраций

1. Установите зависимости и скачайте ассеты:

```bash
./install.sh
```

2. Активируйте окружение:

```bash
source .venv/bin/activate
```

3. Запустите сбор данных:

```bash
python -m collect_data.run
```

По умолчанию запись идёт в папку `simulation_data_fixed/`, а целевое число эпизодов задаётся в `collect_data/config.py` через `num_demo`.

Что происходит во время записи:

- если `use_master_arm=True`, управление идёт с leader-руки;
- если `use_master_arm=False`, можно управлять с клавиатуры;
- запись эпизода стартует автоматически при первом заметном движении;
- `Z` сбрасывает сцену и очищает текущий несохранённый эпизод;
- `X` принудительно сохраняет текущий эпизод;
- если куб успешно поставлен на тарелку, эпизод сохраняется автоматически.

Клавиши в режиме управления с клавиатуры:

- `Q/A` — `shoulder_pan`
- `W/S` — `shoulder_lift`
- `E/D` — `elbow_flex`
- `I/K` — `wrist_flex`
- `O/L` — `wrist_roll`
- `Space` — открыть/закрыть gripper
- `Z` — reset сцены

### В каком формате сохраняется датасет

Датасет сохраняется в формате **LeRobotDataset**, совместимом с пайплайнами `lerobot`. Корневая папка по умолчанию: `simulation_data_fixed/`.

В каждом кадре записываются:

- `observation.images.front` — фронтальная камера, видео `480x640x3`;
- `observation.images.side` — боковая камера, видео `480x640x3`;
- `observation.state` — вектор `float32` размера `6`;
- `action` — вектор `float32` размера `6`;
- `task` — строка с названием задания, по умолчанию `Put cube on plate`.

Порядок координат в `observation.state` и `action` одинаковый:

- `shoulder_pan.pos`
- `shoulder_lift.pos`
- `elbow_flex.pos`
- `wrist_flex.pos`
- `wrist_roll.pos`
- `gripper.pos`

Единицы измерения:

- первые 5 значений — углы суставов в градусах;
- `gripper.pos` — скаляр в диапазоне `0..100`.

Внутри папки датасета `lerobot` хранит метаданные эпизодов и фич в `meta/`, а изображения кодирует как видеофайлы, а не как отдельные PNG на каждый кадр. Это удобно для последующего обучения и загрузки через `LeRobotDataset.resume(...)`.

### Сбор датасета в реальности

Помимо симуляции, датасет можно собирать на реальном железе. У нас есть такой сетап:

- **Leader-рука** — человек держит её и показывает движение (телеоперация).
- **Follower-рука** — физический робот SO-101, который в реальном времени повторяет движения leader-руки и взаимодействует с объектами.
- **Две камеры** — фиксируют происходящее с разных ракурсов; видео вместе с командами суставов и сохраняется как датасет.

Такой подход называется **имитационным обучением (imitation learning)**: человек один раз показывает правильное поведение, а модель учится его воспроизводить. Для записи и упаковки демонстраций используется библиотека [LeRobot](https://github.com/huggingface/lerobot).

Официальная документация по записи датасета через `lerobot-record`: [LeRobot Record Function](https://huggingface.co/docs/lerobot/il_robots#record-function).

Мы рекомендуем запускать запись так:

```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=my_follower \
    --robot.cameras="{ front: {type: opencv, index_or_path: /dev/video0, fourcc: MJPG, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: /dev/video2, fourcc: MJPG, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=my_leader \
    --dataset.repo_id=local/record-test \
    --dataset.root=./record-test \
    --dataset.fps=10 \
    --dataset.num_episodes=50 \
    --dataset.single_task="Put cube on plate" \
    --dataset.push_to_hub=False \
    --display_data=false \
    --dataset.streaming_encoding=false \
    --dataset.vcodec=h264 \
    --dataset.encoder_threads=2 \
    --robot.disable_torque_on_disconnect=false
```

Что здесь важно:

- `--robot.port=/dev/ttyACM1` и `--teleop.port=/dev/ttyACM0` нужно подставить под свои устройства;
- `--robot.cameras=...` задаёт две камеры: `front` и `side`;
- `--dataset.root=./record-test` определяет локальную папку с датасетом;
- `--dataset.repo_id=local/record-test` позволяет работать локально, без отправки на Hub;
- `--dataset.single_task="Put cube on plate"` задаёт текстовое описание задачи;
- `--dataset.streaming_encoding=false` и `--dataset.vcodec=h264` фиксируют предсказуемое локальное кодирование видео.

На Linux, если во время записи датасета не работают клавиши `Left`, `Right` и `Escape`, проверьте, что установлена переменная окружения `$DISPLAY`. Подробнее: [pynput limitations for Linux](https://pynput.readthedocs.io/en/latest/limitations.html#linux).

## Обучение моделей

Для первого эксперимента мы предлагаем начинать со **SmolVLA**: это хороший базовый VLA-пайплайн, на котором удобно быстро проверить весь цикл целиком, от датасета до инференса в симуляции.

Для обучения и инференса мы используем готовый образ на Docker Hub: [dpaleyev/lerobot-workshop](https://hub.docker.com/r/dpaleyev/lerobot-workshop).

Перед началом работы скачайте образ:

```bash
docker pull dpaleyev/lerobot-workshop:latest
```

### Обучение SmolVLA

Скрипт `run_official_smolvla_train_cached.sh` оборачивает `lerobot-train` в `docker run`, автоматически:

- монтирует репозиторий в `/app`;
- пробрасывает GPU;
- сохраняет Hugging Face cache в `outputs/hf_cache`;
- добавляет несколько дефолтных флагов для обучения.

Базовый запуск:

```bash
./run_official_smolvla_train_cached.sh \
    --policy.type=smolvla \
    --policy.device=cuda \
    --dataset.repo_id=local/record-test \
    --dataset.root=/app/record-test \
    --output_dir=/app/outputs/train/smolvla_so101 \
    --job_name=smolvla_so101
```

Все дополнительные аргументы после имени скрипта пробрасываются напрямую в `lerobot-train`, поэтому сюда можно добавлять свои параметры `dataset`, `policy`, `training` и `eval`.

### Оценка модели в симуляторе

Для оценки чекпоинта в MuJoCo используйте `run_smolvla_inference.py`. На удалённой машине мы рекомендуем запускать его в контейнере через VNC, чтобы видеть окно симуляции:

```bash
docker run --rm --gpus all \
    -v "$PWD:/app" \
    -w /app \
    dpaleyev/lerobot-workshop:latest \
    python run_smolvla_inference.py \
        --policy-path /app/outputs/train/smolvla_so101/checkpoints/last/pretrained_model \
        --dataset-root /app/record-test \
        --dataset-repo-id local/record-test \
        --episodes 10 \
        --max-steps 250 \
        --fps 10 \
        --summary-path /app/outputs/eval/smolvla_so101_summary.json
```

Этот скрипт:

- загружает обученный SmolVLA checkpoint;
- подтягивает статистики датасета для нормализации;
- прогоняет несколько эпизодов в MuJoCo;
- считает число успешных эпизодов и success rate;
- сохраняет итоговый JSON-отчёт, если передан `--summary-path`.



