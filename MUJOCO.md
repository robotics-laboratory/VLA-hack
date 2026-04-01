# MuJoCo — быстрый старт

---

## Что такое MuJoCo?

**MuJoCo** (Multi-Joint dynamics with Contact) — физический симулятор, разработанный специально для робототехники. Он умеет:
- точно считать динамику твёрдых тел (суставы, контакты, трение)
- работать очень быстро (десятки тысяч шагов в секунду)
- рендерить картинку с камер (для обучения нейросетей)

В этом проекте MuJoCo симулирует роборуку **SO-101**, которая должна перекладывать кружку на тарелку.

---

## Ключевые понятия

| Термин | Что это |
|--------|---------|
| **model** (`mjModel`) | Статическое описание сцены — тела, суставы, геометрия. Загружается из XML. |
| **data** (`mjData`) | Текущее состояние симуляции — позиции, скорости, силы. Меняется на каждом шаге. |
| **body** | Твёрдое тело (кружка, звено руки, стол). |
| **joint** | Сустав между телами (вращательный `hinge`, линейный `slide`). |
| **geom** | Геометрическая форма для коллизий и рендера (box, sphere, mesh…). |
| **site** | Невидимая точка в пространстве — удобно для целевых позиций IK, датчиков. |
| **actuator** | Двигатель, управляющий суставом. Получает команды из вектора управления. |
| **qpos** | Вектор обобщённых координат (углы суставов, позиции слайдеров). |
| **qvel** | Вектор обобщённых скоростей. |
| **ctrl** | Вектор команд actuator-ам. |

---

## Добавление объектов через отдельные файлы

### `<include>` — главный инструмент

Сцена в MuJoCo не обязана быть одним монолитным XML. Тег `<include>` вставляет содержимое другого файла целиком — как `#include` в C. Это позволяет держать каждый объект в своём файле и переиспользовать их.

Так устроен `asset/example_scene_y.xml` в нашем проекте:

```xml
<mujoco model="Scanned Objects">
    <option integrator="RK4" noslip_iterations="20" />

    <!-- Пол -->
    <include file="./tabletop/object/floor_isaac_style.xml" />
    <!-- Стол + камеры -->
    <include file="./tabletop/object/object_table.xml" />
    <!-- Рука SO-101 -->
    <include file="so101/so101_new_calib.xml" />
    <!-- Объекты манипуляции -->
    <include file="./objaverse/mug_5/model_new.xml"/>
    <include file="./objaverse/plate_11/model_new.xml"/>
</mujoco>
```

> **Важно:** пути в `file=""` — **относительно файла, в котором написан `<include>`**, а не рабочей директории Python.

---

### Формат файла объекта

Файл, подключаемый через `<include>`, — это **фрагмент** XML, у которого корневой тег `<mujocoinclude>` (для фрагментов) или полноценный `<mujoco>`. Типичная структура:

```xml
<!-- asset/tabletop/object/obj_cylinder.xml -->
<mujocoinclude>
    <worldbody>
        <body name="obj_cylinder_01" pos="0.3 0.0 0.82">
            <joint type="free" />
            <geom fromto="0 0 0.1  0 0 0" size="0.025"
                  type="cylinder"
                  density="500"
                  friction="1 0.005 0.0001"
                  rgba="0.5 0.56 0.43 1" />
        </body>
    </worldbody>
</mujocoinclude>
```

Или полноценный файл, если у объекта есть свои ассеты (меши, текстуры):

```xml
<!-- asset/objaverse/mug_5/model_new.xml -->
<mujoco model="mug">
  <asset>
    <mesh file="visual/model_normalized_0.obj" name="mug_vis"
          scale="0.12 0.12 0.12" />
    <texture type="2d" name="mug_tex" file="visual/image0.png" />
    <material name="mug_mat" texture="mug_tex" specular="0.5" shininess="0.36" />
  </asset>
  <worldbody>
    <body name="body_obj_mug_5">
      <freejoint />
      <geom type="mesh" mesh="mug_vis" material="mug_mat"
            friction="0.95 0.3 0.1" density="100"
            solimp="0.998 0.998 0.001" solref="0.001 1"
            contype="0" conaffinity="0" group="2" />
      <!-- Невидимые collision-меши -->
      <geom type="mesh" mesh="mug_coll" group="3" rgba="0 0 0 0"
            friction="0.95 0.3 0.1" density="100"
            solimp="0.998 0.998 0.001" solref="0.001 1" />
      <!-- Опорные точки для проверки успеха -->
      <site name="bottom_site_mug_5" pos="0 0 -0.043" size="0.005" rgba="0 0 0 0"/>
      <site name="top_site_mug_5"    pos="0 0  0.043" size="0.005" rgba="0 0 0 0"/>
    </body>
  </worldbody>
</mujoco>
```

Чтобы добавить новый объект в сцену — достаточно одной строки в главном XML:

```xml
<include file="./objaverse/my_new_object/model.xml"/>
```

---

### Пошаговая инструкция: добавить простой примитив

**1. Создайте файл** `asset/tabletop/object/obj_mybox.xml`:

```xml
<mujocoinclude>
    <worldbody>
        <body name="body_obj_mybox" pos="0.35 0.0 0.85">
            <joint type="free" />
            <geom type="box"
                  size="0.03 0.03 0.03"
                  density="300"
                  friction="0.8 0.005 0.0001"
                  rgba="0.8 0.2 0.2 1" />
        </body>
    </worldbody>
</mujocoinclude>
```

**2. Подключите** в `asset/example_scene_y.xml`:

```xml
<include file="./tabletop/object/obj_mybox.xml"/>
```

**3. Если нужна рандомизация** — назовите `body` с префиксом `body_obj_`, тогда `SimpleEnv.reset()` автоматически разместит его случайно:

```python
obj_names = self.env.get_body_names(prefix="body_obj_")
# → ['body_obj_mug_5', 'body_obj_plate_11', 'body_obj_mybox']
```

---

## Параметры `<geom>` — полный разбор

`<geom>` — это одновременно и форма для коллизий, и форма для рендера (если не разделить явно).

### Форма и размер

| Атрибут | Что задаёт | Пример |
|---------|-----------|--------|
| `type` | Тип примитива | `box`, `sphere`, `cylinder`, `capsule`, `plane`, `mesh` |
| `size` | Размер (половины сторон для box, радиус для sphere) | `"0.05 0.05 0.05"` |
| `fromto` | Начало и конец (только для `cylinder`, `capsule`) | `"0 0 0  0 0 0.15"` |
| `pos` | Смещение от центра body | `"0 0 0.01"` |
| `euler` | Поворот в градусах (roll pitch yaw) | `"0 90 0"` |
| `quat` | Поворот в кватернионе (w x y z) | `"0.707 0 0.707 0"` |

> `size` для `box` — это **полуразмеры**: `size="0.05 0.05 0.05"` даёт куб 10×10×10 см.
> Для `sphere` — один радиус: `size="0.04"`.
> Для `cylinder`/`capsule` при использовании `fromto` — только радиус.

### Физические свойства

| Атрибут | Что задаёт | Типичные значения |
|---------|-----------|------------------|
| `density` | Плотность кг/м³ (масса = плотность × объём) | `100`–`1000` (вода = 1000) |
| `mass` | Масса напрямую в кг (вместо density) | `0.1`, `0.5` |
| `friction` | Три коэф. трения: **скольжение, кручение, качение** | `"0.95 0.3 0.1"` (кружка), `"1 0.005 0.0001"` (цилиндр) |

Про `friction` подробнее:
```
friction="скольж  кручение  качение"
          ↑        ↑          ↑
       главное  важно для  редко важно
       трение    винтов
```
Чем выше первое число — тем лучше объект держится в захвате. У кружки `0.95`, у скользкого объекта можно поставить `0.3`.

### Контакты и коллизии

| Атрибут | Что задаёт |
|---------|-----------|
| `solimp` | Параметры контактного импеданса `"dmin dmax width"` — насколько "мягкий" контакт |
| `solref` | Параметры контактного ссылочного поведения `"timeconst dampratio"` — скорость восстановления |
| `contype` | Битовая маска типа коллизии (с чем сталкивается) |
| `conaffinity` | Битовая маска аффинности (с кем может столкнуться) |
| `group` | Группа рендера: `2` = видимый, `3` = невидимый (только коллизии) |

```xml
<!-- Видимый geom (рендер), без коллизий — чисто декоративный -->
<geom type="mesh" mesh="mug_vis" contype="0" conaffinity="0" group="2" material="mug_mat"/>

<!-- Невидимый geom (только коллизии) -->
<geom type="mesh" mesh="mug_coll" group="3" rgba="0 0 0 0"/>
```

Так устроены все объекты в `objaverse/` — у кружки отдельно визуальный меш и 30+ упрощённых collision-мешей.

Для простого примитива можно совместить всё в одном geom (без разделения):

```xml
<geom type="box" size="0.03 0.03 0.03" rgba="0.8 0.2 0.2 1" friction="0.8 0.005 0.0001" density="300"/>
```

### Внешний вид

| Атрибут | Что задаёт | Пример |
|---------|-----------|--------|
| `rgba` | Цвет и прозрачность (R G B A от 0 до 1) | `"0.8 0.2 0.2 1"` |
| `material` | Имя материала из `<asset>` (перекрывает rgba) | `"mug_mat"` |

### Параметры `<body>`

| Атрибут | Что задаёт | Пример |
|---------|-----------|--------|
| `name` | Уникальное имя | `"body_obj_mybox"` |
| `pos` | Положение в координатах родителя (x y z в метрах) | `"0.35 0.0 0.85"` |
| `euler` | Начальная ориентация (roll pitch yaw в градусах) | `"0 0 45"` |
| `quat` | Начальная ориентация кватернионом | `"1 0 0 0"` |

### Совет: `<inertial>` когда нужен?

Если не указать `<inertial>`, MuJoCo вычислит инерцию из `density` и формы geom — это работает для большинства простых случаев. Явно задавайте `<inertial>` только если у объекта сложная форма или неравномерное распределение массы:

```xml
<inertial pos="0 0 0" mass="0.5" diaginertia="0.001 0.001 0.001" />
```

---

## XML-описание сцены

MuJoCo описывает сцену в XML-файле. Структура примерно такая:

```xml
<mujoco model="my_scene">

  <asset>
    <!-- Меши, текстуры, материалы -->
    <mesh name="arm_link" file="arm_link.stl"/>
  </asset>

  <worldbody>
    <!-- Всё что есть в сцене — вложенные body -->

    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="5 5 0.1" pos="0 0 0"/>

    <body name="table" pos="0.3 0 0.5">
      <geom type="box" size="0.3 0.3 0.05"/>
    </body>

    <body name="shoulder_pan" pos="0 0 0.8">
      <joint name="shoulder_pan" type="hinge" axis="0 0 1"
             range="-110 110"/>
      <geom type="mesh" mesh="arm_link"/>

      <!-- Вложенные body = цепочка звеньев руки -->
      <body name="shoulder_lift" pos="0 0 0.1">
        <joint name="shoulder_lift" type="hinge" axis="0 1 0"/>
        ...
      </body>
    </body>

    <body name="body_obj_mug_5" pos="0.3 0 0.82">
      <joint type="free"/>   <!-- свободно летающий объект -->
      <geom type="mesh" mesh="mug"/>
      <site name="bottom_site_mug_5" pos="0 0 -0.04"/>
      <site name="top_site_mug_5"    pos="0 0  0.06"/>
    </body>

  </worldbody>

  <actuator>
    <!-- Каждый actuator управляет одним joint -->
    <position name="shoulder_pan_act" joint="shoulder_pan"
              kp="100" ctrlrange="-1.92 1.92"/>
  </actuator>

  <sensor>
    <framepos name="ee_pos" objtype="body" objname="gripper"/>
  </sensor>

</mujoco>
```

**Главное правило:** иерархия `<body>` определяет кинематическую цепочку. Позиция/ориентация дочернего body задаётся относительно родителя.

---

## Python API — минимальный пример

```python
import mujoco
import numpy as np

# Загрузка модели
model = mujoco.MjModel.from_xml_path("scene.xml")
data  = mujoco.MjData(model)

# Шаг симуляции
mujoco.mj_step(model, data)

# Читаем позицию сустава "shoulder_pan"
joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "shoulder_pan")
angle = data.qpos[model.jnt_qposadr[joint_id]]

# Управляем actuator
act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "shoulder_pan_act")
data.ctrl[act_id] = 0.5  # радианы (для position actuator)

# Получаем позицию тела
body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
pos = data.xpos[body_id]   # numpy array [x, y, z]
rot = data.xmat[body_id]   # матрица поворота 3x3 (row-major)
```

> В нашем проекте всё это обёрнуто в класс `MuJoCoParserClass` (`mujoco_env/mujoco_parser.py`) — вам скорее всего не нужно трогать голое API.

---

## Как устроено в нашем проекте

```
SimpleEnv  (mujoco_env/y_env.py)
    │
    └── MuJoCoParserClass  (mujoco_env/mujoco_parser.py)
            │
            └── mujoco.MjModel / MjData  ← сырое API
```

### `SimpleEnv` — главный класс, с которым вы работаете

```python
from mujoco_env.y_env import SimpleEnv

env = SimpleEnv(
    xml_path="asset/example_scene_y.xml",
    action_type="delta_joint_angle",  # или "joint_angle", "eef_pose"
    state_type="joint_angle",         # или "ee_pose", "delta_q"
    seed=42,
)
```

#### Типы action

| `action_type` | Размер вектора | Что передаёте |
|--------------|---------------|---------------|
| `joint_angle` | 6 | Абсолютные углы суставов [j1..j5, gripper] в радианах |
| `delta_joint_angle` | 6 | Приращения углов [dj1..dj5, gripper] в радианах |
| `eef_pose` | 7 | [dx, dy, dz, droll, dpitch, dyaw, gripper] — IK решается автоматически |

#### Типы state (возвращаемое значение `step()`)

| `state_type` | Размер | Содержимое |
|-------------|--------|------------|
| `joint_angle` | 6 | Текущие углы суставов + gripper |
| `ee_pose` | 6 | [px, py, pz, roll, pitch, yaw] end-effector |
| `delta_q` | 6 | Приращение углов с прошлого шага + gripper |

#### Основные методы

```python
env.reset(seed=0)           # сбросить сцену, рандомизировать объекты
state = env.step(action)    # применить action, вернуть state
env.step_env()              # один шаг физики без применения action
img_agent, img_ego = env.grab_image()  # RGB с двух камер (numpy HxWx3)
env.render()                # показать окно
success = env.check_success()  # True если кружка на тарелке
```

---

## Цикл симуляции — как это работает пошагово

```
┌─────────────────────────────────────────────────────┐
│  while viewer открыт:                               │
│    env.step_env()          ← физика (PD-регулятор)  │
│    if loop_every(fps):     ← ограничение частоты    │
│      action = policy(obs)  ← ваша нейросеть/рука    │
│      obs = env.step(action)                         │
│      imgs = env.grab_image()                        │
│      env.render()                                   │
└─────────────────────────────────────────────────────┘
```

Почему `step_env()` вызывается отдельно? Потому что физика должна крутиться быстро (например 1000 Гц), а запись/управление — медленнее (например 30–50 fps). `step_env()` — это один тик физики; `step(action)` — это высокоуровневая команда, которая обновляет целевые углы.

---

## Суставы SO-101 и их порядок

```
Индекс  Имя             Диапазон (рад)
  0     shoulder_pan    [-1.92,  1.92]   ← поворот основания (влево/вправо)
  1     shoulder_lift   [-1.75,  1.75]   ← подъём плеча
  2     elbow_flex      [-1.69,  1.69]   ← сгиб локтя
  3     wrist_flex      [-1.66,  1.66]   ← сгиб запястья
  4     wrist_roll      [-2.74,  2.84]   ← вращение запястья
  5     gripper         [-0.17,  1.75]   ← 0 = закрыт, ~1.2 = открыт
```

Стартовая поза (в градусах): `[0, -30, 75, -45, 0]` — рука слегка приподнята над столом.


## Полезные ссылки

- [Официальная документация MuJoCo](https://mujoco.readthedocs.io)
- [XML Reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html) — справочник по всем тегам
- [Python Bindings](https://mujoco.readthedocs.io/en/stable/python.html) — все функции Python API
- [Туториал по моделированию](https://mujoco.readthedocs.io/en/stable/modeling.html)
