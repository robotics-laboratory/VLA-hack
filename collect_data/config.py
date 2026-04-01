from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class CollectDataConfig:
    # Seed для воспроизводимости раскладки объектов в сцене.
    seed: int | None = 0
    # Имя датасета в терминах LeRobot.
    repo_name: str = "so101_pnp"
    # Сколько эпизодов нужно записать за сессию.
    num_demo: int = 20
    # Локальная папка, куда будет сохранён датасет.
    root: Path = Path("./simulation_data_fixed")
    # True: управление с leader-arm, False: управление с клавиатуры.
    use_master_arm: bool = True
    # Серийный порт leader-arm.
    leader_port: str = "/dev/ttyACM0"
    # Идентификатор leader-arm для master_arm_control.
    leader_id: str = "my_leader"
    # Порог движения, после которого автоматически стартует запись.
    motion_threshold: float = 0.03
    # Текстовое имя задачи, записывается в поле task.
    task_name: str = "Put cube on plate"
    # MuJoCo XML-сцена с роботом и объектами.
    xml_path: str = "./asset/example_scene_y.xml"
    # Частота записи датасета в кадрах в секунду.
    fps: int = 10
    # Размер кадров после resize перед записью в датасет.
    image_size: tuple[int, int] = (640, 480)  # (width, height) для PIL.resize
    # Параметры видеокодирования/записи LeRobot.
    image_writer_threads: int = 10
    image_writer_processes: int = 5
    batch_encoding_size: int = 1
    vcodec: str = "h264"
    metadata_buffer_size: int = 10
    streaming_encoding: bool = True
    encoder_threads: int | None = None
    # Зафиксированный контракт состояния для совместимости с so_follower.
    state_contract: str = "joint_pos"

    def __post_init__(self) -> None:
        if self.state_contract != "joint_pos":
            raise ValueError(
                "CollectDataConfig.state_contract поддерживает только canonical значение "
                "'joint_pos' (совместимо с lerobot-record / so_follower)."
            )

    @property
    def action_type(self) -> str:
        # Для master-arm пишем абсолютные joint angles, для клавиатуры - delta.
        return "joint_angle" if self.use_master_arm else "delta_joint_angle"


def default_config() -> CollectDataConfig:
    return CollectDataConfig()
