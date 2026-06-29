from dataclasses import dataclass
import importlib
import math
import re
from pathlib import Path
from typing import Sequence


DEFAULT_JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
DEFAULT_JOINT_TORQUE_SCALES = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[2] / "assets" / "models" / "piper_grav_comp.xml"
)
PAYLOAD_700G_MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "assets"
    / "models"
    / "piper_grav_comp_midas_hand.xml"
)
MIDAS_HAND_MODEL_PATH = PAYLOAD_700G_MODEL_PATH


@dataclass(frozen=True)
class GravityTorqueResult:
    gravity_torques: list[float]
    damping_torques: list[float]
    friction_comp_torques: list[float]
    output_scales: list[float]
    joint_torque_scales: list[float]
    command_torques: list[float]


def get_default_model_path() -> Path:
    return DEFAULT_MODEL_PATH


def get_payload_700g_model_path() -> Path:
    return PAYLOAD_700G_MODEL_PATH


def get_midas_hand_model_path() -> Path:
    return MIDAS_HAND_MODEL_PATH


def parse_piper_software_version(version: str | None) -> tuple[int, int, int] | None:
    if not version:
        return None
    match = re.search(r"V(\d+)\.(\d+)(?:[-.](\d+))?", version)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2)), int(match.group(3) or 0)


def firmware_is_after_v18_post2(version: str | None) -> bool:
    """Return True for Piper firmware newer than 1.8.post2."""
    parsed = parse_piper_software_version(version)
    if parsed is None:
        return False
    major, minor, patch = parsed
    if major != 1:
        return major > 1
    if minor != 8:
        return minor > 8
    return patch > 2


def direct_torque_output_scales_for_pyagxarm(
    firmware_version: str | None,
    *,
    driver: str = "default",
) -> tuple[float, ...]:
    """Scale MuJoCo torques before sending through pyAgxArm MIT commands.

    Only the ``default`` driver applies an internal 0.25 factor on joints 1-3
    inside ``move_mit``. For firmware <= 1.8.post2 that matches the arm's 4x
    internal amplification, so MuJoCo torques can be passed through unchanged.
    For newer firmware (> 1.8.post2) the arm no longer amplifies, so joints
    1-3 must be pre-multiplied by 4.0 before calling the default driver.

    The version-specific drivers (``v183``, ``v188``) send MIT torques directly
    without the internal 0.25 factor, matching piper_control's direct scaling.
    """
    if driver != "default":
        return (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    if firmware_is_after_v18_post2(firmware_version):
        return (4.0, 4.0, 4.0, 1.0, 1.0, 1.0)
    return (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)


DEFAULT_GRAVITY_M_S2 = 9.81

# MuJoCo model is authored for desk/upright mount (gravity along -Z).
# Side mounts apply the same world gravity in the model base frame.
GRAVITY_VECTOR_BY_INSTALLATION: dict[str, tuple[float, float, float]] = {
    "horizontal": (0.0, 0.0, -DEFAULT_GRAVITY_M_S2),
    "left": (0.0, DEFAULT_GRAVITY_M_S2, 0.0),
    "right": (0.0, -DEFAULT_GRAVITY_M_S2, 0.0),
}


def gravity_vector_for_installation(
    installation_pos: str | None,
) -> tuple[float, float, float]:
    if installation_pos is None or installation_pos == "unchanged":
        return GRAVITY_VECTOR_BY_INSTALLATION["horizontal"]
    key = installation_pos.strip().lower()
    if key not in GRAVITY_VECTOR_BY_INSTALLATION:
        raise ValueError(
            "installation_pos must be 'horizontal', 'left', 'right', or 'unchanged'"
        )
    return GRAVITY_VECTOR_BY_INSTALLATION[key]


def parse_gravity_vector(values: Sequence[float]) -> tuple[float, float, float]:
    vector = _as_float_list(values, name="gravity", length=3)
    if all(abs(component) < 1e-9 for component in vector):
        raise ValueError("gravity vector must be non-zero")
    return (vector[0], vector[1], vector[2])


def _as_float_list(values: Sequence[float], *, name: str, length: int) -> list[float]:
    result = [float(value) for value in values]
    if len(result) != length:
        raise ValueError(f"{name} must contain exactly {length} values")
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must only contain finite values")
    return result


def _coerce_per_joint(
    value: Sequence[float] | float | None, *, name: str, length: int = 6
) -> list[float]:
    """Accept None (->zeros), a scalar (broadcast), or a length-6 sequence."""
    if value is None:
        return [0.0] * length
    if isinstance(value, (int, float)):
        scalar = float(value)
        if not math.isfinite(scalar) or scalar < 0:
            raise ValueError(f"{name} must be finite and non-negative")
        return [scalar] * length
    result = _as_float_list(value, name=name, length=length)
    if any(v < 0 for v in result):
        raise ValueError(f"{name} values must be non-negative")
    return result


def _deadband_sign(velocity: float, deadband: float) -> float:
    """sign(velocity), but 0 inside +/- deadband to avoid chatter at rest."""
    if velocity > deadband:
        return 1.0
    if velocity < -deadband:
        return -1.0
    return 0.0


class MujocoGravityCompensator:
    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        joint_names: Sequence[str] = DEFAULT_JOINT_NAMES,
        firmware_version: str | None = None,
        driver: str = "default",
        gravity: Sequence[float] | None = None,
        installation_pos: str | None = None,
    ):
        try:
            mujoco = importlib.import_module("mujoco")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "MuJoCo is required for gravity compensation. Install it with "
                "`pip install mujoco` or `pip install piper_control[gravity]`."
            ) from exc

        self.mujoco = mujoco
        self.model_path = Path(model_path or get_default_model_path()).expanduser()
        self.joint_names = tuple(joint_names)
        if len(self.joint_names) != 6:
            raise ValueError("joint_names must contain exactly 6 names")

        self.driver = driver
        self.firmware_version = firmware_version
        self.output_scales = list(
            direct_torque_output_scales_for_pyagxarm(
                firmware_version, driver=driver
            )
        )

        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        if gravity is not None:
            self.set_gravity(gravity)
        elif installation_pos is not None:
            self.set_gravity(gravity_vector_for_installation(installation_pos))
        self._qpos_indices: list[int] = []
        self._qvel_indices: list[int] = []
        self._validate_joint_map()

    def set_firmware_version(
        self, firmware_version: str | None, *, driver: str | None = None
    ) -> None:
        self.firmware_version = firmware_version
        if driver is not None:
            self.driver = driver
        self.output_scales = list(
            direct_torque_output_scales_for_pyagxarm(
                firmware_version, driver=self.driver
            )
        )

    def set_gravity(self, gravity: Sequence[float]) -> tuple[float, float, float]:
        vector = parse_gravity_vector(gravity)
        self.model.opt.gravity[:] = vector
        return vector

    def get_gravity(self) -> tuple[float, float, float]:
        return tuple(float(value) for value in self.model.opt.gravity)

    def set_installation_gravity(self, installation_pos: str | None) -> None:
        self.set_gravity(gravity_vector_for_installation(installation_pos))

    def _validate_joint_map(self) -> None:
        for joint_name in self.joint_names:
            joint_id = self.mujoco.mj_name2id(
                self.model, self.mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            if joint_id < 0:
                raise ValueError(
                    f"MuJoCo model {self.model_path} does not contain joint {joint_name!r}"
                )

            qpos_index = int(self.model.jnt_qposadr[joint_id])
            qvel_index = int(self.model.jnt_dofadr[joint_id])
            joint_type = int(self.model.jnt_type[joint_id])
            if joint_type not in (
                int(self.mujoco.mjtJoint.mjJNT_HINGE),
                int(self.mujoco.mjtJoint.mjJNT_SLIDE),
            ):
                raise ValueError(f"Joint {joint_name!r} must be a single-DoF joint")
            self._qpos_indices.append(qpos_index)
            self._qvel_indices.append(qvel_index)

    def compute(
        self,
        joint_positions: Sequence[float],
        *,
        joint_velocities: Sequence[float] | None = None,
        damping: float = 0.0,
        torque_scale: float = 1.0,
        joint_torque_scales: Sequence[float] | None = None,
        coulomb_comp: Sequence[float] | float | None = None,
        viscous_comp: Sequence[float] | float | None = None,
        comp_velocity_deadband: float = 0.0,
    ) -> GravityTorqueResult:
        positions = _as_float_list(joint_positions, name="joint_positions", length=6)
        velocities = (
            [0.0] * 6
            if joint_velocities is None
            else _as_float_list(joint_velocities, name="joint_velocities", length=6)
        )
        per_joint_scales = (
            list(DEFAULT_JOINT_TORQUE_SCALES)
            if joint_torque_scales is None
            else _as_float_list(
                joint_torque_scales, name="joint_torque_scales", length=6
            )
        )
        if any(scale < 0 or not math.isfinite(scale) for scale in per_joint_scales):
            raise ValueError("joint_torque_scales must be finite and non-negative")
        coulomb = _coerce_per_joint(coulomb_comp, name="coulomb_comp")
        viscous = _coerce_per_joint(viscous_comp, name="viscous_comp")

        # qvel is left at zero so qfrc_bias is the pure gravity term g(q); we do
        # not feed Coriolis/centrifugal C(q,v)v into a hand-drag command.
        self.data.qpos[:] = self.model.qpos0
        self.data.qvel[:] = 0.0
        self.data.qacc[:] = 0.0

        for index, value in zip(self._qpos_indices, positions):
            self.data.qpos[index] = value

        self.mujoco.mj_forward(self.model, self.data)

        gravity_torques = [
            float(self.data.qfrc_bias[index]) for index in self._qvel_indices
        ]
        damping_torques = [-float(damping) * velocity for velocity in velocities]
        # Friction compensation pushes *with* the motion (opposite sign to damping)
        # to cancel joint friction and make the arm feel lighter, like teach mode.
        friction_comp_torques = [
            (fc * _deadband_sign(velocity, comp_velocity_deadband)) + (fv * velocity)
            for fc, fv, velocity in zip(coulomb, viscous, velocities)
        ]
        combined = [
            gravity + damping_torque + friction_torque
            for gravity, damping_torque, friction_torque in zip(
                gravity_torques, damping_torques, friction_comp_torques
            )
        ]
        command_torques = [
            float(torque_scale) * joint_scale * firmware_scale * torque
            for joint_scale, firmware_scale, torque in zip(
                per_joint_scales, self.output_scales, combined
            )
        ]

        return GravityTorqueResult(
            gravity_torques=gravity_torques,
            damping_torques=damping_torques,
            friction_comp_torques=friction_comp_torques,
            output_scales=list(self.output_scales),
            joint_torque_scales=per_joint_scales,
            command_torques=command_torques,
        )
