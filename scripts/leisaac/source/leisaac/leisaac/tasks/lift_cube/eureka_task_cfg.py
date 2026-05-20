"""IsaacLabEureka task registry entries for LeIsaac LiftCube."""

LIFT_CUBE_EUREKA_TASK_ID = "LeIsaac-SO101-LiftCube-Eureka-Direct-v0"

TASKS_CFG_PATCH = {
    LIFT_CUBE_EUREKA_TASK_ID: {
        "description": (
            "lift the red cube from the table with the SO-101 arm in a stable, human-like motion. "
            "The reward should encourage reaching and grasping the cube, lifting it above the base-height "
            "success threshold, keeping the cube stable while lifted, avoiding wrist-flip exploits, and "
            "finishing with a controlled hold at the target height."
        ),
        "success_metric": "self._eureka_success_metric(env_ids)",
        "success_metric_to_win": 1.0,
        "success_metric_tolerance": 0.05,
    }
}
