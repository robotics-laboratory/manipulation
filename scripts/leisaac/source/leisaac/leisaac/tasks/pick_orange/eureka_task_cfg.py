"""IsaacLabEureka task registry entries for LeIsaac PickOrange."""

PICK_ORANGE_EUREKA_TASK_ID = "LeIsaac-SO101-PickOrange-Eureka-Direct-v0"

TASKS_CFG_PATCH = {
    PICK_ORANGE_EUREKA_TASK_ID: {
        "description": (
            "pick three oranges, place them on the plate, and return the SO-101 arm to its rest pose. "
            "The reward should encourage reaching the active unplaced orange, grasping it, lifting it, "
            "moving it above the plate, releasing it on the plate, avoiding disturbance of future oranges, "
            "and completing all three placements."
        ),
        "success_metric": "self._eureka_success_metric(env_ids)",
        "success_metric_to_win": 1.0,
        "success_metric_tolerance": 0.05,
    }
}
