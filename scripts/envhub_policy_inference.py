import torch
from lerobot.envs.factory import make_env
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.utils import build_inference_frame

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_id = "igor-saprygin/smolvla-leisaac-merged"
policy = SmolVLAPolicy.from_pretrained(model_id).to(device)
policy.eval()

preprocess, postprocess = make_pre_post_processors(
    policy.config,
    model_id,
    preprocessor_overrides={"device_processor": {"device": str(device)}},
)

# full dataset-style feature spec for raw env observations
ds_features = {
    "observation.state": {
        "type": "STATE",
        "dtype": "float32",
        "shape": [6],
        "names": [
            "shoulder_pan.pos",
            "shoulder_lift.pos",
            "elbow_flex.pos",
            "wrist_flex.pos",
            "wrist_roll.pos",
            "gripper.pos",
        ],
    },
    "observation.images.camera1": {
        "type": "VISUAL",
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channels"],
    },
    "observation.images.camera2": {
        "type": "VISUAL",
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channels"],
    },
    "observation.images.camera3": {
        "type": "VISUAL",
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channels"],
    },
    "action": {
        "type": "ACTION",
        "dtype": "float32",
        "shape": [6],
        "names": [
            "shoulder_pan.pos",
            "shoulder_lift.pos",
            "elbow_flex.pos",
            "wrist_flex.pos",
            "wrist_roll.pos",
            "gripper.pos",
        ],
    },
}

envs_dict = make_env(
    "LightwheelAI/leisaac_env:envs/so101_pick_orange.py",
    n_envs=1,
    trust_remote_code=True,
)

suite_name = next(iter(envs_dict))
sync_vector_env = envs_dict[suite_name][0]
env = sync_vector_env.envs[0].unwrapped

obs, info = env.reset()

while True:
    policy_dict = obs["policy"]

    front = policy_dict["front"][0]
    wrist = policy_dict["wrist"][0]
    camera3 = torch.zeros_like(front)

    joint_pos = policy_dict["joint_pos"][0].detach().cpu().numpy()
    state_dict = {
        "shoulder_pan.pos": float(joint_pos[0]),
        "shoulder_lift.pos": float(joint_pos[1]),
        "elbow_flex.pos": float(joint_pos[2]),
        "wrist_flex.pos": float(joint_pos[3]),
        "wrist_roll.pos": float(joint_pos[4]),
        "gripper.pos": float(joint_pos[5]),
    }

    policy_obs = {
        "observation.images.camera1": front,
        "observation.images.camera2": wrist,
        "observation.images.camera3": camera3,
        "observation.state": state_dict,
    }

    obs_frame = build_inference_frame(
        observation=policy_obs,
        ds_features=ds_features,
        device=device,
        task="",
        robot_type="",
    )

    model_input = preprocess(obs_frame)

    with torch.no_grad():
        action = policy.select_action(model_input)

    action = postprocess(action)

    if action.ndim == 1:
        action = action.unsqueeze(0)

    obs, reward, terminated, truncated, info = env.step(action)

    if bool(terminated) or bool(truncated):
        obs, info = env.reset()