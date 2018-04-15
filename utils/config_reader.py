import json
import pathlib
import os

DEFAULT_EXPT_PATH = pathlib.Path("../expts")


def process_config(json_file_name):

    if not os.path.exists(json_file_name):
        raise RuntimeError('Config path "{}" is invalid!'.format(json_file_name))
    with open(json_file_name, "r") as f:
        config_dict = json.load(f)

    # Experiment name
    path = pathlib.Path(json_file_name)
    file_name = path.name
    config_dict["expt_name"] = str(path.stem)

    # Summary and checkpoint dirs
    expt_path = DEFAULT_EXPT_PATH / config_dict["expt_name"]
    config_dict["summary_dir"] = str(expt_path / "summary")
    config_dict["checkpoint_dir"] = str(expt_path / "checkpoint")
    config_dict["checkpoint_full_path"] = os.path.join(config_dict["checkpoint_dir"], "ckpt")

    return config_dict