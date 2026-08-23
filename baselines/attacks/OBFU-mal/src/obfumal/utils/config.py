import configparser
import os
from typing import Dict, Any


def apply_ini_overrides(config: Dict[str, Any], ini_path: str = "config.ini") -> Dict[str, Any]:
    """
    Optional overrides for paths in config dict using config.ini.
    This does not alter algorithmic logic; it only redirects I/O paths.
    """
    if not os.path.exists(ini_path):
        return config

    parser = configparser.ConfigParser()
    parser.read(ini_path)

    if "paths" in parser:
        paths = parser["paths"]
        if "sample_dir" in paths:
            config.setdefault("env", {})["sample_dir"] = paths.get("sample_dir")
        if "output_dir" in paths:
            config.setdefault("env", {})["output_dir"] = paths.get("output_dir")
        if "upx_path" in paths:
            config.setdefault("env", {})["upx_path"] = paths.get("upx_path")
        if "darkarmour_path" in paths:
            config.setdefault("env", {})["darkarmour_path"] = paths.get("darkarmour_path")
        if "lightgbm_model" in paths:
            config.setdefault("detector", {})["model_path"] = paths.get("lightgbm_model")
        if "dqn_model" in paths:
            config.setdefault("training", {})["save_path"] = paths.get("dqn_model")
        if "runs_dir" in paths:
            config.setdefault("training", {})["run_dir"] = paths.get("runs_dir")
            config.setdefault("evaluation", {})["run_dir"] = paths.get("runs_dir")

    return config
