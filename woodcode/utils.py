import yaml
import os

# Default path for parameter storage
PARAMS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "params"))


def load_params(file_path):
    """Loads parameters from a YAML file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Parameter file not found: {file_path}")

    with open(file_path, "r") as f:
        return yaml.safe_load(f) or {}


def load_all_params(params_dir=None, project_name=None):
    """
    Loads and merges all YAML parameter files from a given directory.

    - Merges all module-specific parameters (general, nwb, spatial_tuning, etc.).
    - If a project-specific YAML exists (e.g., experiment_1.yaml), it overrides module defaults.

    :param params_dir: Directory containing parameter YAML files.
    :param project_name: Optional project-specific configuration to apply overrides.
    :return: Dictionary of merged parameters.
    """
    if params_dir is None:
        params_dir = PARAMS_DIR  # Use default params directory

    params = {}

    # Load all module-specific YAML files
    for filename in os.listdir(params_dir):
        if filename.endswith(".yaml"):
            file_path = os.path.join(params_dir, filename)
            params.update(load_params(file_path))

    # Apply project-specific overrides (if exists)
    if project_name:
        project_file = os.path.join(params_dir, f"{project_name}.yaml")
        if os.path.exists(project_file):
            params.update(load_params(project_file))

    return params


def save_params(file_path, params):
    """Saves the modified parameters back to a YAML file."""
    with open(file_path, "w") as f:
        yaml.dump(params, f, default_flow_style=False)
    print(f"Parameters saved to {file_path}")


def save_project_params(params, project_name, params_dir=None):
    """
    Saves modified parameters to the project-specific YAML file.

    :param params: Dictionary of parameters to save.
    :param project_name: Project name (used to determine the YAML file).
    :param params_dir: Directory to save parameters (defaults to `params/`).
    """
    if params_dir is None:
        params_dir = PARAMS_DIR  # Use default params directory

    project_file = os.path.join(params_dir, f"{project_name}.yaml")
    save_params(project_file, params)
