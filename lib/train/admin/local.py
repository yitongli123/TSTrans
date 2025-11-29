class EnvironmentSettings:
    def __init__(self):
        # Set your local paths here.
        self.workspace_dir = '$PROJECT_ROOT$'    # Base directory for the current project.
        self.tensorboard_dir = '$PROJECT_ROOT$/tensorboard'    # Directory for tensorboard files.
        self.svdata_dir = '$PROJECT_ROOT$/data'  # Directory for the applied datasets.
      
