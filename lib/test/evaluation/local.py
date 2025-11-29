from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.
    settings.svdataset_path = '$PROJECT_ROOT$/data'  # Directory for the applied datasets.
    settings.sv248s_path = '$PROJECT_ROOT$/lib/train/data_specs/sv248s/val_split.txt'  # Path for split_file of sv248s_test.
    settings.satsot_path = '$PROJECT_ROOT$/lib/train/data_specs/satsot/val_split.txt'  # Path for split_file of satsot_test.
    settings.viso_path = '$PROJECT_ROOT$/lib/train/data_specs/viso/val_split.txt'  # Path for split_file of viso_test.
    settings.prj_dir = '$PROJECT_ROOT$'  # Base directory for the current project.
    settings.save_dir = '$PROJECT_ROOT$/output'  # Directory for saving outputs.
    settings.result_plot_path = '$PROJECT_ROOT$/output/test/result_plots'  # Where to store evaluation plots.
    settings.results_path = '$PROJECT_ROOT$/output/test/tracking_results'  # Where to store tracking results.
    

    return settings

