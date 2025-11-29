import importlib
import os


class EnvSettings:
    def __init__(self):
        test_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

        self.results_path = '{}/tracking_results/'.format(test_path)
        self.result_plot_path = '{}/result_plots/'.format(test_path)
        self.svdataset_path = ''
        self.sv248s_path = ''
        self.satsot_path = ''
        self.viso_path = '' 
        self.prj_dir = ''
        self.save_dir = ''


def env_settings():
    env_module_name = 'lib.test.evaluation.local'
    try:
        env_module = importlib.import_module(env_module_name)
        return env_module.local_env_settings()
    except:
        env_file = os.path.join(os.path.dirname(__file__), 'local.py')

        raise RuntimeError('YOU HAVE NOT SETUP YOUR local.py!!!\n Go to "{}" and set all the paths you need. '
                           'Then try to run again.'.format(env_file))