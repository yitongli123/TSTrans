import matplotlib.pyplot as plt
import argparse
import os
plt.rcParams['figure.figsize'] = [8, 8]
import sys

prj_path = os.path.join(os.getcwd())
if prj_path not in sys.path:
    sys.path.append(prj_path)
print('System paths:', sys.path)

from lib.test.analysis.plot_results import plot_results, print_results
from lib.test.evaluation import get_dataset, trackerlist


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze metrics on tracking results.')
    parser.add_argument('--tracking_results', type=str, default='./output/test/tracking_results/tstrans/vitb_256_mae_ce_ep600', help='Data path of tracking results')
    parser.add_argument('--analyzed_split', type=str, default='sv248s_test', help='Name of the analyzed dataset.')
    args = parser.parse_args()

    trackers = []
    dataset_name = args.analyzed_split
    """tstrans"""
    trackers.extend(trackerlist(name='TSTrans', parameter_name='vitb_256_mae_ce_ep600', dataset_name=dataset_name,
                                run_ids=None, display_name='TSTrans'))
    trackers[-1].results_dir = [args.tracking_results]
    

    dataset = get_dataset(dataset_name)
    plot_results(trackers, dataset, dataset_name, merge_results=False, plot_types=('success', 'prec'),
                skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05)
    print_results(trackers, dataset, dataset_name, merge_results=False, plot_types=('success', 'prec', 'fps'), force_evaluation=True)
