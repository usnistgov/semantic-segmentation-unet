# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import json
import numpy as np
import pandas as pd
import logging


class TrainingStats():
    def __init__(self):
        self.epoch_data = list()
        self.global_data = dict()
        self.best_epoch = None
        self.accumulator = dict()

    def append_accumulate(self, metric_name: str, value):
        if metric_name not in self.accumulator.keys():
            self.accumulator[metric_name] = list()
        self.accumulator[metric_name].append(value)

    def close_accumulate(self, epoch, metric_name: str, method: str = 'mean', default_value=np.nan):
        if metric_name not in self.accumulator.keys():
            # metric is missing, add the default value
            self.add(epoch, metric_name, default_value)
            return

        if method == 'mean' or method == 'avg' or method == 'average':
            value = float(np.mean(self.accumulator[metric_name]))
        elif method == 'sum':
            value = float(np.sum(self.accumulator[metric_name]))
        elif method == 'median':
            value = float(np.median(self.accumulator[metric_name]))
        else:
            raise RuntimeError("Invalid accumulation method: {}".format(method))

        self.add(epoch, metric_name, value)
        del self.accumulator[metric_name]

    def add(self, epoch: int, metric_name: str, value):
        while len(self.epoch_data) <= epoch:
            self.epoch_data.append(dict())

        self.epoch_data[epoch]['epoch'] = epoch
        self.epoch_data[epoch][metric_name] = value

        logging.info('{}: {}'.format(metric_name, value))

    def update_global(self, epoch):
        if epoch > len(self.epoch_data):
            raise RuntimeError('Missing data at epoch {} in epoch stats'.format(epoch))

        self.best_epoch = epoch
        epoch_data = self.epoch_data[epoch]
        for k, v in epoch_data.items():
            self.add_global(k, v)

    def add_global(self, metric_name: str, value):
        self.global_data[metric_name] = value

    def get(self, metric_name: str, aggregator=None):
        data = list()
        for epoch_metrics in self.epoch_data:
            if metric_name not in epoch_metrics.keys():
                # raise RuntimeError('Missing data for metric "{}" in epoch stats'.format(metric_name))
                data.append(None)  # use this if you want to silently fail
            else:
                data.append(epoch_metrics[metric_name])
        if aggregator is not None:
            data = np.asarray(data, dtype=float)  # by force cast to float, None becomes nan
            if aggregator == 'mean':
                data = np.nanmean(data)
            elif aggregator == 'median':
                data = np.nanmedian(data)
            elif aggregator == 'sum':
                data = np.nansum(data)
            else:
                raise RuntimeError('Invalid aggregator: {}'.format(aggregator))
        return data

    def get_epoch(self, metric_name: str, epoch: int):
        if epoch > len(self.epoch_data) or epoch < 0:
            # raise RuntimeError('Missing data for metric "{}" at epoch {} in epoch stats'.format(metric_name, epoch))
            return None  # use this if you want to silently fail

        epoch_data = self.epoch_data[epoch]
        if metric_name not in epoch_data.keys():
            # raise RuntimeError('Missing data for metric "{}" at epoch {} in epoch stats'.format(metric_name, epoch))
            return None  # use this if you want to silently fail

        return epoch_data[metric_name]

    def get_global(self, metric_name: str):
        if metric_name not in self.global_data.keys():
            # raise RuntimeError('Missing data for metric "{}" in global stats'.format(metric_name))
            return None  # use this if you want to silently fail
        return self.global_data[metric_name]

    def render_and_save_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, output_folder: str, metric_name: str, epoch: int = None):
        from matplotlib import pyplot as plt
        import sklearn.metrics

        ofldr = os.path.join(output_folder, 'confusion_matrix')
        if not os.path.exists(ofldr):
            os.makedirs(ofldr)

        sklearn.metrics.ConfusionMatrixDisplay.from_predictions(y_true, y_pred, colorbar=False)
        fig = plt.gcf()
        fig.set_size_inches(10, 10)
        plt.savefig(os.path.join(ofldr, "epoch{:04d}_".format(epoch) + metric_name + ".png"))
        plt.close(fig)

        sklearn.metrics.ConfusionMatrixDisplay.from_predictions(y_true, y_pred, values_format='.2g', normalize='true', colorbar=False)
        fig = plt.gcf()
        fig.set_size_inches(10, 10)
        plt.savefig(os.path.join(ofldr, "epoch{:04d}_".format(epoch) + metric_name + "_norm" + ".png"))
        plt.close(fig)

    def plot_all_metrics(self, output_dirpath: str, all_one_figure: bool = False):
        from matplotlib import pyplot as plt

        df = pd.DataFrame(self.epoch_data)
        col_list = list(df.columns)

        fig = plt.figure(figsize=(8, 4), dpi=200)

        for cm in {'loss', 'accuracy', 'wall_time'}:
            # core_metrics = ['train_{}'.format(cm),'val_{}'.format(cm),'test_{}'.format(cm)]
            # core_metrics = [a for a in core_metrics if a in col_list]

            core_metrics = [a for a in col_list if a.endswith(cm)]
            [col_list.remove(m) for m in core_metrics]
            # plot the loss curves

            if len(core_metrics) > 0:
                if not all_one_figure:
                    plt.clf()
                ax = plt.gca()
                for col in core_metrics:
                    x = df['epoch'].to_list()
                    y = df[col].to_list()
                    ax.plot(x, y, '-')
                ax.legend(core_metrics)

                # plot the best indicator after the legend, so that it does not show up in the legend
                if self.best_epoch is not None:
                    for col in core_metrics:
                        y = df[col].to_list()
                        y1_best = y[self.best_epoch]
                        plt.plot(self.best_epoch, y1_best, marker='*', c='k')

                ax.set_xlabel('Epoch')
                ax.set_ylabel('{}'.format(cm))
                plt.tight_layout()
                if not all_one_figure:
                    plt.savefig(os.path.join(output_dirpath, '{}.png'.format(cm)))

        # plot all metrics if its useful (its usually not)
        for col in col_list:
            if col == 'epoch':
                continue  # don't plot epochs against itself
            if not all_one_figure:
                plt.clf()
            ax = plt.gca()
            x = df['epoch'].to_list()
            y = df[col].to_list()
            ax.plot(x, y, 'o-', markersize=5, linewidth=1)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(col)
            plt.tight_layout()
            if not all_one_figure:
                plt.savefig(os.path.join(output_dirpath, '{}.png'.format(col)))
        if all_one_figure:
            plt.savefig(os.path.join(output_dirpath, 'all-plots.png'))
        plt.close(fig)

    def export(self, output_folder: str):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # convert self.epoch_data into pandas dataframe
        df = pd.DataFrame(self.epoch_data)
        df.to_csv(os.path.join(output_folder, 'detailed_stats.csv'), index=False, encoding="ascii")

        # # Code to serialize numpy arrays in a more readable format (instead of using jsonpickle)
        # class NumpyArrayEncoder(json.JSONEncoder):
        #     def default(self, obj):
        #         if isinstance(obj, np.ndarray):
        #             return obj.tolist()
        #         return json.JSONEncoder.default(self, obj)
        #
        # with open(os.path.join(output_folder, 'stats.json'), 'w') as fh:
        #     json.dump(self.global_data, fh, ensure_ascii=True, indent=2, cls=NumpyArrayEncoder)

        with open(os.path.join(output_folder, 'stats.json'), 'w') as fh:
            json.dump(self.global_data, fh, ensure_ascii=True, indent=2)


