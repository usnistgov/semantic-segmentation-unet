# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import json
import numpy as np
import pandas as pd
import sklearn.metrics
import logging

logger = logging.getLogger()


class TrainingStats():
    def __init__(self):
        self.epoch_data = list()
        self.global_data = dict()
        self.metric_names = list()

    def add(self, epoch: int, metric_name: str, value):
        while len(self.epoch_data) <= epoch:
            self.epoch_data.append(dict())

        self.epoch_data[epoch]['epoch'] = epoch
        self.epoch_data[epoch][metric_name] = value

        logger.info('{}: {}'.format(metric_name, value))

    def update_global(self, epoch):
        if epoch > len(self.epoch_data):
            raise RuntimeError('Missing data at epoch {} in epoch stats'.format(epoch))

        epoch_data = self.epoch_data[epoch]
        for k, v in epoch_data.items():
            self.add_global(k, v)

    def add_global(self, metric_name: str, value):
        self.global_data[metric_name] = value

    def get(self, metric_name: str):
        data = list()
        for epoch_metrics in self.epoch_data:
            if metric_name not in epoch_metrics.keys():
                raise RuntimeError('Missing data for metric "{}" in epoch stats'.format(metric_name))
                # data.append(None)  # use this if you want to silently fail
            else:
                data.append(epoch_metrics[metric_name])
        return data

    def get_epoch(self, metric_name: str, epoch: int):
        if epoch > len(self.epoch_data):
            raise RuntimeError('Missing data for metric "{}" at epoch {} in epoch stats'.format(metric_name, epoch))
            # return None  # use this if you want to silently fail

        epoch_data = self.epoch_data[epoch]
        if metric_name not in epoch_data.keys():
            raise RuntimeError('Missing data for metric "{}" at epoch {} in epoch stats'.format(metric_name, epoch))
            # return None  # use this if you want to silently fail

        return epoch_data[metric_name]

    def get_global(self, metric_name: str):
        if metric_name not in self.global_data.keys():
            raise RuntimeError('Missing data for metric "{}" in global stats'.format(metric_name))
            # return None  # use this if you want to silently fail
        return self.global_data[metric_name]

    def export(self, output_folder: str, plot_flag=True):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # convert self.epoch_data into pandas dataframe
        df = pd.DataFrame(self.epoch_data)
        df.to_csv(os.path.join(output_folder, 'detailed_stats.csv'), index=False, encoding="ascii")

        if plot_flag:
            # plot all metrics if its useful (its usually not)
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(8, 4), dpi=200)
            for col in df.columns:
                if col == 'epoch':
                    continue  # don't plot epochs against itself
                try:
                    plt.clf()
                    ax = plt.gca()
                    if 'confusion' in col:
                        cm = self.epoch_data[-1][col]
                        disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
                        disp.plot()
                    else:
                        x = df['epoch'].to_list()
                        y = df[col].to_list()
                        ax.plot(x, y, 'o-', markersize=5, linewidth=1)
                        ax.set_xlabel('Epoch')
                        ax.set_ylabel(col)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_folder, '{}.png'.format(col)))
                except Exception as e:
                    pass
            plt.close(fig)

        for key in self.global_data.keys():
            val = self.global_data[key]
            if isinstance(val, np.ndarray):
                self.global_data[key] = val.tolist()

        with open(os.path.join(output_folder, 'stats.json'), 'w') as fh:
            json.dump(self.global_data, fh, ensure_ascii=True, indent=2)
