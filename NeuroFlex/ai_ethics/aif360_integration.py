# MIT License
# 
# Copyright (c) 2024 VishwamAI
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.inprocessing import PrejudiceRemover
from aif360.algorithms.postprocessing import EqOddsPostprocessing
import pandas as pd
import numpy as np

class AIF360Integration:
    def __init__(self):
        self.dataset = None
        self.privileged_groups = None
        self.unprivileged_groups = None

    def load_dataset(self, df, label_name, favorable_classes, protected_attribute_names, privileged_classes):
        """
        Load dataset into AIF360 BinaryLabelDataset format.
        """
        self.dataset = BinaryLabelDataset(df=df,
                                          label_names=[label_name],
                                          protected_attribute_names=protected_attribute_names,
                                          privileged_protected_attributes=privileged_classes,
                                          favorable_label=favorable_classes[0],
                                          unfavorable_label=favorable_classes[1])

        self.privileged_groups = [{protected_attribute_names[i]: privileged_classes[i][0]}
                                  for i in range(len(protected_attribute_names))]
        self.unprivileged_groups = [{protected_attribute_names[i]: privileged_classes[i][-1]}
                                    for i in range(len(protected_attribute_names))]

    def compute_metrics(self):
        """
        Compute fairness metrics.
        """
        metrics = BinaryLabelDatasetMetric(self.dataset,
                                           unprivileged_groups=self.unprivileged_groups,
                                           privileged_groups=self.privileged_groups)

        return {
            'disparate_impact': metrics.disparate_impact(),
            'statistical_parity_difference': metrics.statistical_parity_difference()
        }

    def mitigate_bias(self, method='reweighing'):
        """
        Apply bias mitigation technique.
        """
        if method == 'reweighing':
            RW = Reweighing(unprivileged_groups=self.unprivileged_groups,
                            privileged_groups=self.privileged_groups)
            return RW.fit_transform(self.dataset)
        elif method == 'prejudice_remover':
            PR = PrejudiceRemover(sensitive_attr=self.dataset.protected_attribute_names[0], eta=25.0)
            return PR.fit_transform(self.dataset)
        elif method == 'equalized_odds':
            EO = EqOddsPostprocessing(unprivileged_groups=self.unprivileged_groups,
                                      privileged_groups=self.privileged_groups)
            return EO.fit_transform(self.dataset, self.dataset)
        else:
            raise ValueError("Unsupported bias mitigation method")

    def evaluate_fairness(self, original_metrics, mitigated_metrics):
        """
        Compare fairness metrics before and after mitigation.
        """
        evaluation = {}
        for metric in original_metrics:
            evaluation[metric] = {
                'before': original_metrics[metric],
                'after': mitigated_metrics[metric],
                'improvement': mitigated_metrics[metric] - original_metrics[metric]
            }
        return evaluation
