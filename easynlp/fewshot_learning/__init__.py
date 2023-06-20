# coding=utf-8
# Copyright (c) 2020 Alibaba PAI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .fewshot_application import CPTClassification, FewshotClassification
from .fewshot_dataset import FewshotBaseDataset
from .fewshot_evaluator import CPTEvaluator, PromptEvaluator
from .fewshot_predictor import (CPTPredictor, FewshotPyModelPredictor,
                                PromptPredictor)


"""
要实现一个层次分类的, 我估计这些类都要重写一遍, 好坑. 还要改下训练器

- [PaddleNLP基于ERNIR3.0文本分类: WOS数据集为例(层次分类)](https://developer.aliyun.com/article/1065692)
- [手把手搭建一个语义检索系统](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/applications/neural_search)
- [超多数据集](https://segmentfault.com/a/1190000042487495)

PaddleNLP 还是很不错的, 可以看看.
"""
