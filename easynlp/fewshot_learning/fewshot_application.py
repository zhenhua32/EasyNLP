# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team and The Alibaba PAI team.
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

import json
import os
import traceback

import torch
from torch import nn

from easynlp.appzoo.application import Application
from easynlp.modelzoo import AutoConfig, AutoModelForMaskedLM, AutoTokenizer
from easynlp.utils import io
from easynlp.utils.logger import logger
from easynlp.utils.losses import cross_entropy


class FewshotClassification(Application):
    """An application class for supporting fewshot learning (PET and P-tuning)."""

    def __init__(self, pretrained_model_name_or_path=None, user_defined_parameters=None, **kwargs):
        super(FewshotClassification, self).__init__()
        if kwargs.get("from_config"):
            self.config = kwargs.get("from_config")
            # 没注意到, 模型是 AutoModelForMaskedLM
            self.backbone = AutoModelForMaskedLM.from_config(self.config)
        # for pretrained model, initialize from the pretrained model
        else:
            self.config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
            self.backbone = AutoModelForMaskedLM.from_pretrained(pretrained_model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)

        # if p-tuning, model's embeddings should be changed
        try:
            self.user_defined_parameters_dict = user_defined_parameters.get("app_parameters")
        except KeyError:
            traceback.print_exc()
            exit(-1)
        pattern = self.user_defined_parameters_dict.get("pattern")
        assert pattern is not None, "You must define the pattern for PET learning"
        pattern_list = pattern.split(",")
        cnt = 0
        for i in range(len(pattern_list)):
            if pattern_list[i] == "<pseudo>":
                pattern_list[i] = "<pseudo-%d>" % cnt
                cnt += 1
        if cnt > 0:
            self.tokenizer.add_tokens(["<pseudo-%d>" % i for i in range(cnt)])
        # 根据新的词汇表大小, 调整 embedding 的大小
        self.backbone.resize_token_embeddings(len(self.tokenizer))
        print("embedding size: %d" % len(self.tokenizer))
        self.config.vocab_size = len(self.tokenizer)

    def forward(self, inputs):
        """
        看看前向传播, 这个就是 MLM 任务, 普通的预训练模型
        """
        if "mask_span_indices" in inputs:
            inputs.pop("mask_span_indices")
        outputs = self.backbone(**inputs)
        return {"logits": outputs.logits}

    def compute_loss(self, forward_outputs, label_ids):
        """
        计算损失
        """
        prediction_scores = forward_outputs["logits"]
        # logits 的 shape 是 (batch_size, seq_len, vocab_size), labels 的 shape 是 (batch_size, seq_len)
        # 现在变成 (batch_size * seq_len, vocab_size), (batch_size * seq_len)
        # 就是要找出概率最高的那个词, 然后计算交叉熵损失
        masked_lm_loss = cross_entropy(prediction_scores.view(-1, self.config.vocab_size), label_ids.view(-1))
        return {"loss": masked_lm_loss}

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        # Instantiate model
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        model = cls(pretrained_model_name_or_path=pretrained_model_name_or_path, from_config=config, **kwargs)
        state_dict = None
        weights_path = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
        if not io.exists(weights_path):
            return model
        with io.open(weights_path, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")

        # Load from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if "gamma" in key:
                new_key = key.replace("gamma", "weight")
            if "beta" in key:
                new_key = key.replace("beta", "bias")
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, "_metadata", None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=""):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs
            )
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + ".")

        start_prefix = ""
        logger.info("Loading model...")
        load(model, prefix=start_prefix)
        logger.info("Load finished!")
        if len(missing_keys) > 0:
            logger.info(
                "Weights of {} not initialized from pretrained model: {}".format(model.__class__.__name__, missing_keys)
            )
        if len(unexpected_keys) > 0:
            logger.info(
                "Weights from pretrained model not used in {}: {}".format(model.__class__.__name__, unexpected_keys)
            )
        if len(error_msgs) > 0:
            raise RuntimeError(
                "Error(s) in loading state_dict for {}:\n\t{}".format(model.__class__.__name__, "\n\t".join(error_msgs))
            )
        return model


class FewshotMultiLayerClassification(Application):
    """
    多层次分类
    """

    def __init__(self, pretrained_model_name_or_path=None, user_defined_parameters=None, **kwargs):
        super(FewshotClassification, self).__init__()
        if kwargs.get("from_config"):
            self.config = kwargs.get("from_config")
            # 没注意到, 模型是 AutoModelForMaskedLM
            self.backbone = AutoModelForMaskedLM.from_config(self.config)
        # for pretrained model, initialize from the pretrained model
        else:
            self.config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
            self.backbone = AutoModelForMaskedLM.from_pretrained(pretrained_model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)

        # if p-tuning, model's embeddings should be changed
        try:
            self.user_defined_parameters_dict = user_defined_parameters.get("app_parameters")
        except KeyError:
            traceback.print_exc()
            exit(-1)
        pattern = self.user_defined_parameters_dict.get("pattern")
        assert pattern is not None, "You must define the pattern for PET learning"
        pattern_list = pattern.split(",")
        cnt = 0
        for i in range(len(pattern_list)):
            if pattern_list[i] == "<pseudo>":
                pattern_list[i] = "<pseudo-%d>" % cnt
                cnt += 1
        if cnt > 0:
            self.tokenizer.add_tokens(["<pseudo-%d>" % i for i in range(cnt)])
        # 根据新的词汇表大小, 调整 embedding 的大小
        self.backbone.resize_token_embeddings(len(self.tokenizer))
        print("embedding size: %d" % len(self.tokenizer))
        self.config.vocab_size = len(self.tokenizer)

    def forward(self, inputs):
        """
        看看前向传播, 这个就是 MLM 任务, 普通的预训练模型
        """
        if "mask_span_indices" in inputs:
            inputs.pop("mask_span_indices")
        outputs = self.backbone(**inputs)
        return {"logits": outputs.logits}

    def compute_loss(self, forward_outputs, label_ids):
        """
        计算损失, 似乎也不用修改
        """
        prediction_scores = forward_outputs["logits"]
        # logits 的 shape 是 (batch_size, seq_len, vocab_size), labels 的 shape 是 (batch_size, seq_len)
        # 现在变成 (batch_size * seq_len, vocab_size), (batch_size * seq_len)
        # 就是要找出概率最高的那个词, 然后计算交叉熵损失
        masked_lm_loss = cross_entropy(prediction_scores.view(-1, self.config.vocab_size), label_ids.view(-1))
        return {"loss": masked_lm_loss}

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        # Instantiate model
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        model = cls(pretrained_model_name_or_path=pretrained_model_name_or_path, from_config=config, **kwargs)
        state_dict = None
        weights_path = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
        if not io.exists(weights_path):
            return model
        with io.open(weights_path, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")

        # Load from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if "gamma" in key:
                new_key = key.replace("gamma", "weight")
            if "beta" in key:
                new_key = key.replace("beta", "bias")
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, "_metadata", None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=""):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs
            )
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + ".")

        start_prefix = ""
        logger.info("Loading model...")
        load(model, prefix=start_prefix)
        logger.info("Load finished!")
        if len(missing_keys) > 0:
            logger.info(
                "Weights of {} not initialized from pretrained model: {}".format(model.__class__.__name__, missing_keys)
            )
        if len(unexpected_keys) > 0:
            logger.info(
                "Weights from pretrained model not used in {}: {}".format(model.__class__.__name__, unexpected_keys)
            )
        if len(error_msgs) > 0:
            raise RuntimeError(
                "Error(s) in loading state_dict for {}:\n\t{}".format(model.__class__.__name__, "\n\t".join(error_msgs))
            )
        return model


class CPTClassification(FewshotClassification):
    """An application class for supporting CPT fewshot learning."""

    def __init__(self, pretrained_model_name_or_path=None, user_defined_parameters=None, **kwargs):
        super(CPTClassification, self).__init__(pretrained_model_name_or_path, user_defined_parameters, **kwargs)

        circle_loss_config = self.user_defined_parameters_dict.get("circle_loss_config")
        if circle_loss_config:
            circle_loss_config = json.loads(circle_loss_config)
        else:
            circle_loss_config = dict()
        # 新的损失函数
        self.loss_fcn = CircleLoss(**circle_loss_config)

    def forward(self, inputs, do_mlm=False):
        # currently CPT only supports models that share the structures with bert and hfl/chinese-roberta-wwm-ext
        if "mask_span_indices" in inputs:
            inputs.pop("mask_span_indices")
        if "label_ids" in inputs:
            inputs.pop("label_ids")
        if do_mlm:
            outputs = self.backbone(**inputs)
            return {"logits": outputs.logits}
        else:
            """
            cls 的结构
            (cls): BertOnlyMLMHead(
                (predictions): BertLMPredictionHead(
                (transform): BertPredictionHeadTransform(
                    (dense): Linear(in_features=768, out_features=768, bias=True)
                    (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                )
                (decoder): Linear(in_features=768, out_features=21128, bias=True)
                )
            )
            """
            # x 的 shape 是 (batch_size, seq_len, hidden_size)
            x = self.backbone.bert(**inputs)[0]
            # outputs 的 shape 是 (batch_size, seq_len, hidden_size)
            outputs = self.backbone.cls.predictions.transform(x)
            return {"features": outputs}

    def compute_loss(self, forward_outputs, label_ids):
        """
        forward_outputs["features"].shape 是 (batch_size, seq_len, hidden_size)
        label_ids.shape 是 (batch_size, seq_len)
        """
        # 有效的特征和标签
        features = forward_outputs["features"][label_ids > 0]
        features = nn.functional.normalize(features)
        # 每一行只有一个标签, 所以 seq_len 这个维度就消失了
        labels = label_ids[label_ids > 0]
        # 计算损失, features 的 shape 是 (batch_size, hidden_size), labels 的 shape 是 (batch_size,)
        loss = self.loss_fcn(features, labels)
        return {"loss": loss}


class CircleLoss(nn.Module):
    """
    圆形损失

    CircleLoss 是一种度量学习的损失函数，它可以用于深度特征学习，特别是人脸识别等任务¹³。CircleLoss 的思想是根据每个相似度分数离最优值的远近，给予不同的优化权重，使得类内相似度最大化，类间相似度最小化¹⁴。CircleLoss 可以从一个统一的相似度对优化的角度，兼容类别标签和对标签，同时退化为 Triplet Loss 或 Softmax Loss 等常见的损失函数¹⁵。

    源: 与必应的对话， 2023/6/14
    (1) Circle Loss: 一个基于对优化的统一视角-CVPR2020 - 知乎. https://zhuanlan.zhihu.com/p/126701500.
    (2) Circle loss思想的简单分析理解：Circle Loss: A Unified Perspective of Pair .... https://blog.csdn.net/qq_34124009/article/details/106900412.
    (3) 度量学习DML之Circle Loss_胖胖大海的博客-CSDN博客. https://blog.csdn.net/cxx654/article/details/122158148.
    (4) Circle Loss：从统一的相似性对的优化角度进行深度特征学习 | CVPR 2020 Oral - 知乎. https://zhuanlan.zhihu.com/p/143589143.
    (5) Circle Loss阅读笔记 - 知乎 - 知乎专栏. https://zhuanlan.zhihu.com/p/120676832.
    """

    def __init__(self, margin: float = 0.4, gamma: float = 64, k: float = 1, distance_function="cos") -> None:
        super(CircleLoss, self).__init__()
        self.m = margin
        self.gamma = gamma
        self.k = k
        self.soft_plus = nn.Softplus()
        # 定义距离函数
        if distance_function == "cos":
            self.dist_fcn = lambda X: X @ X.transpose(1, 0)
        else:
            raise NotImplementedError

    def forward(self, features, labels):
        """
        features 的 shape 是 (batch_size, hidden_size)
        labels 的 shape 是 (batch_size, )
        """
        """
        这是一段 PyTorch 代码，它的作用是计算 CircleLoss 的损失值。每一行的维度如下：

        - sim = self.dist_fcn(features).view(-1)：这一行是计算特征向量之间的相似度分数，然后将其展平为一维向量。sim 的维度是 (batch_size * batch_size, )。
        - mask = labels.unsqueeze(1) == labels.unsqueeze(0)：这一行是根据标签生成一个掩码矩阵，表示哪些样本对是同类的，哪些是异类的。mask 的维度是 (batch_size, batch_size)。
        - pos = mask.triu(diagonal=1).view(-1)：这一行是从掩码矩阵中提取出上三角部分（不包括对角线），然后展平为一维向量。pos 表示正样本对的位置，其元素为 True 或 False。pos 的维度是 (batch_size * (batch_size - 1) / 2, )。
        - neg = mask.logical_not().triu(diagonal=1).view(-1)：这一行是对掩码矩阵取反，然后提取出上三角部分（不包括对角线），然后展平为一维向量。neg 表示负样本对的位置，其元素为 True 或 False。neg 的维度是 (batch_size * (batch_size - 1) / 2, )。
        - sp = sim[pos]：这一行是从相似度分数中筛选出正样本对的分数。sp 的维度是 (K, )，其中 K 是正样本对的数量。
        - sn = sim[neg]：这一行是从相似度分数中筛选出负样本对的分数。sn 的维度是 (L, )，其中 L 是负样本对的数量。
        - ap = (1 / self.k) * torch.clamp_min(-sp.detach() + 1 + self.m, min=0.0)：这一行是计算正样本对的权重因子，其中 self.k 是一个超参数，self.m 是一个阈值。ap 的维度和 sp 相同，即 (K, )。
        - an = torch.clamp_min(sn.detach() + self.m, min=0.0)：这一行是计算负样本对的权重因子，其中 self.m 是一个阈值。an 的维度和 sn 相同，即 (L, )。
        - delta_p = 1 - self.m：这一行是计算正样本对的最优值，即类内相似度的目标值。delta_p 是一个标量。
        - delta_n = self.m：这一行是计算负样本对的最优值，即类间相似度的目标值。delta_n 是一个标量。
        - logit_p = -ap * (sp - delta_p) * self.gamma：这一行是计算正样本对的损失项，其中 self.gamma 是一个尺度因子。logit_p 的维度和 sp 相同，即 (K, )。
        - logit_n = an * (sn - delta_n) * self.gamma：这一行是计算负样本对的损失项，其中 self.gamma 是一个尺度因子。logit_n 的维度和 sn 相同，即 (L, )。
        - loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))：这一行是计算总的损失值，其中 self.soft_plus 是一个软化版的 ReLU 函数。loss 是一个标量。
        """
        sim = self.dist_fcn(features).view(-1)
        mask = labels.unsqueeze(1) == labels.unsqueeze(0)
        pos = mask.triu(diagonal=1).view(-1)
        neg = mask.logical_not().triu(diagonal=1).view(-1)
        sp = sim[pos]
        sn = sim[neg]
        ap = (1 / self.k) * torch.clamp_min(-sp.detach() + 1 + self.m, min=0.0)
        an = torch.clamp_min(sn.detach() + self.m, min=0.0)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = -ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma
        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))

        return loss
