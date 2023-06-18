import json
import torch
from threading import Lock
from ..dataset import BaseDataset
from ...modelzoo.models.bert import BertTokenizerFast


class InformationExtractionDataset(BaseDataset):
    """
    信息抽取的数据集定义
    """

    def __init__(self, pretrained_model_name_or_path, data_file, input_schema, max_seq_length, *args, **kwargs):
        super(InformationExtractionDataset, self).__init__(
            data_file, input_schema=input_schema, output_format="dict", *args, **kwargs
        )

        self.max_seq_length = max_seq_length
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path)

    def convert_single_row_to_example(self, row):
        """
        转换单个样本
        """
        # 第一列是 id
        id = row[self.column_names[0]]
        # 第二列是 instruction, 但是会去掉首尾的两个字符, 即 [' 和 ']
        instruction = row[self.column_names[1]][2:-2]
        # 第三列是开始的索引, 也要去掉首尾的 []
        start = row[self.column_names[2]][1:-1]
        if start == "":
            start = []
        else:
            # 用逗号分隔的
            start = start.split(",")
            # 转换成 int 类型
            start = [int(i) for i in start]
        # 第四列是结束的索引, 处理方式和开始的索引一样
        end = row[self.column_names[3]][1:-1]
        if end == "":
            end = []
        else:
            end = end.split(",")
            end = [int(i) for i in end]
        # 第五列是目标, 这里怎么没提到是用 | 分隔的
        target = row[self.column_names[4]]

        # 原来如此, 这里有 input_ids 等字段
        example = self.tokenizer(
            instruction,
            truncation=True,
            max_length=self.max_seq_length,
            padding="max_length",
            return_offsets_mapping=True,  # 这会多一个 offsets_mapping 字段
        )

        # 每个样本再加上 5 个字段
        example["id"] = id
        example["instruction"] = instruction
        example["start"] = start
        example["end"] = end
        example["target"] = target

        return example

    def batch_fn(self, features):
        """
        批次聚合处理
        """
        batch = []
        for f in features:
            batch.append(
                {
                    "input_ids": f["input_ids"],
                    "token_type_ids": f["token_type_ids"],
                    "attention_mask": f["attention_mask"],
                }
            )

        # 原来 pad 的输入是个 dict 的 list
        batch = self.tokenizer.pad(
            batch,
            padding="max_length",  # 为了index不出错直接Padding到max length，如果用longest，后面的np.unravel_index也要改
            max_length=self.max_seq_length,
            return_tensors="pt",
        )

        # labels 的 shape 是 (batch_size, 1, max_seq_length, max_seq_length)
        labels = torch.zeros(
            len(features), 1, self.max_seq_length, self.max_seq_length
        )  # 阅读理解任务entity种类为1 [bz, 1, max_len, max_len]
        for feature_id, feature in enumerate(features):  # 遍历每个样本
            starts, ends = feature["start"], feature["end"]
            offset = feature["offset_mapping"]  # 表示tokenizer生成的token对应原始文本中字符级别的位置区间
            # 定义一个字典，key是字符级别的位置，value是对应的token的索引
            position_map = {}
            for i, (m, n) in enumerate(offset):
                if i != 0 and m == 0 and n == 0:
                    continue
                for k in range(m, n + 1):
                    position_map[k] = i  # 字符级别的第k个字符属于分词i
            for start, end in zip(starts, ends):
                # 对每一个首尾答案位置, 索引风格是左包含, 右不包含, 所以右边减一后就是左右都包含了
                end -= 1
                # MRC 是机器阅读理解任务
                # MRC 没有答案时则把label指向CLS
                if start == 0:
                    assert end == -1
                    labels[feature_id, 0, 0, 0] = 1
                else:
                    if start in position_map and end in position_map:
                        # 指定下列元素为1，说明表示第feature_id个样本的预测区间
                        labels[feature_id, 0, position_map[start], position_map[end]] = 1

        batch["label_ids"] = labels

        # 重新整理结构, 变成大数组
        tempid = []
        tempinstruction = []
        tempoffset_mapping = []
        temptarget = []
        for i in range(len(features)):
            tempid.append(features[i]["id"])
            tempinstruction.append(features[i]["instruction"])
            tempoffset_mapping.append(features[i]["offset_mapping"])
            temptarget.append(features[i]["target"])

        batch["id"] = tempid
        batch["instruction"] = tempinstruction
        batch["offset_mapping"] = tempoffset_mapping
        batch["target"] = temptarget

        return batch
