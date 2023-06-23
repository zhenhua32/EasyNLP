"""
https://www.yuque.com/easyx/easynlp/rxne07
用的数据是这里面的

在当前目录下运行 python fewshot_multi_layer.py
"""

import sys
import os
import json

cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(cur_dir)
sys.path.append(root_dir)


from easynlp.fewshot_learning.fewshot_dataset import FewshotMultiLayerBaseDataset
from easynlp.fewshot_learning.fewshot_application import FewshotMultiLayerClassification
from easynlp.fewshot_learning.fewshot_evaluator import PromptMultiLayerEvaluator
from easynlp.core import Trainer
from easynlp.utils import initialize_easynlp, get_args
from easynlp.utils.global_vars import parse_user_defined_parameters
from easynlp.utils import get_pretrain_model_path


def main():
    initialize_easynlp()
    args = get_args()
    print(args)

    user_defined_parameters = parse_user_defined_parameters(args.user_defined_parameters)
    pretrained_model_name_or_path = get_pretrain_model_path(
        user_defined_parameters.get("pretrain_model_name_or_path", None)
    )

    # 定义数据集
    train_dataset = FewshotMultiLayerBaseDataset(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        data_file=args.tables.split(",")[0],
        max_seq_length=args.sequence_length,
        input_schema=args.input_schema,
        first_sequence=args.first_sequence,
        second_sequence=args.second_sequence,
        label_name=args.label_name,
        label_enumerate_values=args.label_enumerate_values,
        user_defined_parameters=user_defined_parameters,
        is_training=True,
    )

    valid_dataset = FewshotMultiLayerBaseDataset(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        data_file=args.tables.split(",")[-1],
        max_seq_length=args.sequence_length,
        input_schema=args.input_schema,
        first_sequence=args.first_sequence,
        second_sequence=args.second_sequence,
        label_name=args.label_name,
        label_enumerate_values=args.label_enumerate_values,
        user_defined_parameters=user_defined_parameters,
        is_training=False,
    )

    model = FewshotMultiLayerClassification(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        user_defined_parameters=user_defined_parameters,
    )

    evaluator = PromptMultiLayerEvaluator(
        valid_dataset=valid_dataset,
        user_defined_parameters=user_defined_parameters,
        eval_batch_size=max(args.micro_batch_size, 128),
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        user_defined_parameters=user_defined_parameters,
        evaluator=evaluator,
    )

    trainer.train()


if __name__ == "__main__":
    main()
