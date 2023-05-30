"""
https://www.yuque.com/easyx/easynlp/rxne07
用的数据是这里面的
"""

import sys
import os

print(sys.path)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from easynlp.appzoo import ClassificationDataset
from easynlp.appzoo import get_application_model, get_application_evaluator
from easynlp.core import Trainer
from easynlp.utils import initialize_easynlp, get_args
from easynlp.utils.global_vars import parse_user_defined_parameters
from easynlp.utils import get_pretrain_model_path


if __name__ == "__main__":
    initialize_easynlp()
    args = get_args()
    print(args)

    # 手动定义参数
    args.tables = "./tmp/train.tsv,./tmp/dev.tsv"
    args.input_schema = "label:str:1,sid1:str:1,sid2:str:1,sent1:str:1,sent2:str:1"
    args.first_sequence = "sent1"
    args.second_sequence = "sent2"
    args.label_name = "label"
    args.label_enumerate_values = "0,1"
    args.checkpoint_dir = "./tmp/checkpoint_dir"
    args.learning_rate = 3e-5
    args.epoch_num = 3
    args.random_seed = 42
    args.logging_steps = 1
    args.save_checkpoint_steps = 50
    args.sequence_length = 128
    args.micro_batch_size = 10
    args.app_name = "text_classify"
    args.user_defined_parameters = "pretrain_model_name_or_path=bert-small-uncased"

    user_defined_parameters = parse_user_defined_parameters(args.user_defined_parameters)
    pretrained_model_name_or_path = get_pretrain_model_path(
        user_defined_parameters.get("pretrain_model_name_or_path", None)
    )

    # 定义数据集
    train_dataset = ClassificationDataset(
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

    valid_dataset = ClassificationDataset(
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

    # 定义模型, TODO: 没看懂这个是什么任务
    model = get_application_model(
        app_name=args.app_name,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        num_labels=len(valid_dataset.label_enumerate_values),
        user_defined_parameters=user_defined_parameters,
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        user_defined_parameters=user_defined_parameters,
        evaluator=get_application_evaluator(
            app_name=args.app_name,
            valid_dataset=valid_dataset,
            user_defined_parameters=user_defined_parameters,
            eval_batch_size=args.micro_batch_size,
        ),
    )

    trainer.train()
