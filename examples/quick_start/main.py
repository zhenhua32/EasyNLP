import sys

sys.path.append("../../")

from easynlp.core import Trainer
from easynlp.appzoo import ClassificationDataset, SequenceClassification
from easynlp.utils import initialize_easynlp


"""
基本所有内容都要在 __main__ 中执行, 不然会重复导入. TODO: 原因未知
主要问题是这个文件会被反复导入, 然后 initialize_easynlp 里的 initializing torch distributed 会报错
我不知道为什么会被重复导入, 是在哪里发生的.
"""

if __name__ == "__main__":
    args = initialize_easynlp()

    train_dataset = ClassificationDataset(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        data_file=args.tables,
        max_seq_length=args.sequence_length,
        input_schema=args.input_schema,
        first_sequence=args.first_sequence,
        label_name=args.label_name,
        label_enumerate_values=args.label_enumerate_values,
        is_training=True,
    )

    model = SequenceClassification(pretrained_model_name_or_path=args.pretrained_model_name_or_path)
    Trainer(model=model, train_dataset=train_dataset).train()


"""
cd examples/quick_start/
python main.py `
  --mode train `
  --tables=./tmp/train_toy.tsv `
  --input_schema=label:str:1,sid1:str:1,sid2:str:1,sent1:str:1,sent2:str:1 `
  --first_sequence=sent1 `
  --label_name=label `
  --label_enumerate_values=0,1 `
  --checkpoint_dir=./tmp/ `
  --epoch_num=1  `
  --app_name=text_classify `
  --user_defined_parameters='pretrain_model_name_or_path=bert-small-uncased'
"""
