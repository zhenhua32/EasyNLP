"""
试下小样本分类

wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/landing_plm/train.csv
wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/landing_plm/dev.csv
"""

import sys
import os
import subprocess


root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
os.environ["PYTHONPATH"] = root_dir + ";" + os.environ.get("PYTHONPATH", "")


"""
easynlp \
    --app_name=text_classify \
    --mode=train \
    --worker_count=1 \
    --worker_gpu=1 \
    --tables=train.csv,dev.csv \
    --input_schema=text:str:1,label:str:1 \
    --first_sequence=text \
    --label_name=label \
    --label_enumerate_values=Positive,Negative \
    --checkpoint_dir=./fewshot_model/ \
    --learning_rate=1e-5 \
    --epoch_num=5 \
    --random_seed=42 \
    --save_checkpoint_steps=100 \
    --sequence_length=512 \
    --micro_batch_size=8 \
    --user_defined_parameters="
        pretrain_model_name_or_path=hfl/macbert-large-zh
        enable_fewshot=True
        label_desc=好,差
        type=pet_fewshot
        pattern=text,是一条商品,label,评。
    "
"""

train_file = os.path.join(root_dir, "demo/tmp/fewshot_data/train.csv")
dev_file = os.path.join(root_dir, "demo/tmp/fewshot_data/dev.csv")
checkpoint_dir = os.path.join(root_dir, "demo/tmp/fewshot_model")
os.makedirs(checkpoint_dir, exist_ok=True)
# model_path = r"G:\code\pretrain_model_dir\chinese-macbert-large"
model_path = r"G:\code\pretrain_model_dir\pai-bert-tiny-zh"
model_path = r"G:\code\pretrain_model_dir\pai-bert-base-zh"

algorithm_type = "PET"

if algorithm_type == "PET":
    # 这个小样本学习才只有 0.825 的准确率
    print("=============PET算法")
    subprocess.run(
        [
            sys.executable,
            os.path.join(root_dir, "easynlp/cli.py"),
            "--app_name=text_classify",
            "--mode=train",
            "--worker_count=1",
            "--worker_gpu=1",
            f"--tables={train_file},{dev_file}",
            "--input_schema=text:str:1,label:str:1",
            "--first_sequence=text",
            "--label_name=label",
            "--label_enumerate_values=Positive,Negative",
            f"--checkpoint_dir={checkpoint_dir}",
            "--learning_rate=1e-5",
            "--epoch_num=5",
            "--random_seed=42",
            "--save_checkpoint_steps=100",
            "--sequence_length=512",
            "--micro_batch_size=8",
            (
                "--user_defined_parameters="
                + f"pretrain_model_name_or_path={model_path}\t"
                + "enable_fewshot=True\t"
                + "label_desc=好,差\t"
                + "type=pet_fewshot\t"
                + "pattern=text,是一条商品,label,评"
            ),
        ]
    )

elif algorithm_type == "P-Tuning":
    print("=============P-Tuning算法")
    subprocess.run(
        [
            sys.executable,
            os.path.join(root_dir, "easynlp/cli.py"),
            "--app_name=text_classify",
            "--mode=train",
            "--worker_count=1",
            "--worker_gpu=1",
            f"--tables={train_file},{dev_file}",
            "--input_schema=text:str:1,label:str:1",
            "--first_sequence=text",
            "--label_name=label",
            "--label_enumerate_values=Positive,Negative",
            f"--checkpoint_dir={checkpoint_dir}",
            "--learning_rate=1e-5",
            "--epoch_num=5",
            "--random_seed=42",
            "--save_checkpoint_steps=100",
            "--sequence_length=512",
            "--micro_batch_size=8",
            (
                "--user_defined_parameters="
                + f"pretrain_model_name_or_path={model_path}\t"
                + "enable_fewshot=True\t"
                + "label_desc=好,差\t"
                + "type=pet_fewshot\t"
                + "pattern=text,<pseudo>,label"
            ),
        ]
    )

elif algorithm_type == "CP-Tuning":
    print("=============CP-Tuning算法")
    subprocess.run(
        [
            sys.executable,
            os.path.join(root_dir, "easynlp/cli.py"),
            "--app_name=text_classify",
            "--mode=train",
            "--worker_count=1",
            "--worker_gpu=1",
            f"--tables={train_file},{dev_file}",
            "--input_schema=text:str:1,label:str:1",
            "--first_sequence=text",
            "--label_name=label",
            "--label_enumerate_values=Positive,Negative",
            f"--checkpoint_dir={checkpoint_dir}",
            "--learning_rate=1e-5",
            "--epoch_num=5",
            "--random_seed=42",
            "--save_checkpoint_steps=100",
            "--sequence_length=512",
            "--micro_batch_size=8",
            (
                "--user_defined_parameters="
                + f"pretrain_model_name_or_path={model_path}\t"
                + "enable_fewshot=True\t"
                + "type=cpt_fewshot\t"
                + "pattern=text,是一条商品,label,评。"
            ),
        ]
    )

elif algorithm_type == "FT":
    # 标准微调能到 0.86875
    print("=============标准微调")
    subprocess.run(
        [
            sys.executable,
            os.path.join(root_dir, "easynlp/cli.py"),
            "--app_name=text_classify",
            "--mode=train",
            "--worker_count=1",
            "--worker_gpu=1",
            f"--tables={train_file},{dev_file}",
            "--input_schema=text:str:1,label:str:1",
            "--first_sequence=text",
            "--label_name=label",
            "--label_enumerate_values=Positive,Negative",
            f"--checkpoint_dir={checkpoint_dir}",
            "--learning_rate=1e-5",
            "--epoch_num=5",
            "--random_seed=42",
            "--save_checkpoint_steps=100",
            "--sequence_length=512",
            "--micro_batch_size=8",
            ("--user_defined_parameters=" + f"pretrain_model_name_or_path={model_path}\t"),
        ]
    )
