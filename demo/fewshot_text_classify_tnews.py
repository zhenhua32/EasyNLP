"""
试下小样本分类, 用的是新闻数据集

wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/landing_plm/train.csv
wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/landing_plm/dev.csv

在当前目录下运行 python fewshot_text_classify.py
"""

import sys
import os
import subprocess


root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
os.environ["PYTHONPATH"] = root_dir + ";" + os.environ.get("PYTHONPATH", "")

train_file = r"G:\dataset\text_classify\tnews\fewshot\train.csv"
dev_file = r"G:\dataset\text_classify\tnews\fewshot\dev.csv"
label_file = r"G:\dataset\text_classify\tnews\fewshot\label.json"

checkpoint_dir = os.path.join(root_dir, "demo/tmp/fewshot_model")
os.makedirs(checkpoint_dir, exist_ok=True)
# model_path = r"G:\code\pretrain_model_dir\chinese-macbert-large"
# model_path = r"G:\code\pretrain_model_dir\pai-bert-tiny-zh"
model_path = r"G:\code\pretrain_model_dir\pai-bert-base-zh"

input_schema = "text:str:1,label:str:1"
pattern = "一条,label,的新闻,text"
label_desc = "故事,文化,娱乐,体育,财经,房产,汽车,教育,科技,军事,旅游,国际,股票,农业,电竞"
label_enumerate_values = "100,101,102,103,104,106,107,108,109,110,112,113,114,115,116"
learning_rate = 1e-5
epoch_num = 20
sequence_length = 128
micro_batch_size = 128

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
            f"--label_enumerate_values={label_enumerate_values}",
            f"--checkpoint_dir={checkpoint_dir}",
            f"--learning_rate={learning_rate}",
            f"--epoch_num={epoch_num}",
            "--random_seed=42",
            "--save_checkpoint_steps=100",
            f"--sequence_length={sequence_length}",
            f"--micro_batch_size={micro_batch_size}",
            (
                "--user_defined_parameters="
                + f"pretrain_model_name_or_path={model_path}\t"
                + "enable_fewshot=True\t"
                + f"label_desc={label_desc}\t"
                + "type=pet_fewshot\t"
                + "pattern=一条,label,的新闻,text"
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
            f"--input_schema={input_schema}",
            "--first_sequence=text",
            "--label_name=label",
            f"--label_enumerate_values={label_enumerate_values}",
            f"--checkpoint_dir={checkpoint_dir}",
            f"--learning_rate={learning_rate}",
            f"--epoch_num={epoch_num}",
            "--random_seed=42",
            "--save_checkpoint_steps=100",
            f"--sequence_length={sequence_length}",
            f"--micro_batch_size={micro_batch_size}",
            (
                "--user_defined_parameters="
                + f"pretrain_model_name_or_path={model_path}\t"
                + "enable_fewshot=True\t"
                + f"label_desc={label_desc}\t"
                + "type=pet_fewshot\t"
                + "pattern=<pseudo>,label,<pseudo>,text"
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
            f"--input_schema={input_schema}",
            "--first_sequence=text",
            "--label_name=label",
            f"--label_enumerate_values={label_enumerate_values}",
            f"--checkpoint_dir={checkpoint_dir}",
            f"--learning_rate={learning_rate}",
            f"--epoch_num={epoch_num}",
            "--random_seed=42",
            "--save_checkpoint_steps=100",
            f"--sequence_length={sequence_length}",
            f"--micro_batch_size={micro_batch_size}",
            (
                "--user_defined_parameters="
                + f"pretrain_model_name_or_path={model_path}\t"
                + "enable_fewshot=True\t"
                + "type=cpt_fewshot\t"
                + f"pattern={pattern}"
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
            f"--input_schema={input_schema}",
            "--first_sequence=text",
            "--label_name=label",
            f"--label_enumerate_values={label_enumerate_values}",
            f"--checkpoint_dir={checkpoint_dir}",
            f"--learning_rate={learning_rate}",
            f"--epoch_num={epoch_num}",
            "--random_seed=42",
            "--save_checkpoint_steps=100",
            f"--sequence_length={sequence_length}",
            f"--micro_batch_size={micro_batch_size}",
            ("--user_defined_parameters=" + f"pretrain_model_name_or_path={model_path}\t"),
        ]
    )
