"""
在当前目录下运行 python train.py
"""

import sys
import os
import subprocess


root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(root_dir)
sys.path.append(root_dir)
os.environ["PYTHONPATH"] = root_dir + ";" + os.environ.get("PYTHONPATH", "")

script_dir = os.path.dirname(os.path.abspath(__file__))
# train_file = os.path.join(script_dir, "tmp/data/train_part.tsv")
train_file = os.path.join(script_dir, "tmp/data/train.tsv")
valid_file = os.path.join(script_dir, "tmp/data/dev.tsv")
checkpoint_dir = os.path.join(script_dir, "tmp/model")
os.makedirs(checkpoint_dir, exist_ok=True)
model_path = r"G:\code\pretrain_model_dir\pai-bert-base-zh"

"""
python main.py \
--mode train \
--tables=train.tsv,dev.tsv \
--input_schema=id:str:1,instruction:str:1,start:str:1,end:str:1,target:str:1 \
--worker_gpu=4 \
--app_name=information_extraction \
--sequence_length=512 \
--weight_decay=0.0 \
--micro_batch_size=2 \
--checkpoint_dir=./information_extraction_model/ \
--data_threads=5 \
--user_defined_parameters='pretrain_model_name_or_path=hfl/macbert-large-zh' \
--save_checkpoint_steps=500 \
--gradient_accumulation_steps=8 \
--epoch_num=3  \
--learning_rate=2e-05  \
--random_seed=42
"""

subprocess.run(
    [
        sys.executable,
        os.path.join(script_dir, "main.py"),
        f"--tables={train_file},{valid_file}",
        "--input_schema=id:str:1,instruction:str:1,start:str:1,end:str:1,target:str:1",
        "--worker_gpu=1",
        "--app_name=information_extraction",
        "--sequence_length=128",
        "--weight_decay=0.0",
        "--micro_batch_size=256",
        f"--checkpoint_dir={checkpoint_dir}",
        "--data_threads=0",
        f"--user_defined_parameters=pretrain_model_name_or_path={model_path}",
        "--save_checkpoint_steps=100",
        # "--gradient_accumulation_steps=8",
        "--epoch_num=3",
        "--learning_rate=2e-05",
        "--random_seed=42",
    ]
)
