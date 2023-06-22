import os
import json
import sys
import subprocess
from collections import defaultdict
from transformers import AutoTokenizer


def parse_label(label_file, model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # 先解析 label
    with open(label_file, "r", encoding="utf-8") as f:
        # 这边 id 是 int
        label2id = json.load(f)
        label2id = {k: str(v) for k, v in label2id.items()}

    # 想想每一层的标签应该怎么建立, 现在假设每层都是完整的, 也就是每个样本的标签都会到最后一层
    label_enumerate_values = defaultdict(list)
    label_desc = defaultdict(list)
    for label, label_id in label2id.items():
        label_split = label.split(">")
        # 标签应该只放最后一层的. TODO: 同名怎么办?
        idx = len(label_split) - 1
        label_enumerate_values[idx].append(label_id)
        label_desc[idx].append(label_split[-1])

    # 其实应该还是有序的, 因为添加的时候是从左到右添加的
    # 计算 label_desc 的最大长度
    for idx in sorted(label_desc.keys()):
        cur_list = label_desc[idx]
        cur_max_len = max([len(tokenizer.tokenize(x)) for x in cur_list])
        print(f"layer_{idx}, max_label_len: {cur_max_len}")
        # 填充到最大长度
        label_desc[idx] = [x + "[PAD]" * (cur_max_len - len(tokenizer.tokenize(x))) for x in cur_list]

    label_enumerate_values_new = []
    for idx in sorted(label_enumerate_values.keys()):
        label_enumerate_values_new.append(",".join(label_enumerate_values[idx]))
    label_enumerate_values_new = "@@".join(label_enumerate_values_new)

    label_desc_new = []
    for idx in sorted(label_desc.keys()):
        label_desc_new.append(",".join(label_desc[idx]))
    label_desc_new = "@@".join(label_desc_new)

    return label_enumerate_values_new, label_desc_new


cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(cur_dir)
sys.path.append(root_dir)
os.environ["PYTHONPATH"] = root_dir + ";" + os.environ.get("PYTHONPATH", "")

train_file = r"G:\dataset\text_classify\网页层次分类\train.csv"
test_file = r"G:\dataset\text_classify\网页层次分类\test.csv"
label_file = r"G:\dataset\text_classify\网页层次分类\label.json"
model_dir = r"G:\code\pretrain_model_dir\pai-bert-base-zh"
checkpoint_dir = os.path.join(cur_dir, "tmp/fewshot_multi_layer")
os.makedirs(checkpoint_dir, exist_ok=True)

label_enumerate_values, label_desc = parse_label(label_file, model_dir)
input_schema = "text:str:1,label0:str:1,label1:str:1"
label_name = "label0,label1"
pattern = "一条,label0,label1,的新闻,text"


cmd_list = [
    sys.executable,
    os.path.join(cur_dir, "fewshot_multi_layer.py"),
    "--mode=train",
    "--worker_count=1",
    "--worker_gpu=1",
    f"--tables={train_file},{test_file}",
    f"--input_schema={input_schema}",
    "--first_sequence=text",
    f"--label_name={label_name}",
    f"--label_enumerate_values={label_enumerate_values}",
    f"--checkpoint_dir={checkpoint_dir}",
    "--learning_rate=1e-5",
    "--epoch_num=5",
    "--random_seed=42",
    "--save_checkpoint_steps=100",
    "--sequence_length=128",
    "--micro_batch_size=64",
    (
        "--user_defined_parameters="
        + f"pretrain_model_name_or_path={model_dir}\t"
        + f"label_desc={label_desc}\t"
        + f"pattern={pattern}"
    ),
]


if __name__ == "__main__":
    print(cmd_list)
    subprocess.run(cmd_list)
