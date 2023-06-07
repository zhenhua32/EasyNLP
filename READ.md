[TOC]

像这种, 如果还在频繁更新, 就建立个 read 分支来读代码.

# 安装

```bash
conda activate nlp
python -m venv ./venv
.\venv\Scripts\activate

git clone https://github.com/alibaba/EasyNLP.git
pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple
cd EasyNLP
python setup.py install
```

# 文档

[EasyNLP中文文档](https://www.yuque.com/easyx/easynlp/iobg30)

[API文档](http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/easynlp/easynlp_docs/html/index.html)

阿里的文档也写的太烂了, 而且是面向 linux 的, 没有做 windows 适配.
这文档看了跟没看一样.

实用文档:

- [ModelZoo列表](https://www.yuque.com/easyx/easynlp/cn0uh8)

# 目录理解

- easynlp: 代码目录
- examples: 示例目录

其他都不用看了, 就以上这是核心目录.

- docs: API文档目录, 不用看了, 很久没更新了. `git log docs`, 就提交了一次.


![桃花与猫](./my_doc/image/桃花与猫.jpg)
