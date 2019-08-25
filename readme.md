## 模型

基于BERT实现的阅读理解模型，在答案片段抽取网络的基础上额外加了一个分类器用于判别是否型问题和不可回答问题。

## 数据集

该模型专为法研杯（[赛事官网](http://cail.cipsc.org.cn/)）阅读理解赛道的题目而设计，比赛数据集来自于官方提供法律文书问答数据集，可在[此处](https://pan.baidu.com/s/1p4NJDhboKSsbFQOwRgDszg)下载（提取码：8w0y）。

数据集格式如下图所示，大部分问题的答案为从文档中直接抽取得到，另外还包含拒答以及是否类（YES/NO）问题。

![](https://img2018.cnblogs.com/blog/1018727/201907/1018727-20190715235304603-404055187.jpg)

## Baseline

[https://github.com/china-ai-law-challenge/CAIL2019/tree/master/%E9%98%85%E8%AF%BB%E7%90%86%E8%A7%A3](https://github.com/china-ai-law-challenge/CAIL2019/tree/master/阅读理解)

## 要求

- Python = 3.6
- Tensorflow >= 1.11.0

## 准备

- 下载数据集：https://pan.baidu.com/s/1p4NJDhboKSsbFQOwRgDszg 
  提取码：8w0y
- 下载BERT预训练中文模型：https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip

## 训练

```
python run_cail_with_yorn.py \
  --vocab_file=$MODEL_HOME/vocab.txt \
  --bert_config_file=$MODEL_HOME/bert_config.json \
  --init_checkpoint=$MODEL_HOME/bert_model.ckpt \
  --do_train=True \
  --train_file=$DATA_DIR/big_train_data.json \
  --train_batch_size=8 \
  --learning_rate=3e-5 \
  --num_train_epochs=7.0 \
  --max_seq_length=512 \
  --output_dir=$OUTPUT_DIR/cail_yorn/
```

`$MODEL_HOME`是BERT预训练语言模型的目录，`$DATA_DIR`是数据集的路径，`$OUTPUT_DIR`是输出文件路径，保存训练模型等文件。

## 预测

```
python run_cail_with_yorn.py \
  --vocab_file=$MODEL_HOME/vocab.txt \
  --bert_config_file=$MODEL_HOME/bert_config.json \
  --do_predict=True \
  --predict_file=$DATA_DIR/test_data.json \
  --max_seq_length=512 \
  --output_dir=$OUTPUT_DIR/cail_yorn/
```
## TODO
Ensemble模型
