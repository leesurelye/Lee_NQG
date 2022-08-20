- debug command :

# Data Size 
18478 - train 
Vocab Size  45000
---
trail_time = 12 BLEU_score: 0.085


|trail_time| gated_self_attention|local coverage mechanism|BlEU score|ROUGE score|Meteor score|
|------|------|------|------|------|------|
|12|yes|yes|size 5: 0.085|--|--|
|13|yes|global|size 5: 0.083|--|--|
|14|yes|local |size 5:0.081|||
|15|no| local|size 5: 0.074|||
|16|yes|yes| size 5: simple 0.5|||
|17|yes|no|0.020|||
|18|yes|yes|0.085|||
|19|yes|yes local step 5| 0.088|||
|22|yes|yes 没有使用答案标注|0.038||
|24|yes|yes graph yes|0.090||

用了forcing ratio 整体损失在上升
```shell
tensorboard --logdir /home1/liyue/Lee_NQG/debug/tensorboard --host 10.20.13.18
```
coverage loss recoder
- one batch:
```text

```

---
# 版本说明
- version : `V1`: 第26次实验，效果比较好是因为添加了K交叉验证
    - 本次版本只是为了验证第26次实验的效果
    - 添加了graph， `input_a`
    - 添加了评估代码，文件夹为 qgvalcap ,之前的评估代码出现错误
    - 之前版本的文本推理方式，导致不能直接用已经写好的评估代码评估，因此这个修改了文本生成结果的存储方式
    - 原始的试验次数的代码是26，这一次修改成31，因为30次实验出现性能急剧下降的问题
    - 删除了其他版本中的版本注释，将各个版本之间的代码分开
    - 等待实验结果
- 实验结果
- min_loss : 0.7755
- auto evaluate

Bleu_1: 0.39130

Bleu_2: 0.25926

Bleu_3: 0.18987

Bleu_4: 0.14425

METEOR: 0.20922

ROUGE_L: 0.37399

注意：由于每一次实验的gold和predict的分词结果不同，会导致自动评估的分数相差很大：具体体现在：

Bleu_1: 0.32381

Bleu_2: 0.18684

Bleu_3: 0.12041

Bleu_4: 0.08093

METEOR: 0.18492

ROUGE_L: 0.28197

---
这一版本修改的内容是添加weight_decay参数
weight_decay = 0 改为 0.001
