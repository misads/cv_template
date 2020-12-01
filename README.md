# cv_template

　　一个图像复原或分割的Baseline，可以用于去雾🌫、去雨🌧、去模糊、夜景🌃复原、超分辨率👾、像素级分割等等。
  
  
　　<img alt='preview' src='http://www.xyu.ink/wp-content/uploads/2020/11/dehaze2.png' height=300/>


## Highlights

- 特色功能
  - [x] 快速搭建baseline，只需生成输入和标签对应的txt文件，无需修改代码即可运行
  - [x] (参数控制)多模型
  - [x] 训练过程监控
  - [x] 命令行日志记录
  - [x] 支持TTA
  - [x] 自动化超参数搜索🔍

<!-- 
## To do List

- 模型
  - [ ] FFA-Net
  - [ ] Multi-Scale Boosted Dehazing Network with Dense Feature Fusion 
  - [ ] Cascaded Refinement
  - [ ] PANet
- 平台支持
  - [ ] 多GPU支持
  - [ ] 测试时支持多`batch_size`
  
- TTA
  - [ ] 放大、色相、饱和度、亮度
  - [ ] `flip`
  - [ ] 多尺度测试
  - [ ] ttach库
- 其他Tricks
  - [ ] 使用fp_16训练，提高训练速度
  - [ ] One_Cycle 学习率 -->

## 环境需求

```yaml
python >= 3.6
torch >= 1.0
tensorboardX >= 1.6
utils-misc >= 0.0.5
mscv >= 0.0.3
```

## 使用方法

### 训练和验证模型

① 生成输入图片和标签对应的train.txt和val.txt

```bash
# !- bash
python utils/make_filelist.py --input datasets/images/ --label /datasets/labels --val_ratio 0.1 --out datasets
```

　　这会在`datasets`目录下生成一个train.txt和一个val.txt，每行是一对样本的输入和gt的绝对路径，用空格隔开。

② 训练模型

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --tag ffa --model FFA --epochs 20 -b 2 --lr 0.0001 # --tag用于区分每次实验，可以是任意字符串
```

③ 验证训练的模型

```bash
CUDA_VISIBLE_DEVICES=0 python eval.py --model FFA -b 2 --load checkpoints/ffa/20_FFA.pt
```

④ 恢复中断的训练

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --tag ffa_resume --model FFA --epochs 20 -b 2 --lr 0.0001 --load checkpoints/ffa/10_FFA.pt --resume
```

⑤ 在测试集上测试

```bash
CUDA_VISIBLE_DEVICES=0 python submit.py --model FFA --load checkpoints/ffa/20_FFA.pt
```

### 记录和查看日志

　　所有运行的命令和运行命令的时间戳会自动记录在`run_log.txt`中。

　　不同实验的详细日志和Tensorboard日志文件会记录在`logs/<tag>`文件夹中，checkpoint文件会保存在`checkpoints/<tag>`文件夹中。


### 参数说明

`--tag`参数是一次操作(`train`或`eval`)的标签，日志会保存在`logs/标签`目录下，保存的模型会保存在`checkpoints/标签`目录下。  

`--model`是使用的模型，所有可用的模型定义在`network/__init__.py`中。  

`--epochs`是训练的代数。  

`-b`参数是`batch_size`，可以根据显存的大小调整。  

`-w`参数是`num_workers`，即读取数据的进程数，如果需要用pdb来debug，将这个参数设为0。  

`--lr`是初始学习率。

`--load`是加载预训练模型。  

`--resume`配合`--load`使用，会恢复上次训练的`epoch`和优化器。  

`--gpu`指定`gpu id`，目前只支持单卡训练。  

另外还可以通过参数调整优化器、学习率衰减、验证和保存模型的频率等，详细请查看`python train.py --help`。  


### 清除不需要的实验记录

　　运行 `python clear.py --tag your_tag` 可以清除不需要的实验记录，注意这是不可恢复的，如果你不确定你在做什么，请不要使用这条命令。


## 如何添加新的模型：

```
如何添加新的模型：

① 复制network目录下的Default文件夹，改成另外一个名字(比如MyNet)。

② 在network/__init__.py中import你的Model并且在models = {}中添加它。
    from MyNet.Model import Model as MyNet
    models = {
        'default': Default,
        'MyNet': MyNet,
    }

③ 尝试 python train.py --model MyNet 看能否成功运行
```
