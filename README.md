# cv_template

　　一个图像复原或分割的Baseline，可以用于去雾🌫、去雨🌧、去模糊、夜景🌃复原、超分辨率👾、像素级分割等等。

## Highlights

  - [x] 快速搭建baseline，只需生成filelist，无需修改代码即可运行
  - [x] (参数控制)多模型
  - [x] 训练过程监控
  - [x] 测试时支持TTA


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
  - [ ] One_Cycle 学习率

## Prerequisites

```yaml
python >= 3.6
torch >= 1.0
tensorboardX >= 1.6
utils-misc >= 0.0.5
mscv >= 0.0.3
```

## 生成filelist

```bash
# !- bash
python utils/make_filelist.py --input datasets/images/ --label /datasets/labels --val_ratio 0.1 --out datasets
```

## Code Usage

#### 训练

```bash
# !- bash
python3 train.py --tag run1 --model FFA -b 2 --epochs 500 --gpu 1
```

#### 验证

```bash
# !- bash
python3 eval.py --tag pengzhang --load checkpoints/run1/480_FFA.pt --tta
```

#### 更多用法请参考

```bash
# !- bash
python help.py
```

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