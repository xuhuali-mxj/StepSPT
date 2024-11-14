# StepSPT
The code of paper "Step-wise Distribution Alignment Guided Style Prompt Tuning for Source-free Cross-domain Few-shot Learning"

# 1. Setup
conda creat --name stepspt python=3.9

conda activate stepspt

conda install pytorch torchvision -c pytorch

conda install pandas

pip install numpy

pip install argparse

pip install math

pip install os

pip install sklearn

pip install scipy

pip install PIL

pip install abc


# 2. Code clone
git clone https://github.com/xuhuali-mxj/StepSPT.git

cd StepSPT

# 3. Dataset
For the 4 datasets CropDiseases, EuroSAT, ISIC, and ChestX, we refer to the [BS-CDFSL](https://github.com/IBM/cdfsl-benchmark) repo. For PatternNet, please get from [PatternNet](https://huggingface.co/datasets/blanchon/PatternNet).

# 4. Run StepSPT

## Based on ConvNeXt

Our method aims at improving the performance of pretrained source model on the target FSL task. We introduce the style prompt and step-wise distribution alignment, helping the pretrained large-scale model to learn the decision boundary.

Please set your data address in [configs.py](configs.py).

The pretrained source model 'convnext_base_22k_224.pth' can be download in [ConvNeXt](https://github.com/facebookresearch/ConvNeXt).

We start from run.sh. Taking 5-way 1-shot as an example, the code runing process can be done as,

```
python ./stepspt.py  --dtarget CropDisease  --n_shot 1
```

## Based on other backbones

Coming soon.

# 5. Acknowledge
Our code is built upon the implementation of [FTEM_BSR_CDFSL](https://github.com/liubingyuu/FTEM_BSR_CDFSL) and [TPDS](https://github.com/tntek/TPDS). Thanks for their work.

