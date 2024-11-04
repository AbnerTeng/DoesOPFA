# Does One Pattern Fit All?

Yu-Chen Den,  Kendro Vincent, Zhong-Wei Yeh, and Ru-En Shih

## Overview

Official code base for [Does One Pattern Fit All? Image Analysis for Different Equity Styles](https://drive.google.com/file/d/19dFmDpXK6NVw9VR4NU9-0DxncarDYvkl/view?usp=drive_link). Containes scripts to reproduce experiments.

## Prerequisites

```bash
conda create -n img_analysis python=3.10
conda activate img_analysis

pip install -r requirements.txt
```

## Usage

### Data Preparation

Use the data from `dat/clean_data.parquet` to generate images, and split them into training and testing sets.

Make sure that your serivce has enough memory and CPU / GPU resources to complete the following commands.

```bash
chmod +x script/gen_fig.sh script/transfer_fig_parallel.sh

bash script/gen_fig.sh
bash script/trans_fig_parallel.sh
```

After running the above shell scripts, you'll have images transferred from the raw data. Then, the next step will be generate labels for training and testing.

```bash
chmod +x script/gen_label.sh
bash script/gen_label.sh
```

### Training & Evaluation

Train / test the CNN models with the generated images.

Note that we can test different class of stocks images on different models.

**Train / Evaluate binary classification models**

```bash
chmod +x script/train_bin.sh
bash script/train_bin.sh
```

**Train / Evaluate multi-class classification models**

```bash
chmod +x script/train_multi.sh
bash script/train_multi.sh
```

### Form Portfolio

TBD

## Experiments

### Test the pre-trained models

Instead of directly provide the model checkpoints, we use huggingface to store our models. You can download the model repo from [here](https://huggingface.co/Abner0803/multiclass-stock-cnn).

After downloading the model repo, you can test the pre-trained models by running the command in the `README.md` of the model repo.