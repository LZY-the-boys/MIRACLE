# MIRACLE

EMNLP2023-findings: MIRACLE: Towards Personalized Dialogue Generation with Latent-Space Multiple Personal Attribute Control

Dataset is uploaded in [here](https://huggingface.co/datasets/lu-vae/Miracle-Conversation).

The arxiv paper is on holding... 

![image](https://github.com/LZY-the-boys/MIRACLE/assets/72137647/ad539a33-2ee1-4a16-b045-4b8dd1f37de0)

## The Concept

We modeling the  multi-facted personality as the fusion of multiple personal attribute ($P_1, P_2, \cdots, P_N$), where each attribute may have many aspects ($p_1, p_2, \cdots, p_m$).

## Setup

This repo is build based on [ 🤗 huggingface transformers](https://github.com/huggingface/transformers) and [torchdiffeq](https://github.com/rtqichen/torchdiffeq).

The training and evaluating can be done with a single RTX3090/4090.

The recommendate environment :

```
conda create -n miracle python=3.9
pip install -r requirements.txt
```

## Training and evaluating

In the following part, we assume you have downloaded [the dataset we used](https://huggingface.co/datasets/lu-vae/Miracle-Conversation)

Firstly, you should train two model for evaluation:
- `single-personal-attribute text classifier` for calculating personalization score.
```
ATTR=languagestyle bash train_classifier.sh
```
- `NLI model` for calculating dialogue resposne semantic coherence.
```
bash train_nli.sh
```

Then, you can train our Miracle and generate personalized responses by:
```
bash pipeline.sh
```
which call `train_cvae.sh` for training and `gen_cave.sh` for generation.

Notice that for different dataset or different senarioes, you may need to adjust hyper-parameters to gain better results. 

## Customize you data

To train our model, you can use any personl-attribute-dense dataset in the format like our released data. 

We also upload our [ChatGPT-API script](make_data_chatgpt.py) as reference. It generate personalized response in aspect level. Notice that you need to prepare you own topics in `dataset/sample_topics`:

```
python chatgpt_data.py --key 'Your OpenAI API key' --aspect 'xx1'
```

With the collected data, you need to format them as follows in `dataset/dialogue_yy.jsonl` for each attribute:
```
{"input": ["user post1", "user post2"], "output": ["model resp1", "model resp2"], "tag": 'xx1'}
{"input": ["user post1", "user post2"], "output": ["model resp1", "model resp2"], "tag": 'xx2'}
...

```
We recommendate you to clean them before use to make sure it's personality richness.

Before training, change the `schema.py` to set you dataset file paths.

## Citation

```
@inproceedings{miracle,
    title = "MIRACLE: Towards Personalized Dialogue Generation with Latent-Space Multiple Personal Attribute Control",
    author = "Zhenyi Lu  and
      Wei Wei and
      Xiaoye Qu  and
      XianLing Mao  and
      Dangyang Chen  and
      Jixiong Chen",
    year = "2023",
    publisher = "Association for Computational Linguistics",
}
```
