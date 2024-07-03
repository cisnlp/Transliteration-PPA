# Transliteration-PPA

This is the repository for the paper
"Breaking the Script Barrier in Multilingual Pre-Trained Language Models with Transliteration-Based Post-Training Alignment",
which aims to improve the cross-lingual transfer of multilingual models by better aligning the representations of languages
written in different scripts. We utilize a rule-based transliteration system ([uroman](https://github.com/isi-nlp/uroman)) to
generate Latin-script transliterations of sentences in different scripts, and then use our proposed post-pretraining alignment (PPA)
method to fine-tune a pretrained multilingual model on the paired data.

Our method aims to better align cross-script representations at sentence and token levels by leveraging transliteration
to a common script. We use [Glot500](https://github.com/cisnlp/Glot500), a mPLM pretrained on over
500 languages, as our source model, and fine-tune it on datasets formed from two different language groups (
**Mediterranean-Amharic-Farsi** and **South+East Asian Languages**
).
The languages in each group share areal features but are written in different scripts. We evaluate the fine-tuned models on
a series of downstream tasks, and show that after PPA, the models consistently outperform the original Glot500 model when
trasnferring from different source languages.

Paper on arXiv: https://arxiv.org/abs/2406.19759

<div style="text-align: center;">
    <img src="/plots/architecture.png" width="800" height="400" />
</div>

<div style="text-align: center;">
    <img src="/plots/languages.png" width="800" height="400" />
</div>

## Models on Huggingface
- Mediterranean-Amharic-Farsi: https://huggingface.co/orxhelili/translit_ppa_mediterranean
- South+East Asian Languages: https://huggingface.co/orxhelili/translit_ppa_se_asian


## Transliteration Data Generation

Given a text file containing sentences in different language-scripts, use the following command to generate a csv file
where each row is a pair of sentences (in its original script and in the Latin script).

```
python utils/preprocess_dataset.py
```

## Fine-tuning on the Paired Data

An example script for fine-tuning the model based on SLURM is provided in `sbatch_scripts/sbatch_finetune.sh`.

## Evaluation
Please refer to [Glot500](https://github.com/cisnlp/Glot500) and [SIB200](https://github.com/dadelani/sib-200)
for downloading the datasets used for evaluation.

Example scripts based on SLURM for running the different evaluations are provided in `sbatch_scripts/`.


## Citation

If you find our models useful for your research, please considering citing:

```
@article{xhelili2024breaking,
  title={Breaking the Script Barrier in Multilingual Pre-Trained Language Models with Transliteration-Based Post-Training Alignment},
  author={Xhelili, Orgest and Liu, Yihong and Sch{\"u}tze, Hinrich},
  journal={arXiv preprint arXiv:2406.19759},
  year={2024}
}
```


## Acknowledgements

This repository is built on top of [TransliCo](https://github.com/cisnlp/Translico) and [Glot500](https://github.com/cisnlp/Glot500)
