# Multilingual Evaluation of Vision–Language Compositionality

This directory contains the code and data used for the
multilingual version of SugarCrepe and SugarCrepe++. It includes the prepared
English, German, Spanish, and Persian benchmark records, translation manifests
and exact-collision reports, the evaluator, the stored model predictions, and
the script that reconstructs the filtered result tables.

## Contents

```text
code/
  benchmark_config.py          Shared benchmark subsets and caption fields
  translate.py                  Translate SC or SC++ with TranslateGemma-4B
  evaluate.py                   Compute SC accuracy and SC++ ITT/TOT
  summarize_results.py          Apply union collision filtering and print tables
data/
  sc/{en,de,es,fa}/             Aligned SugarCrepe records
  scpp/{en,de,es,fa}/           Aligned SugarCrepe++ records
  val2017/000000412240.jpg       Image used only in the report's example figure
output/
  sc/                           Stored aggregate and per-sample SC predictions
  scpp/                         Stored aggregate and per-sample SC++ predictions
requirements.txt
```


## Environment

The experiments used one NVIDIA Tesla V100 GPU with 32 GB memory under Debian
GNU/Linux 13. The translation and evaluation code requires CUDA. Create an
environment and install the dependencies with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

TranslateGemma is gated on Hugging Face. Accept the model's terms and export a
token before regenerating translations:

```bash
export HF_TOKEN=YOUR_TOKEN
```

## COCO images

Model evaluation requires the complete [COCO-2017](https://cocodataset.org/#download) validation set.
Download and extract `val2017.zip` from the COCO website so that images appear
under `data/val2017/`, for example `data/val2017/000000412240.jpg`. Only the
single image used by the paper is included here.

## Regenerate translations

The released translations are already present under `data/`. To regenerate
them into a separate directory without overwriting the released records, run:

```bash
python code/translate.py \
  --task sc \
  --data-root data/sc/en \
  --output-root regenerated_data/sc \
  --languages de es fa \
  --records-per-batch 10

python code/translate.py \
  --task scpp \
  --data-root data/scpp/en \
  --output-root regenerated_data/scpp \
  --languages de es fa \
  --records-per-batch 10
```

Each translated subset is accompanied by an `*_issues.json`
file containing exact caption collisions.

## Re-run model evaluation

The following example reproduces one SugarCrepe run. Change `LANG`, model, and
checkpoint as shown below to reproduce the results.

```bash
LANG=fa

python code/evaluate.py \
  --task sugarcrepe \
  --model xlm-roberta-base-ViT-B-32 \
  --pretrained laion5b_s13b_b90k \
  --data_root data/sc/$LANG \
  --coco_image_root data/val2017 \
  --output output/sc/xlm-roberta-base/$LANG \
  --language $LANG \
  --checkpoint_every 200
```

For SugarCrepe++, change the task and data/output roots:

```bash
python code/evaluate.py \
  --task sugarcrepe_pp \
  --model xlm-roberta-base-ViT-B-32 \
  --pretrained laion5b_s13b_b90k \
  --data_root data/scpp/$LANG \
  --coco_image_root data/val2017 \
  --output output/scpp/xlm-roberta-base/$LANG \
  --language $LANG \
  --checkpoint_every 200
```

Evaluate `LANG` in `en de es fa`. The second reported checkpoint is:

```text
model:      xlm-roberta-large-ViT-H-14
pretrained: frozen_laion5b_s13b_b90k
output dir: xlm-roberta-large
```

All inference output is checkpointed atomically and can be resumed.

## Reproduce the reported tables

The stored predictions can reconstruct the report tables:

```bash
python code/summarize_results.py --output-root output
```

The script reads all `*_per_sample.json` files, forms the union of records with
an exact translated-caption collision in any language, excludes that same union
from every language, and reports per-subset accuracy plus the unweighted
macro-average of the displayed two-decimal subset scores, matching the paper's tables.
