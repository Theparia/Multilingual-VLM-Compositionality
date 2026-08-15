import argparse
import json
import os

import torch
from PIL import Image
from tqdm import tqdm

import open_clip

from benchmark_config import BENCHMARK_SUBSETS, CAPTION_FIELDS, EVALUATION_TO_BENCHMARK


def atomic_json_dump(value, path):
    """Write JSON atomically so interrupted runs do not leave partial files."""
    temporary_path = path + '.tmp'
    with open(temporary_path, 'w', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    os.replace(temporary_path, path)


class ClipEncoder:
    """OpenCLIP image and text encoder used for both benchmarks."""

    def __init__(self, model_name, pretrained, model_cache_dir, device):
        """Load an OpenCLIP checkpoint, tokenizer, and image transformation."""
        self.model, _, self.transform = open_clip.create_model_and_transforms(
            model_name=model_name, pretrained=pretrained, cache_dir=model_cache_dir, device=device
        )
        self.model = self.model.to(device).eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.device = device

    @torch.no_grad()
    def encode_text(self, text):
        """Encode and L2-normalize one caption with the model's text tower."""
        tokens = self.tokenizer([text]).to(self.device)
        return self.model.encode_text(tokens, normalize=True)

    @torch.no_grad()
    def encode_image(self, image):
        """Preprocess, encode, and L2-normalize one image with the vision tower."""
        return self.model.encode_image(self.transform(image).unsqueeze(0).to(self.device), normalize=True)


def load_checkpoint(per_sample_path):
    """Resume support: returns {subset: {sample_id: sample_result}} already computed."""
    if not os.path.exists(per_sample_path):
        return {}
    with open(per_sample_path, encoding='utf-8') as handle:
        details = json.load(handle)
    return details.get('subsets', {})


def write_checkpoint(aggregate_paths, per_sample_path, metadata, dataset, per_sample_results):
    """Persist aggregate metrics and per-sample results for resumable evaluation."""
    for metric_key, aggregate_path in aggregate_paths.items():
        metrics = {}
        for c, data_dict in dataset.items():
            subset_results = per_sample_results.get(c, {})
            if subset_results:
                correct_cnt = sum(1 for r in subset_results.values() if r[metric_key])
                metrics[c] = correct_cnt / len(data_dict)
        atomic_json_dump(metrics, aggregate_path)
    details = {**metadata, 'subsets': per_sample_results}
    atomic_json_dump(details, per_sample_path)


def score_sugarcrepe(encoder, image_root, data):
    """Score one SugarCrepe positive--negative caption pair against its image."""
    with Image.open(os.path.join(image_root, data['filename'])) as image:
        image = image.convert('RGB')
        img_emb = encoder.encode_image(image)
    pos_emb = encoder.encode_text(data['caption'])
    neg_emb = encoder.encode_text(data['negative_caption'])
    pos_score = (pos_emb @ img_emb.t()).item()
    neg_score = (neg_emb @ img_emb.t()).item()
    margin = pos_score - neg_score
    return {
        'positive_score': pos_score,
        'negative_score': neg_score,
        'margin': margin,
        'correct': bool(margin > 0),
        'tie': bool(margin == 0),
    }


def score_sugarcrepe_pp(encoder, image_root, data):
    """Compute ITT and TOT decisions for one SugarCrepe++ caption triplet."""
    p1_emb = encoder.encode_text(data['caption'])
    p2_emb = encoder.encode_text(data['caption2'])
    neg_emb = encoder.encode_text(data['negative_caption'])

    # TOT is text-only and comes from the same P1/P2/N embeddings used by ITT.
    p1_ref = (p1_emb @ neg_emb.t()).item()
    p2_ref = (p2_emb @ neg_emb.t()).item()
    p1_p2 = (p1_emb @ p2_emb.t()).item()
    result = {
        'p1_ref': p1_ref,
        'p2_ref': p2_ref,
        'p1_p2': p1_p2,
        'TOT': bool(p1_p2 > p1_ref and p1_p2 > p2_ref),
    }

    # ITT requires both positives to beat the negative against the image.
    with Image.open(os.path.join(image_root, data['filename'])) as image:
        img_emb = encoder.encode_image(image.convert('RGB'))
    image_p1 = (p1_emb @ img_emb.t()).item()
    image_p2 = (p2_emb @ img_emb.t()).item()
    image_neg = (neg_emb @ img_emb.t()).item()
    result.update({
        'image_p1': image_p1,
        'image_p2': image_p2,
        'image_neg': image_neg,
        'image_p1_neg_correct': bool(image_p1 > image_neg),
        'image_p2_neg_correct': bool(image_p2 > image_neg),
        'ITT': bool(image_p1 > image_neg and image_p2 > image_neg),
    })

    return result


def evaluate(image_root, dataset, encoder, task, metadata,
             aggregate_paths, per_sample_path, checkpoint_every=200):
    """Evaluate every pending sample and periodically save resumable results.
    Return aggregate metrics together with the complete per-sample records."""
    caption_fields = CAPTION_FIELDS[EVALUATION_TO_BENCHMARK[task]]
    score_fn = score_sugarcrepe if task == 'sugarcrepe' else score_sugarcrepe_pp

    per_sample_results = load_checkpoint(per_sample_path)

    for c, data_dict in dataset.items():
        subset_results = per_sample_results.setdefault(c, {})
        already_done = set(subset_results)
        remaining_ids = [sid for sid in data_dict if sid not in already_done]
        if not remaining_ids:
            print(f'{c}: already complete ({len(subset_results)}/{len(data_dict)}), skipping')
            continue
        if already_done:
            print(f'{c}: resuming, {len(already_done)}/{len(data_dict)} already done')

        since_checkpoint = 0
        progress = tqdm(remaining_ids, desc=f'evaluating {c}', initial=len(already_done), total=len(data_dict))
        for sample_id in progress:
            data = data_dict[sample_id]
            scores = score_fn(encoder, image_root, data)

            sample_result = {
                'sample_id': str(sample_id),
                'subset': c,
                'data_file': f'{c}.json',
                'filename': data['filename'],
                **{field: data[field] for field in caption_fields},
                **scores,
            }
            for field in caption_fields:
                source_field = f'source_{field}'
                if source_field in data:
                    sample_result[source_field] = data[source_field]
            subset_results[str(sample_id)] = sample_result

            since_checkpoint += 1
            if since_checkpoint >= checkpoint_every:
                write_checkpoint(aggregate_paths, per_sample_path, metadata, dataset, per_sample_results)
                since_checkpoint = 0

        # Always checkpoint at the end of a subset, so a kill between subsets loses nothing.
        write_checkpoint(aggregate_paths, per_sample_path, metadata, dataset, per_sample_results)

    metrics = {
        metric_key: {c: sum(1 for r in per_sample_results[c].values() if r[metric_key]) / len(dataset[c]) for c in dataset}
        for metric_key in aggregate_paths
    }
    return metrics, per_sample_results


def is_complete(per_sample_path, dataset):
    """Return whether a checkpoint contains a result for every dataset example."""
    subsets = load_checkpoint(per_sample_path)
    return all(len(subsets.get(c, {})) == len(data_dict) for c, data_dict in dataset.items())


def run(args, pretrained, device, dataset):
    """Load the requested checkpoint and run one benchmark evaluation."""
    result_stem = f'{args.model}-{pretrained}'
    per_sample_path = os.path.join(args.output, f'{result_stem}_per_sample.json')

    if is_complete(per_sample_path, dataset):
        print(f"Skipping {result_stem}: already complete at {per_sample_path}")
        return

    print(f"Evaluating {result_stem} (task={args.task})")
    encoder = ClipEncoder(args.model, pretrained, args.model_cache_dir, device)

    if args.task == 'sugarcrepe':
        aggregate_paths = {'correct': os.path.join(args.output, f'{result_stem}.json')}
        decision_rule = 'correct when positive_score > negative_score; ties are incorrect'
    else:
        aggregate_paths = {
            'ITT': os.path.join(args.output, f'ITT_{result_stem}.json'),
            'TOT': os.path.join(args.output, f'TOT_{result_stem}.json'),
        }
        decision_rule = (
            'ITT=1 when both sim(image,P1) and sim(image,P2) exceed sim(image,N); '
            'TOT=1 when sim(P1,P2) exceeds both sim(P1,N) and sim(P2,N), using the same text tower, no image'
        )
    metadata = {
        'model': args.model,
        'pretrained': pretrained,
        'task': args.task,
        'language': args.language,
        'data_root': args.data_root,
        'decision_rule': decision_rule,
    }
    metrics, _ = evaluate(
        args.coco_image_root, dataset, encoder, args.task, metadata,
        aggregate_paths, per_sample_path, checkpoint_every=args.checkpoint_every,
    )
    print(metrics)
    print(f'Results saved to: {per_sample_path} and {list(aggregate_paths.values())}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=tuple(EVALUATION_TO_BENCHMARK), default='sugarcrepe')
    parser.add_argument('--model', type=str, default="RN50", help="OpenCLIP model name")
    parser.add_argument('--pretrained', type=str, default="openai", help="OpenCLIP checkpoint name")
    parser.add_argument('--model_cache_dir', default=None, type=str, help="Directory to where downloaded models are cached")
    parser.add_argument('--output', type=str, default=None, help="Directory to where results are saved")

    parser.add_argument('--coco_image_root', type=str, required=True, help="Directory containing COCO images")
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--subsets', nargs='+', default=None, help="Override the default subset list for --task")
    parser.add_argument('--language', type=str, default=None, help="Language label stored in per-sample result metadata")
    parser.add_argument('--checkpoint_every', type=int, default=200, help="Flush results to disk after this many newly-scored samples")

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    subset_names = args.subsets or BENCHMARK_SUBSETS[EVALUATION_TO_BENCHMARK[args.task]]
    dataset = {c: json.load(open(f'{args.data_root}/{c}.json', 'r', encoding='utf-8')) for c in subset_names}
    if args.task == 'sugarcrepe_pp':
        missing_caption2 = {c for c, d in dataset.items() if not all('caption2' in r for r in d.values())}
        if missing_caption2:
            raise ValueError(f"--task sugarcrepe_pp requires 'caption2' in every record; missing in subsets: {missing_caption2}")

    os.makedirs(args.output, exist_ok=True)

    run(args, args.pretrained, device, dataset)
