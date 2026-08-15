from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
from itertools import combinations
from pathlib import Path

import torch
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)

from benchmark_config import BENCHMARK_SUBSETS, CAPTION_FIELDS

TARGET_CODES = {
    "fa": "fa-IR",
    "de": "de-DE",
    "es": "es-ES",
}


def atomic_json_dump(value: object, path: Path) -> None:
    """Write JSON atomically so interrupted runs do not leave partial files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    os.replace(temporary_path, path)


def normalized_text(text: str) -> str:
    """Normalize text for robust equality checks between translated captions."""
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip().casefold()


class TranslateGemmaTranslator:
    def __init__(
        self,
        model_id: str,
        load_in_8bit: bool,
        hf_token: str | None,
    ) -> None:
        """Load TranslateGemma and its processor on an available CUDA device."""
        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA GPU is required for this notebook pipeline.")
        self.dtype = torch.bfloat16
        self.processor = AutoProcessor.from_pretrained(model_id, token=hf_token)
        self.tokenizer = self.processor.tokenizer
        # Decoder-only models must be left padded for correct batched generation.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        model_kwargs = {
            "device_map": "auto",
            "low_cpu_mem_usage": True,
            "token": hf_token,
            "dtype": self.dtype,
        }
        if load_in_8bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True
            )

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            **model_kwargs,
        ).eval()
        self.model_id = model_id

    @torch.inference_mode()
    def translate(self, texts: list[str], target_code: str) -> list[str]:
        """Translate a batch of English strings into the requested target language."""
        encoded = []
        for text in texts:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "source_lang_code": "en",
                            "target_lang_code": target_code,
                            "text": text,
                        }
                    ],
                }
            ]
            single = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            encoded.append(
                {
                    "input_ids": single["input_ids"][0],
                    "attention_mask": single["attention_mask"][0],
                }
            )

        inputs = self.tokenizer.pad(encoded, padding=True, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        prompt_length = inputs["input_ids"].shape[1]

        generated = self.model.generate(
            **inputs,
            do_sample=False,
            num_beams=1,
            max_new_tokens=128,
        )
        generated = generated[:, prompt_length:]
        translations = self.processor.batch_decode(
            generated,
            skip_special_tokens=True,
        )
        translations = [translation.strip() for translation in translations]
        if any(not translation for translation in translations):
            empty_indices = [i for i, t in enumerate(translations) if not t]
            raw = self.processor.batch_decode(generated, skip_special_tokens=False)
            raise RuntimeError(
                "TranslateGemma returned an empty translation for input(s) "
                f"{[texts[i] for i in empty_indices]}. Raw decode: "
                f"{[raw[i] for i in empty_indices]}"
            )
        return translations


def translate_subset(
    translator: TranslateGemmaTranslator,
    source_path: Path,
    output_path: Path,
    target_code: str,
    fields: tuple[str, ...],
    records_per_batch: int,
    max_samples: int | None = None,
) -> int:
    """Translate all pending records in one subset and persist resumable output.
    The function also writes a report of caption pairs that collapse after normalization.
    """
    with source_path.open(encoding="utf-8") as handle:
        source_data = json.load(handle)

    if output_path.exists():
        with output_path.open(encoding="utf-8") as handle:
            translated_data = json.load(handle)
    else:
        translated_data = {}

    pending_ids = [sample_id for sample_id in source_data if sample_id not in translated_data]
    if max_samples is not None:
        pending_ids = pending_ids[:max_samples]
    total = len(source_data)
    failed_batches: list[dict] = []
    processed_count = 0
    n = len(fields)

    for start in range(0, len(pending_ids), records_per_batch):
        batch_ids = pending_ids[start : start + records_per_batch]
        texts: list[str] = []
        for sample_id in batch_ids:
            record = source_data[sample_id]
            texts.extend(record[field] for field in fields)

        try:
            translations = translator.translate(texts, target_code)
        except Exception as error:  # noqa: BLE001 - keep the run alive across batches
            print(
                f"{output_path.name}: batch {batch_ids} failed ({error}); "
                "skipping for now, will retry on next run",
                flush=True,
            )
            failed_batches.append({"sample_ids": batch_ids, "error": str(error)})
            atomic_json_dump(
                failed_batches, output_path.with_name(output_path.stem + "_errors.json")
            )
            continue
        for index, sample_id in enumerate(batch_ids):
            source_record = source_data[sample_id]
            translated_fields = {
                field: translations[n * index + j] for j, field in enumerate(fields)
            }
            translated_data[sample_id] = {
                **source_record,
                **{f"source_{field}": source_record[field] for field in fields},
                **translated_fields,
            }

        # A completed record is always persisted together, making interruption safe.
        atomic_json_dump(translated_data, output_path)
        processed_count += len(batch_ids)
        print(f"{output_path.name}: {len(translated_data)}/{total}", flush=True)

    # Pairwise collapse check across whichever fields were translated (2 fields
    # for --task sc: one check; 3 fields for --task scpp: all 3 pairs).
    issues = []
    for sample_id, record in translated_data.items():
        normalized = {field: normalized_text(record[field]) for field in fields}
        for field_a, field_b in combinations(fields, 2):
            if normalized[field_a] == normalized[field_b]:
                issues.append(
                    {
                        "sample_id": sample_id,
                        "reason": f"{field_a}_collapsed_with_{field_b}",
                        field_a: record[field_a],
                        field_b: record[field_b],
                    }
                )
    atomic_json_dump(issues, output_path.with_name(output_path.stem + "_issues.json"))
    return processed_count


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line options for the translation pipeline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=list(BENCHMARK_SUBSETS), default="sc", help="sc: caption+negative_caption over the 7 SugarCrepe subsets. scpp: caption+negative_caption+caption2 over the 5 SugarCrepe++ subsets.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--languages", nargs="+", choices=TARGET_CODES, default=list(TARGET_CODES))
    parser.add_argument(
        "--subsets",
        nargs="+",
        choices=BENCHMARK_SUBSETS["sc"],
        default=None,
        help="Subsets to translate (default: all subsets for --task)",
    )
    parser.add_argument("--records-per-batch", type=int, default=10)
    parser.add_argument("--model-id", default="google/translategemma-4b-it")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument(
        "--limit-samples",
        type=int,
        default=None,
        help="Cap total samples processed per language, for quick test runs.",
    )
    args = parser.parse_args()

    if args.subsets is None:
        args.subsets = list(BENCHMARK_SUBSETS[args.task])
    else:
        invalid = set(args.subsets) - set(BENCHMARK_SUBSETS[args.task])
        if invalid:
            parser.error(f"--subsets {sorted(invalid)} not valid for --task {args.task} (allowed: {BENCHMARK_SUBSETS[args.task]})")

    return args


def main() -> None:
    """Run multilingual translation for the requested benchmark, languages, and subsets."""
    args = parse_args()
    fields = CAPTION_FIELDS[args.task]
    translator = TranslateGemmaTranslator(
        model_id=args.model_id,
        load_in_8bit=args.load_in_8bit,
        hf_token=args.hf_token,
    )

    print(
        f"GPU: {torch.cuda.get_device_name(0)}; dtype: {translator.dtype}; "
        f"model: {args.model_id}; task: {args.task}; fields: {fields}"
    )
    for language in args.languages:
        language_output = args.output_root / language
        manifest = {
            "model": args.model_id,
            "task": args.task,
            "fields": list(fields),
            "source_language": "en",
            "target_language": TARGET_CODES[language],
            "requested_subsets": args.subsets,
            "decoding": {
                "do_sample": False,
                "num_beams": 1,
                "max_new_tokens": 128,
            },
            "load_in_8bit": args.load_in_8bit,
        }
        atomic_json_dump(manifest, language_output / "translation_manifest.json")

        remaining_samples = args.limit_samples
        for subset in args.subsets:
            if remaining_samples is not None and remaining_samples <= 0:
                break
            source_path = args.data_root / f"{subset}.json"
            if not source_path.exists():
                raise FileNotFoundError(source_path)
            print(f"Translating {subset} ({', '.join(fields)}) to {TARGET_CODES[language]}")
            processed = translate_subset(
                translator=translator,
                source_path=source_path,
                output_path=language_output / f"{subset}.json",
                target_code=TARGET_CODES[language],
                fields=fields,
                records_per_batch=args.records_per_batch,
                max_samples=remaining_samples,
            )
            if remaining_samples is not None:
                remaining_samples -= processed


if __name__ == "__main__":
    main()
