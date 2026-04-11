# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Built-in maker functions that transform HuggingFace datasets into
conversation-style examples consumable by VLM processors.
"""

from dataclasses import dataclass
import glob
import json
import os
import random
from pathlib import Path
import re
from typing import Any, Dict, List

from datasets import concatenate_datasets, load_dataset
import yaml

from megatron.bridge.data.vlm_datasets.token_utils import json2token
from megatron.bridge.utils.common_utils import resolve_path

random.seed(42)

def make_rdr_dataset(
    path_or_dataset: str = "quintend/rdr-items", split: str = "train", **kwargs
) -> List[Dict[str, Any]]:
    """Load and preprocess the RDR dataset for image-to-text fine-tuning.

    Returns a list of examples with a "conversation" field that includes an image and text.
    """
    dataset = load_dataset(path_or_dataset, split=split)

    def format(example):
        return {
            "conversation": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": example["image"]},
                        {"type": "text", "text": "Describe this image."},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": example["text"]}],
                },
            ],
        }

    return [format(example) for example in dataset]


def make_cord_v2_dataset(
    path_or_dataset: str = "naver-clova-ix/cord-v2", split: str = "train", **kwargs
) -> List[Dict[str, Any]]:
    """Load and preprocess the CORD-V2 dataset for image-to-text fine-tuning."""
    dataset = load_dataset(path_or_dataset, split=split)

    def format(example):
        ground_truth = json.loads(example["ground_truth"])
        if "gt_parses" in ground_truth:
            assert isinstance(ground_truth["gt_parses"], list)
            gt_jsons = ground_truth["gt_parses"]
        else:
            assert "gt_parse" in ground_truth and isinstance(ground_truth["gt_parse"], dict)
            gt_jsons = [ground_truth["gt_parse"]]

        text = random.choice([json2token(gt_json, sort_json_key=True) for gt_json in gt_jsons])

        return {
            "conversation": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": example["image"]},
                        {"type": "text", "text": "Describe this image."},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": text}]},
            ],
        }

    return [format(example) for example in dataset]

# random pack var num of examples into a pack
def make_cord_v2_mocked_pack_dataset(
    path_or_dataset: str = "naver-clova-ix/cord-v2", split: str = "train", **kwargs
) -> List[Dict[str, Any]]:
    """Load and preprocess the CORD-V2 dataset for image-to-text fine-tuning."""
    dataset = load_dataset(path_or_dataset, split=split)

    def format(example):
        ground_truth = json.loads(example["ground_truth"])
        if "gt_parses" in ground_truth:
            assert isinstance(ground_truth["gt_parses"], list)
            gt_jsons = ground_truth["gt_parses"]
        else:
            assert "gt_parse" in ground_truth and isinstance(ground_truth["gt_parse"], dict)
            gt_jsons = [ground_truth["gt_parse"]]

        text = random.choice([json2token(gt_json, sort_json_key=True) for gt_json in gt_jsons])

        return {
            "conversation": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": example["image"]},
                        {"type": "text", "text": "Describe this image."},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": text}]},
            ],
        }

    orig_examples = [format(example) for example in dataset]
    packs = [[]]
    max_pack_size = 3
    current_pack_size = random.randint(1, max_pack_size)
    for example in orig_examples:
        current_pack = packs[-1]
        if len(current_pack) >= current_pack_size:
            packs.append([])
            current_pack_size = random.randint(1, max_pack_size)
            current_pack = packs[-1]
        
        current_pack.append(example)
        
    return packs

def make_medpix_dataset(
    path_or_dataset: str = "mmoukouba/MedPix-VQA", split: str = "train", **kwargs
) -> List[Dict[str, Any]]:
    """Load and preprocess the MedPix dataset for image-to-text fine-tuning."""
    dataset = load_dataset(path_or_dataset, split=split)

    def format(example):
        return {
            "conversation": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": example["image_id"]},
                        {"type": "text", "text": example["question"]},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": example["answer"]}]},
            ],
        }

    return [format(example) for example in dataset]


def make_raven_dataset(
    path_or_dataset: str = "HuggingFaceM4/the_cauldron",
    subset: str = "raven",
    split: str = "train",
    **kwargs,
) -> List[Dict[str, Any]]:
    """Load and preprocess the Raven subset from the Cauldron dataset.

    This subset follows the IDEFICS-style layout where each sample contains:
    - ``images``: a (possibly empty) list of PIL images
    - ``texts``: a list of conversation dictionaries. For Raven, ``texts[0]``
      is a *single* turn stored as a dictionary with two keys::

          {"user": "<question>", "assistant": "<answer>"}

      Only the first element is used.  The ``user`` string is taken as the
      user prompt, and ``assistant`` is the ground-truth answer.

    Conversation building policy:
    1. All images are placed at the beginning of the user turn followed by the
       textual prompt.
    2. The assistant turn contains the answer text.

    Examples missing either images or the required fields are filtered out.
    """
    if split != "train":
        raise ValueError("Raven dataset only supports train split. Please set `train.eval_iters=0`.")
    dataset = load_dataset(path_or_dataset, subset, split=split)

    def format(example):
        images = example.get("images", [])
        texts = example.get("texts", [])
        if not images or not texts or not isinstance(texts[0], dict):
            return None

        user_prompt = texts[0].get("user")
        assistant_answer = texts[0].get("assistant")
        if user_prompt is None or assistant_answer is None:
            return None

        user_content: List[Dict[str, Any]] = [{"type": "image", "image": img} for img in images]
        user_content.append({"type": "text", "text": user_prompt})

        assistant_content = [{"type": "text", "text": assistant_answer}]

        return {
            "conversation": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ]
        }

    formatted = (format(example) for example in dataset)
    # Filter out any None values from malformed rows.
    return [ex for ex in formatted if ex is not None]


def make_llava_video_178k_dataset(
    video_root_path: str,
    path_or_dataset: str = "lmms-lab/LLaVA-Video-178K",
    subsets: str | List[str] = "0_30_s_nextqa",
    split: str = "open_ended",
) -> List[Dict[str, Any]]:
    """Load and preprocess a subset of the *LLaVA-Video-178K* dataset.

    Each row contains:
    - ``video``: path or URL to the MP4 file.
    - ``conversations``: a **two-turn** list::

          [{"from": "human", "value": "<video>\n<question>"},
           {"from": "gpt",   "value": "<answer>"}]

      We map this schema to our internal multimodal conversation format:

      User turn  →  [video, user prompt]
      Assistant  →  answer text

    Note:
        Video files are assumed to be pre-downloaded and stored locally in the
        ``video_root_path`` directory. Rows with missing videos or empty
        conversations are filtered out from the final output.

    Args:
        video_root_path: Root directory where video files are stored locally.
        path_or_dataset: HF dataset path or local cache dir.
        subsets: Single subset name or list of the dataset's directory-style
            subsets to load.
        split: Split to load from the dataset. Note that "train" is automatically
            mapped to "open_ended".

    Returns:
        A list of dicts each containing a ``conversation`` field ready for
        downstream VLM processors.
    """
    if isinstance(subsets, str):
        subsets = [subsets]

    if split == "train":
        split = "open_ended"
    elif split in ("validation", "test"):
        raise ValueError("LLaVA-Video-178K dataset only supports train split. Please set `train.eval_iters=0`.")
    individual_datasets = [load_dataset(path_or_dataset, subset, split=split) for subset in subsets]
    dataset = concatenate_datasets(individual_datasets)

    # FIXME: right now we assume the video files are pre-downloaded and stored in the video_root_path
    # we need to modify this to download the video files from the hub if they are not present in the video_root_path

    def clean_prompt(val: str) -> str:
        # Remove placeholder tokens such as <image> or <video>
        val = val.replace("<image>", "").replace("<video>", "").strip()
        return val.lstrip("\n").rstrip()

    def format(example):
        video = example.get("video")
        convs = example.get("conversations", [])
        if video in (None, "") or not convs:
            return None

        conversation: List[Dict[str, Any]] = []

        first_human_handled = False
        for turn in convs:
            role = turn.get("from")
            value = turn.get("value", "")
            if not value:
                continue
            if role == "human":
                content: List[Dict[str, Any]] = []
                if not first_human_handled:
                    abs_path = resolve_path(Path(video_root_path) / video)
                    content.append({"type": "video", "path": str(abs_path)})
                    first_human_handled = True
                content.append({"type": "text", "text": clean_prompt(value)})
                conversation.append({"role": "user", "content": content})
            elif role == "gpt":
                conversation.append({"role": "assistant", "content": [{"type": "text", "text": value.strip()}]})

        if not conversation:
            return None

        return {"conversation": conversation}

    formatted = (format(ex) for ex in dataset)
    return [ex for ex in formatted if ex is not None]


def make_cv17_dataset(
    path_or_dataset: str = "ysdede/commonvoice_17_tr_fixed", split: str = "train", **kwargs
) -> List[Dict[str, Any]]:
    """Load and preprocess the CommonVoice 17 dataset for audio-to-text fine-tuning."""
    dataset = load_dataset(path_or_dataset, split=split)
    # Be robust to simple list-like datasets used in tests without `column_names` attr
    try:
        all_columns = dataset.column_names  # type: ignore[attr-defined]
    except Exception:
        first_example = dataset[0] if len(dataset) > 0 else {}
        all_columns = list(first_example.keys()) if isinstance(first_example, dict) else []
    if hasattr(dataset, "remove_columns"):
        columns_to_remove = [col for col in all_columns if col not in ["audio", "transcription"]]
        dataset = dataset.remove_columns(columns_to_remove)

    def format(example):
        return {
            "conversation": [
                {"role": "user", "content": "<|audio_1|>Transcribe the Turkish audio clip."},
                {"role": "assistant", "content": example["transcription"]},
            ],
            "audio": (example["audio"]["array"], example["audio"]["sampling_rate"]),
        }

    return [format(example) for example in dataset]

# jsonl dataset already in valid chatml format, either HF or OpenAI, no need furthr process, just pass through
def make_jsonl_zip_chatml_dataset(
    annotation_path: str, 
    image_zip_path: str,
    sampling_rate: float=1.0,
    head_n: int=None, # for determinstic selection like from dev/test sets or debug, note this is for every individual file before mixing, not for the final mixed dataset
) -> List[Dict[str, Any]]:
    """A simple maker that assumes the dataset is already in the expected ChatML format."""
    if head_n is not None and head_n > 0 and sampling_rate is not None and 0 < sampling_rate < 1.0:
        raise ValueError("Cannot specify both head_n and sampling_rate for make_jsonl_zip_chatml_dataset, please choose one or the other for deterministic selection.")
    
    examples = []
    with open(annotation_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                example = json.loads(line)
                ind_examples = example if isinstance(example, list) else [example]
                for e in ind_examples:
                    e['image_zip_path'] = image_zip_path
                    if 'conversation' not in e and 'messages' in e:
                        e['conversation'] = e.pop('messages')

                    if 'conversation' not in e:
                        raise ValueError(f"Example does not contain 'conversation' or 'messages' field: {e}")
                
                examples.append(example)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON line in dataset: {line.strip()}")

            if head_n is not None and head_n > 0 and len(examples) >= head_n:
                break

    if head_n is None and sampling_rate is not None and 0 < sampling_rate < 1.0:
        sample_size = int(len(examples) * sampling_rate)
        random.shuffle(examples)
        examples = examples[:sample_size]
    
    return examples

# jsonl-zip dataset mix
def make_zipmix_dataset(
    datamix_config_path: str=None, 
    annotation_path: str=None,
    image_zip_path: str=None,
    sampling_rate: float=None,
    head_n: int=None,
    split: str='train',
    **kwargs
) -> List[Dict[str, Any]]:
    if datamix_config_path is not None:
        mix_config = DatasetMixConfig(datamix_config_path)
        dataset_items = mix_config.get_dataset_items(split=split)
    
    else:
        dataset_items = None
        if annotation_path is not None and image_zip_path is not None:
            dataset_items = [
                DatasetItem(annotation_path=annotation_path, image_zip_path=image_zip_path, sampling_rate=sampling_rate)
            ]
        else:
            raise ValueError("Either datamix_config_path or both annotation_path and image_zip_path must be provided!")

    examples = []
    if dataset_items is not None:
        for item in dataset_items:
            item_examples = make_jsonl_zip_chatml_dataset(annotation_path=item.annotation_path, image_zip_path=item.image_zip_path, sampling_rate=item.sampling_rate, head_n=item.head_n)
            examples.extend(item_examples)
    return examples


@dataclass
class DatasetItem:
    annotation_path: str
    image_zip_path: str
    sampling_rate: float
    head_n: int = None

    def __post_init__(self):
        if self.sampling_rate is not None and (self.sampling_rate < 0 or self.sampling_rate > 1):
            raise ValueError(f"sampling_rate must be between 0 and 1!")

        if self.head_n is not None and self.head_n <= 0:
            raise ValueError(f"head_n must be a positive integer!")

        if self.sampling_rate is not None and 0 < self.sampling_rate < 1 and self.head_n is not None:
            raise ValueError(f"Cannot specify both head_n and sampling_rate for DatasetItem, please choose one or the other for deterministic selection.")


def glob_files_via_path_regex(dir: str, path_regex: str) -> List[str]:
    files = glob.glob(os.path.join(dir, '**/*'), recursive=True)
    files = [f for f in files if os.path.isfile(f)]
    # used for post-filtering files based on regex pattern
    if path_regex is not None:
        regex = re.compile(path_regex)
        files = [f for f in files if regex.search(f)]
    return sorted(files)

@dataclass
class DatasetConfig:
    # Three cases:
    # 1. annotations_path is DIR, images_path is DIR, and path_regex is applied to both to find pairs, assume both DIRs share the same structure except the extension
    # 2. annotations_path is DIR, images_path is ZIP, and path_regex is applied to annotations_dir to find annotation files, each annotation file corresponds to the same single ZIP file containing all images
    # 3. annotations_path is JSONL file, images_path is ZIP, and path_regex is not used, each line in the jsonl file contains the annotation info and the corresponding image file name in the ZIP file
    # Other than that, the config is considered invalid and should raise an error
    annotations_path: str
    images_path: str # if zip_path, we assume all annotations share the same single zip file; otherwise, we assume annotations_dir and iamges_dir have the same structure and share the same path regex
    path_regex: str # if images_dir_or_zip_path is a zip file path, this regex only applies to annotation files; otherwise, it applies to both annotation and image files to find the corresponding pairs
    sampling_rate: float = 1.0
    head_n: int = None # for determinstic selection like from dev/test sets or debug, note this is for every individual file before mixing, not for the final mixed dataset

    def __post_init__(self):
        if self.sampling_rate is not None and (self.sampling_rate < 0 or self.sampling_rate > 1):
            raise ValueError(f"sampling_rate must be between 0 and 1!")

        if self.head_n is not None and self.head_n <= 0:
            raise ValueError(f"head_n must be a positive integer!")

        if self.sampling_rate is not None and 0 < self.sampling_rate < 1 and self.head_n is not None:
            raise ValueError(f"Cannot specify both head_n and sampling_rate for DatasetConfig, please choose one or the other for deterministic selection!")
        
        if not (
            (Path(self.annotations_path).is_dir() and Path(self.images_path).is_dir() and self.path_regex)
            or (Path(self.annotations_path).is_dir() and self.images_path.endswith(".zip"))
            or (self.annotations_path.endswith(".jsonl") and self.images_path.endswith(".zip"))
        ):
            raise ValueError(
                f"Invalid DatasetConfig: annotations_path='{self.annotations_path}', images_path='{self.images_path}'.\n"
                f"Expected one of the following cases:\n"
        )

    def get_dataset_items(self) -> List[DatasetItem]:
        # Case 1
        if Path(self.annotations_path).is_dir() and Path(self.images_path).is_dir() and self.path_regex:
            # find all annotation files matching the regex
            annotation_paths = glob_files_via_path_regex(self.annotations_path, self.path_regex)
            dataset_items = []
            for annotation_path in annotation_paths:
                if not Path(annotation_path).is_file():
                    raise ValueError(f"Annotation path {annotation_path} is not a file!")
                rel_path = os.path.relpath(annotation_path, self.annotations_path)
                # assume image relative path only differ in extension i.e., .jsonl to .zip
                image_path = (Path(self.images_path) / rel_path).with_suffix('.zip')
                if not image_path.is_file():
                    raise ValueError(f"Image file {image_path} not found for annotation {annotation_path}!")
                
                dataset_items.append(DatasetItem(annotation_path=str(annotation_path), image_zip_path=str(image_path), sampling_rate=self.sampling_rate, head_n=self.head_n))
            
            return dataset_items
        # Case 2
        elif Path(self.annotations_path).is_dir() and self.images_path.endswith(".zip"):
            annotation_paths = glob_files_via_path_regex(self.annotations_path, self.path_regex)
            dataset_items = []
            for annotation_path in annotation_paths:
                if not Path(annotation_path).is_file():
                    raise ValueError(f"Annotation path {annotation_path} is not a file!")
                dataset_items.append(DatasetItem(annotation_path=str(annotation_path), image_zip_path=self.images_path, sampling_rate=self.sampling_rate, head_n=self.head_n))
            return dataset_items
        # Case 3
        elif self.annotations_path.endswith(".jsonl") and self.images_path.endswith(".zip"):
            if not Path(self.annotations_path).is_file():
                raise ValueError(f"Annotation path {self.annotations_path} is not a file!")
            if not Path(self.images_path).is_file():
                raise ValueError(f"Image zip path {self.images_path} is not a file!")
            return [DatasetItem(annotation_path=self.annotations_path, image_zip_path=self.images_path, sampling_rate=self.sampling_rate, head_n=self.head_n)]
        else:
            raise ValueError(f"Invalid DatasetConfig: annotations_path='{self.annotations_path}', images_path='{self.images_path}'.\n")


class DatasetMixConfig:
    def __init__(self, config_path: str):
        # yaml file
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)

        self.split_configs = {}
        for split, split_config in config_data.items():
            self.split_configs[split] = [DatasetConfig(**item) for item in split_config]
            
    def get_dataset_items(self, split: str) -> List[DatasetItem]:
        if split not in self.split_configs:
            raise ValueError(f"Split '{split}' not found in config!")
        dataset_items = []
        for config in self.split_configs[split]:
            dataset_items.extend(config.get_dataset_items())
        return dataset_items