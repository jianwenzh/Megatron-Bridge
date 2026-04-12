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
Core dataset types for conversation-style VLM examples.
"""

import io
from typing import Any, Callable, Dict, List, Optional
import zipfile

import torch
from PIL import Image
import copy

from megatron.bridge.data.vlm_datasets.collate import COLLATE_FNS


class VLMConversationDataset(torch.utils.data.Dataset):
    """Repeating wrapper over a list of HF-style conversation examples.

    - Each base example is expected to contain a "conversation" key following
      processor.apply_chat_template conventions. Optional modality fields like
      "audio" are passed through and consumed by the collate function.
    - Dataset length is set to a target length and indexes wrap around the
      underlying list to meet the requested size.
    - A `collate_fn` attribute is exposed so the framework can pass it to the
      DataLoader.
    """

    def __init__(
        self,
        base_examples: List[Dict[str, Any]]|List[List[Dict[str, Any]]], # support both unpacked and packed examples (list of examples or list of list of examples)
        target_length: int,
        processor: Any,
        collate_impl: Optional[Callable[[list, Any], Dict[str, torch.Tensor]]] = None,
        **collate_kwargs: Any,
    ) -> None:
        assert isinstance(base_examples, list) and len(base_examples) > 0, "base_examples must be a non-empty list"
        self._base_examples = base_examples
        self._length = int(max(0, target_length))
        self._processor = processor
        # Choose collate implementation by processor type name when not provided
        collate_key = type(processor).__name__ if processor is not None else "default"
        selected_impl = collate_impl or COLLATE_FNS.get(collate_key, COLLATE_FNS["default"])  # type: ignore[index]

        def _bound_collate(batch: list) -> Dict[str, torch.Tensor]:
            return selected_impl(batch, self._processor, **collate_kwargs)  # type: ignore[call-arg]

        self.collate_fn = _bound_collate
        
        # If an example specifies "image_zip_path", it assumes the images stored in the zip file. In this case, we assume all images in the same example share the single same zip file. We maintain a map from zip file path to ZipFile object to avoid repeatedly opening the same zip file for each example. The ZipFile objects will be closed when the dataset is destructed.
        self.image_zip_files_map = None # map from path to ZipFile object

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self._length == 0:
            raise IndexError("Empty dataset")
        base = self._base_examples[idx % len(self._base_examples)]
        if isinstance(base, list): # packed_example
            loaded_examples = []
            for example in base:
                image_zip_path = example.get('image_zip_path', None)
                if image_zip_path:
                    example = copy.deepcopy(example) # avoid modifying the original example in self._base_examples since it may be shared across multiple indexes
                    self._load_images_from_zip(example)
                loaded_examples.append(example)
            return loaded_examples
        else: # single example
            image_zip_path = base.get('image_zip_path', None)
            if image_zip_path:
                base = copy.deepcopy(base) # avoid modifying the original example in self._base_examples since it may be shared across multiple indexes
                self._load_images_from_zip(base)
            return base

    def _load_images_from_zip(self, example: Dict[str, Any]):
        image_zip_path = example['image_zip_path']
        if self.image_zip_files_map is None:
            self.image_zip_files_map = {}
        if image_zip_path not in self.image_zip_files_map:
            self.image_zip_files_map[image_zip_path] = zipfile.ZipFile(image_zip_path, 'r')
        zip_file = self.image_zip_files_map[image_zip_path]
        # load image for each image item
        messages = example['conversation']
        for message in messages:
            for item in message['content']:
                item_type = item['type']
                if item_type == 'image':
                    if item.get('path', None):
                        key = 'path'
                    elif item.get('url', None):
                        key = 'url'
                    elif item.get('image', None):
                        key = 'image'
                    else:
                        raise ValueError(f"No valid image key found in item: {item}")
                    image_path = item[key]
                    # load and open PIL Image object
                    with zip_file.open(image_path) as image_file:
                        image_bytes = image_file.read()
                        image = Image.open(io.BytesIO(image_bytes))
                        image.load()  # Ensure the image is fully loaded before closing the zip file
                        item['image'] = image 
                        if key != 'image':  # Remove the original path/url key if it's different from 'image'
                            del item[key]
                elif item_type == 'image_url': # openai format
                    image_path = item['image_url']['url']
                    with zip_file.open(image_path) as image_file:
                        image_bytes = image_file.read()
                        image = Image.open(io.BytesIO(image_bytes))
                        image.load()
                        item['type'] = 'image'
                        item['image'] = image
                        del item['image_url'] # Remove the original image_url key

    def __del__(self):
        if self.image_zip_files_map is not None:
            for zip_file in self.image_zip_files_map.values():
                zip_file.close()
