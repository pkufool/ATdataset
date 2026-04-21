#!/usr/bin/env python3
# Copyright  2025 Wei Kang (wkang@pku.edu.cn)
#
# See ../../../../LICENSE for clarification regarding multiple authors
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


from atdataset import ATDataloader, FbankExtractor
from tqdm import tqdm
import logging
import torch
import time
from functools import partial

import torch.multiprocessing as mp

from ssentencepiece import Ssentencepiece


def filter_func(sample):
    if sample["audio"].size(1) < 16000 * 5:
        return False
    return True


def map_func(sample, sp):
    sample["tokens"] = sp.encode(sample["text"])
    return sample


def worker_init_fn(worker_id):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    )

def main():
    feature_extractor = FbankExtractor(sample_rate=16000)
    mux_weights = [1350, 2000]
    sp = Ssentencepiece("librispeech-500")
    _map_func = partial(map_func, sp=sp)
    dl = ATDataloader(
        datasets=[
            "data/tars/aishell_train.lst",
            "data/tars/aishell2_train.lst",
        ],
        max_duration=100.0,
        max_samples=100,
        epoch_hours=sum(mux_weights),
        mux_weights=mux_weights,
        mux_intra_batch=True,
        feature_extractor=feature_extractor,
        filter_func=filter_func,
        map_func=_map_func,
        sample_rate=16000,
        num_copies=2,
        use_noise_augment=True,
        noise_manifest="data/tars/musan.lst",
        use_speed_perturb=True,
        use_volume_perturb=True,
        buffer_size=500,
        num_workers=2,
        worker_init_fn=worker_init_fn,
        device=torch.device("cpu"),
    )

    logging.info(f"Dataloader initialized: {dl}.")

    start = time.time()
    for i, batch in enumerate(tqdm(dl, total=len(dl))):
        logging.info(f"Batch {i}: ids={batch['ids']}")
        pass


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    )
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # The context might already be set.
    main()
