from typing import List
import jsonlines
import numpy as np


def sample_data_for_annotation(testset_path: str, generation_paths: List, annotation_sample_size: int):
    count = 0
    with jsonlines.open(testset_path) as lines:
        for line in lines:
            count += 1
    selected_indices = set(np.random.choice(list(range(count)), annotation_sample_size, replace=False))
    for path in [testset_path] + generation_paths:
        outpath = path.replace('.jsonl', '.to_annotate.jsonl')
        _id = 0
        with jsonlines.open(path) as lines, jsonlines.open(outpath, 'w') as writer:
            for i, line in enumerate(lines):
                if i not in selected_indices:
                    continue
                line['_id'] = _id
                writer.write(line)
                _id += 1


def prepare_data(path, outpath):
    pass
