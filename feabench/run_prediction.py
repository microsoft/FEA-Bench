#!/usr/bin/env python3

"""This python script is designed to run inference on a dataset using either the OpenAI or Anthropic API, depending on the model specified. 
It sorts instances by length and continually writes the outputs to a specified file, so that the script can be stopped and restarted without losing progress.
"""

import json
import os
import dotenv
from pathlib import Path
import numpy as np
from datasets import load_dataset, load_from_disk
from feabench.prediction.patch_process import postprocess
from argparse import ArgumentParser
import logging


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
dotenv.load_dotenv()


def parse_model_args(model_args):
    """
    Parses a string of model arguments and returns a dictionary of keyword arguments.

    Args:
        model_args (str): A string of comma-separated key-value pairs representing model arguments.

    Returns:
        dict: A dictionary of keyword arguments parsed from the input string.
    """
    kwargs = dict()
    if model_args is not None:
        for arg in model_args.split(","):
            key, value = arg.split("=")
            # infer value type
            if value in {"True", "False"}:
                kwargs[key] = value == "True"
            elif value.isnumeric():
                kwargs[key] = int(value)
            elif value.replace(".", "", 1).isnumeric():
                kwargs[key] = float(value)
            elif value in {"None"}:
                kwargs[key] = None
            elif value in {"[]"}:
                kwargs[key] = []
            elif value in {"{}"}:
                kwargs[key] = {}
            elif value.startswith("'") and value.endswith("'"):
                kwargs[key] = value[1:-1]
            elif value.startswith('"') and value.endswith('"'):
                kwargs[key] = value[1:-1]
            else:
                kwargs[key] = value
    return kwargs


def main(
    dataset_name_or_path,
    split,
    input_text,
    model_type,
    model_name_or_path,
    shard_id,
    num_shards,
    output_dir,
    model_args,
    max_cost,
    num_proc,
):
    if shard_id is None and num_shards is not None:
        logger.warning(
            f"Received num_shards={num_shards} but shard_id is None, ignoring"
        )
    if shard_id is not None and num_shards is None:
        logger.warning(f"Received shard_id={shard_id} but num_shards is None, ignoring")
    model_args = parse_model_args(model_args)
    model_nickname = model_name_or_path
    if "checkpoint" in Path(model_name_or_path).name:
        model_nickname = Path(model_name_or_path).parent.name
    else:
        model_nickname = Path(model_name_or_path).name
    output_file = f"{model_nickname}__{dataset_name_or_path.split('/')[-1]}__{split}"
    if shard_id is not None and num_shards is not None:
        output_file += f"__shard-{shard_id}__num_shards-{num_shards}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = Path(output_dir, output_file + ".jsonl")
    logger.info(f"Will write to {output_file}")
    existing_ids = set()
    if os.path.exists(output_file):
        with open(output_file) as f:
            for line in f:
                data = json.loads(line)
                instance_id = data["instance_id"]
                existing_ids.add(instance_id)
    logger.info(f"Read {len(existing_ids)} already completed ids from {output_file}")
    if Path(dataset_name_or_path).exists():
        dataset = load_from_disk(dataset_name_or_path)
    else:
        dataset = load_dataset(dataset_name_or_path)
    if not split in dataset:
        raise ValueError(f"Invalid split {split} for dataset {dataset_name_or_path}")
    dataset = dataset[split]
    lens = np.array(list(map(len, dataset[input_text])))
    dataset = dataset.select(np.argsort(lens))
    if len(existing_ids) > 0:
        dataset = dataset.filter(
            lambda x: x["instance_id"] not in existing_ids,
            desc="Filtering out existing ids",
            load_from_cache_file=False,
        )
    if shard_id is not None and num_shards is not None:
        dataset = dataset.shard(num_shards, shard_id, contiguous=True)
    logger.info(f"{len(dataset)} instances to be inferred.")
    inference_args = {
        "test_dataset": dataset,
        "model_name_or_path": model_name_or_path,
        "output_file": output_file,
        "model_args": model_args,
        "existing_ids": existing_ids,
        "max_cost": max_cost,
        "input_text": input_text,
        "num_proc": num_proc,
    }
    if model_type == "openai":
        from .prediction.prediction_openai import openai_inference
        openai_inference(**inference_args)
    elif model_type == "vllm":
        from .prediction.prediction_vllm import vllm_inference
        inference_args.pop("max_cost")
        inference_args.pop("num_proc")
        vllm_inference(**inference_args)
    else:
        raise NotImplementedError(f"{model_type} has not been supported for now.")
    logger.info(f"Done!")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset_name_or_path",
        type=str,
        required=True,
        help="HuggingFace dataset name or local path",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to use",
    )
    parser.add_argument(
        "--input_text",
        type=str,
        default="text",
        help="the item key of dataset as input text to LLMs",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        help="Type of LLMs. For now support 'openai' or 'vllm' two types.",
        choices=['openai', 'vllm'],
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Name of API model or VLLM. Update MODEL* constants in this file to add new models.",
    )
    parser.add_argument(
        "--shard_id",
        type=int,
        default=None,
        help="Shard id to process. If None, process all shards.",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=None,
        help="Number of shards. If None, process all shards.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        required=True,
        help="Path to the output file.",
    )
    parser.add_argument(
        "--model_args",
        type=str,
        default=None,
        help="List of model arguments separated by commas. (e.g. 'top_p=0.95,temperature=0.70')",
    )
    parser.add_argument(
        "--max_cost",
        type=float,
        default=None,
        help="Maximum cost to spend on inference.",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=1,
        help="Parallel API call for openai-type api.",
    )
    args = parser.parse_args()
    main(**vars(args))
