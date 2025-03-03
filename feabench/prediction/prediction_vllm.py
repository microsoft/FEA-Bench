import logging, json, os
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from .patch_process import postprocess

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from vllm import LLM, SamplingParams

# max sequence length input for LLMs
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "128000"))
MAX_GEN_LEN = int(os.environ.get("MAX_GEN_LEN", "4096"))


def get_model_limits_huggingface(model_name_or_path, max_limit=MAX_SEQ_LEN):
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)

    window_lengths = [max_limit]
    
    if hasattr(config, 'max_position_embeddings'):
        window_lengths.append(config.max_position_embeddings)
        logger.info(f"Detect the context limit using max_position_embeddings: {config.max_position_embeddings}")
    if hasattr(config, 'max_seq_length'):
        window_lengths.append(config.max_seq_length)
        logger.info(f"Detect the context limit using max_seq_length: {config.max_seq_length}")
            
    set_limit = min(window_lengths)
    logger.info(f"Set the context limit to: {set_limit}")
    return set_limit


def vllm_inference(
    test_dataset,
    model_name_or_path,
    output_file,
    model_args,
    existing_ids,
    input_text
):
    """
    Runs inference on a dataset using the openai API.

    Args:
    test_dataset (datasets.Dataset): The dataset to run inference on.
    model_name_or_path (str): The name or path of the model to use.
    output_file (str): The path to the output file.
    model_args (dict): A dictionary of model arguments.
    existing_ids (set): A set of ids that have already been processed.
    max_cost (float): The maximum cost to spend on inference.
    """
    # encoding = tiktoken.encoding_for_model(model_name_or_path)
    model_limit = get_model_limits_huggingface(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    test_dataset = test_dataset.filter(
        lambda x: len(tokenizer.encode(x[input_text])) <= model_limit,
        desc="Filtering",
        load_from_cache_file=False,
    )

    # model settings: default t=0.2 top_p=0.95
    temperature = model_args.pop("temperature", 0.2)
    top_p = model_args.pop("top_p", 0.95 if temperature > 0 else 1)
    logger.info(f"Using temperature={temperature}, top_p={top_p} in the LLMs' inference")
    basic_args = {
        "model_name_or_path": model_name_or_path,
    }
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=MAX_GEN_LEN)

    # initialize llm
    num_gpus = torch.cuda.device_count()
    logger.info(f"Current machine has {num_gpus} gpus. Will do tensor parallel across all gpus for one model.")
    llm = LLM(model=model_name_or_path, trust_remote_code=True, tensor_parallel_size=num_gpus)
    logger.info(f"Filtered to {len(test_dataset)} instances")
    for datum in tqdm(test_dataset, desc=f"Inference for {model_name_or_path}"):
        instance_id = datum["instance_id"]
        if instance_id in existing_ids:
            continue
        output_dict = {"instance_id": instance_id}
        output_dict.update(basic_args)

        inputs = datum[input_text]
        system_messages = inputs.split("\n", 1)[0]
        user_message = inputs.split("\n", 1)[1]

        response = llm.chat(
            messages=[
                {"role": "system", "content": system_messages},
                {"role": "user", "content": user_message},
            ],
            sampling_params=sampling_params,
            use_tqdm=False
        )[0]

        logger.info(
            f"input_tokens={len(response.prompt_token_ids)}, output_tokens={len(response.outputs[0].token_ids)}, limit={model_limit}"
        )
        output_dict["text"] = response.prompt
        completion = response.outputs[0].text

        output_dict["full_output"] = completion
        try:
            output_dict["model_patch"] = postprocess(completion, datum, input_text)
        except:
            logger.info(f"Invalid output which cannot be converted to patch. Instance {instance_id}")
            output_dict["model_patch"] = ""
        with open(output_file, "a+") as f:
            f.write(json.dumps(output_dict, ensure_ascii=False) + '\n')
            f.flush()
