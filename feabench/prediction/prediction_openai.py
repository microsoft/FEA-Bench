import logging, json, os
from tqdm import tqdm
from transformers import AutoTokenizer
import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from concurrent.futures import ProcessPoolExecutor, as_completed
from queue import Queue
from threading import Thread

from .patch_process import postprocess
from .tokenizer_online import gpt_tokenize, get_tokenizer_online

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_LIMITS = {
    "claude-instant-1": 100_000,
    "claude-2": 100_000,
    "claude-3-opus-20240229": 200_000,
    "claude-3-sonnet-20240229": 200_000,
    "claude-3-haiku-20240307": 200_000,
    "gpt-3.5-turbo-16k-0613": 16_385,
    "gpt-3.5-turbo-0613": 4_097,
    "gpt-3.5-turbo-1106": 16_385,
    "gpt-4-32k-0613": 32_768,
    "gpt-4-0613": 8_192,
    "gpt-4-1106-preview": 128_000,
    "gpt-4-0125-preview": 128_000,
    "gpt-4-turbo": 128_000,
    "o1-preview": 200_000,
    "o1_2024-12-17": 200_000,
    "o1-2024-12-17": 200_000,
    "o1-mini": 128_000,
    "o1-mini-2024-09-12": 128_000,
    "gpt-4o": 128_000,
    "gpt-4o-2024-05-13": 128_000,
    "deepseek-chat": 64_000,
    "deepseek-reasoner": 64_000,
}

# The cost per token for each model input.
MODEL_COST_PER_INPUT = {
    "claude-instant-1": 0.00000163,
    "claude-2": 0.00001102,
    "claude-3-opus-20240229": 0.000015,
    "claude-3-sonnet-20240229": 0.000003,
    "claude-3-haiku-20240307": 0.00000025,
    "gpt-3.5-turbo-16k-0613": 0.0000015,
    "gpt-3.5-turbo-0613": 0.0000015,
    "gpt-3.5-turbo-1106": 0.000001,
    "gpt-35-turbo-0613": 0.0000015,
    "gpt-35-turbo": 0.0000015,  # probably still 0613
    "gpt-4-0613": 0.00003,
    "gpt-4-32k-0613": 0.00006,
    "gpt-4-32k": 0.00006,
    "gpt-4-1106-preview": 0.00001,
    "gpt-4-0125-preview": 0.00001,
    "gpt-4": 0.00001,
    "gpt-4-turbo": 0.00001,
    "o1": 0.000015,
    "o1-preview": 0.000015,
    "o1_2024-12-17": 0.000015,
    "o1-2024-12-17": 0.000015,
    "o1-mini": 0.000003,
    "o1-mini-2024-09-12": 0.000003,
    "gpt-4o": 0.0000025,
    "gpt-4o-2024-05-13": 0.0000025,
    "deepseek-chat": 0.00000014,
    "deepseek-reasoner": 0.00000055,
}

# The cost per token for each model output.
MODEL_COST_PER_OUTPUT = {
    "claude-instant-1": 0.00000551,
    "claude-2": 0.00003268,
    "claude-3-opus-20240229": 0.000075,
    "claude-3-sonnet-20240229": 0.000015,
    "claude-3-haiku-20240307": 0.00000125,
    "gpt-3.5-turbo-16k-0613": 0.000002,
    "gpt-3.5-turbo-16k": 0.000002,
    "gpt-3.5-turbo-1106": 0.000002,
    "gpt-35-turbo-0613": 0.000002,
    "gpt-35-turbo": 0.000002,
    "gpt-4-0613": 0.00006,
    "gpt-4-32k-0613": 0.00012,
    "gpt-4-32k": 0.00012,
    "gpt-4-1106-preview": 0.00003,
    "gpt-4-0125-preview": 0.00003,
    "gpt-4": 0.00003,
    "gpt-4-turbo": 0.00003,
    "o1-preview": 0.00006,
    "o1_2024-12-17": 0.00006,
    "o1-2024-12-17": 0.00006,
    "o1-mini": 0.000012,
    "o1-mini-2024-09-12": 0.000012,
    "gpt-4o": 0.00001,
    "gpt-4o-2024-05-13": 0.00001,
    "deepseek-chat": 0.00000028,
    "deepseek-reasoner": 0.00000219,
}


def calc_cost(model_name, input_tokens, output_tokens):
    """
    Calculates the cost of a response from the openai API.

    Args:
    response (openai.ChatCompletion): The response from the API.

    Returns:
    float: The cost of the response.
    """
    cost = (
        MODEL_COST_PER_INPUT[model_name] * input_tokens
        + MODEL_COST_PER_OUTPUT[model_name] * output_tokens
    )
    logger.info(
        f"input_tokens={input_tokens}, output_tokens={output_tokens}, cost={cost:.2f}"
    )
    return cost



@retry(wait=wait_random_exponential(min=30, max=1200), stop=stop_after_attempt(10))
def call_chat(model_name_or_path, inputs, temperature, top_p, **model_args):
    """
    Calls the openai API to generate completions for the given inputs.

    Args:
    model_name_or_path (str): The name or path of the model to use.
    inputs (str): The inputs to generate completions for.
    use_azure (bool): Whether to use the azure API.
    temperature (float): The temperature to use.
    top_p (float): The top_p to use.
    **model_args (dict): A dictionary of model arguments.
    """
    system_messages = inputs.split("\n", 1)[0]
    user_message = inputs.split("\n", 1)[1]

    try:
        response = openai.chat.completions.create(
            model=model_name_or_path,
            messages=[
                {"role": "system", "content": system_messages},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            top_p=top_p,
            **model_args,
        )
        # logger.info(response.model)
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        cost = calc_cost(response.model, input_tokens, output_tokens)
        return response, cost
    except openai.BadRequestError as e:
        if e.code == "context_length_exceeded":
            logger.info("Context length exceeded")
            return "", 0.
        raise e
    except AttributeError as e:
        logger.info(f"Response is not well formed: {response}")
        raise e
    except Exception as e:
        logger.info(f"Error during chat: {e}")
        raise e
    

def process_datum_openai(datum, existing_ids, basic_args, input_text, temperature, top_p):
    instance_id = datum["instance_id"]
    if instance_id in existing_ids:
        return None
    output_dict = {"instance_id": instance_id}
    output_dict.update(basic_args)
    output_dict["text"] = f"{datum[input_text]}\n\n"

    response, cost = call_chat(
        output_dict["model_name_or_path"],
        output_dict["text"],
        temperature,
        top_p,
    )
    if response:
        completion = response.choices[0].message.content
        # output_dict["usage"] = dict(response.usage)
        if hasattr(response.choices[0].message, 'reasoning_content'):
            output_dict["reasoning_content"] = response.choices[0].message.reasoning_content
    else:
        logger.info(f"[Error] Empty response: '{response}'. May be caused by context length exceeded or key error.")
        completion = ""
    output_dict["full_output"] = completion

    try:
        output_dict["model_patch"] = postprocess(completion, datum, input_text)
    except Exception as e:
        logger.info(f"Invalid output which cannot be converted to patch. Instance {instance_id}, Error: {e}")
        output_dict["model_patch"] = ""
    return output_dict, cost


def write_to_file(queue, output_file):
    with open(output_file, "a+") as f:
        while True:
            output_dict = queue.get()
            if output_dict is None:
                break
            f.write(json.dumps(output_dict, ensure_ascii=False) + '\n')
            f.flush()


def openai_inference(
    test_dataset,
    model_name_or_path,
    output_file,
    model_args,
    existing_ids,
    max_cost,
    input_text,
    num_proc,
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
    encoding = get_tokenizer_online(model_name_or_path)
    test_dataset = test_dataset.filter(
        lambda x: gpt_tokenize(x[input_text], encoding) <= MODEL_LIMITS[model_name_or_path],
        desc="Filtering",
        load_from_cache_file=False,
    )
    openai_key = os.environ.get("OPENAI_API_KEY", None)
    if openai_key is None:
        raise ValueError(
            "Must provide an api key. Expected in OPENAI_API_KEY environment variable."
        )
    openai.api_key = openai_key
    logger.info(f"Using OpenAI key {'*' * max(0, len(openai_key)-5) + openai_key[-5:]}")

    # check if using azure
    azure_endpoint = os.environ.get("AZURE_ENDPOINT", None)
    if azure_endpoint:
        openai.api_type = "azure"
        openai.azure_endpoint = azure_endpoint
        api_version = os.environ.get("API_VERSION", None)
        if api_version is None:
            raise ValueError(
                f"You have provided azure endpoint url: {azure_endpoint}. So you must provide an api version. Expected in API_VERSION environment variable."
            )
        openai.api_version = api_version

    # check if base url is identified
    base_url = os.environ.get("OPENAI_BASE_URL", None)
    if base_url:
        openai.base_url = base_url
    
    temperature = model_args.pop("temperature", 0.2)
    top_p = model_args.pop("top_p", 0.95 if temperature > 0 else 1)
    logger.info(f"Using temperature={temperature}, top_p={top_p}")
    basic_args = {
        "model_name_or_path": model_name_or_path,
    }
    total_cost = 0
    logger.info(f"Filtered to {len(test_dataset)} instances")
    output_queue = Queue()

    # Start the thread to write to file
    writer_thread = Thread(target=write_to_file, args=(output_queue, output_file))
    writer_thread.start()

    with ProcessPoolExecutor(max_workers=num_proc) as executor:
        futures = [executor.submit(process_datum_openai, datum, existing_ids, basic_args, input_text, temperature, top_p) for datum in test_dataset]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Inference for {model_name_or_path}"):
            result = future.result()
            if result is not None:
                output_dict, cost = result
                total_cost += cost
                logger.info(f"Total Cost: {total_cost:.2f}")
                output_queue.put(output_dict)
                if max_cost is not None and total_cost >= max_cost:
                    logger.info(f"Reached max cost {max_cost}, exiting")
                    break

    # Stop the writer thread
    output_queue.put(None)
    writer_thread.join()
