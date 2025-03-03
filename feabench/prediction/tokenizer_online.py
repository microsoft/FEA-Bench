import os
import logging
import tiktoken
from transformers import AutoTokenizer


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEEPSEEK_TOKENIZER_DIR = os.environ.get("DEEPSEEK_TOKENIZER_PATH", None)

def get_tokenizer_online(model_name):
    if model_name.startswith(("deepseek")):
        logger.info(f"Loading tokenizer from environ DEEPSEEK_TOKENIZER_PATH for {model_name}: {DEEPSEEK_TOKENIZER_DIR}")
        encoding = AutoTokenizer.from_pretrained(DEEPSEEK_TOKENIZER_DIR)
    else:
        try:
            encoding = tiktoken.encoding_for_model(model_name)
        except Exception as e:
            logger.info(f"Tokenizer for model {model_name} is not found in tiktoken with Error: \n{e}\n. Use GPT-4o tiktoken tokenizer instead.")
            encoding = tiktoken.encoding_for_model("gpt-4o")
    return encoding


def gpt_tokenize(string: str, encoding) -> int:
    """Returns the number of tokens in a text string."""
    if type(encoding) == tiktoken.Encoding:
        num_tokens = len(encoding.encode(string, disallowed_special=()))
    else:
        num_tokens = len(encoding.encode(string))
    return num_tokens
