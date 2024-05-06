import pyarrow as pa
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

def read_and_decode(file_path, tokenizer, num_examples=5):
    with open(file_path, 'rb') as f:
        reader = pa.ipc.RecordBatchStreamReader(f)
        table = reader.read_all()
    # Convert the Arrow table to a pandas DataFrame for easier manipulation
    df = table.to_pandas()
    # Decode the first few examples
    for i in range(min(num_examples, len(df))):
        input_ids = df.iloc[i]['input_ids']
        labels = df.iloc[i]['labels']
        labels = np.where(labels == -100, tokenizer.eos_token_id, labels)
        decoded_text = tokenizer.decode(input_ids, skip_special_tokens=True)
        decoded_labels = tokenizer.decode(labels, skip_special_tokens=True)
        print("Text:", decoded_text)
        print("Labels:", decoded_labels)
        print("---")

# Example usage
version = 'af350ec8c160790979cdf3e836f37936'
file_path = f'/workspace/axolotl/last_run_prepared/{version}/data-00000-of-00002.arrow'
# file_path = f'/workspace/axolotl/devtools/last_run_prepared/4452740158a3f8ea9e6186dbd2ad5c75/data-00000-of-00002.arrow'

# Assuming the tokenizer is loaded as follows:
# from axolotl.common.cli import TrainerCliArgs, load_model_and_tokenizer
# cfg = None  # Define your configuration here
# cli_args = TrainerCliArgs()  # Adjust CLI arguments as needed
# model, tokenizer = load_model_and_tokenizer(cfg=cfg, cli_args=cli_args)

tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.1')

# Decode the top 5 examples
read_and_decode(file_path, tokenizer, num_examples=5)