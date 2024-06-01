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
        # Check that input and labels have the same number of tokens
        assert len(input_ids) == len(labels)
        
        # Replace -100 in labels with None for clarity in output
        labels_highlighted = [(label if label != -100 else None) for label in labels]
        
        # Decode texts
        # decoded_text = tokenizer.decode(input_ids, skip_special_tokens=True)
        input_labels = [tokenizer.decode([input_id], skip_special_tokens=True) for input_id in input_ids]
        decoded_labels = [tokenizer.decode([label], skip_special_tokens=True) if label is not None else None for label in labels_highlighted]
        
        # Combine text and labels for highlighted output
        combined_output = []
        gpt_iterating = False
        index_start = 0 # we are starting with the first index
        # once we find a stop we'll decode text up to that point, append it to the combined output
        # and add a start as necessary
        tokens_acc = []
        for input_id, label in zip(input_ids, decoded_labels):
            if label is None: # masked input in this case print as it is
                # if we were gpt iterating earlier but not that input is starting
                # let us append it with a star in front
                if gpt_iterating:
                    gpt_iterating = False
                    # we have encountered a gpt output
                    # let us dump all the text we have so far in tokens_acc, then a *
                    # and then continue accumulating tokens
                    decoded = tokenizer.decode(tokens_acc, skip_special_tokens=True)
                    combined_output.append(decoded)
                    combined_output.append("*")
                    tokens_acc = [input_id]
                else:
                    tokens_acc.append(input_id)  # Masked tokens are enclosed in brackets
            else:
                # token is unmasked, likely output starting
                # let us append it with a star in front
                if not gpt_iterating:
                    gpt_iterating = True
                    decoded = tokenizer.decode(tokens_acc, skip_special_tokens=True)
                    combined_output.append(decoded)
                    combined_output.append("*")
                    tokens_acc = [input_id]
                else:
                    tokens_acc.append(input_id)  # Unmasked tokens are shown with labels

        # Append the last part of the text
        decoded = tokenizer.decode(tokens_acc, skip_special_tokens=True)
        combined_output.append(decoded)

        print("Combined Text and Labels:")
        print("".join(combined_output))
        # print(decoded_labels)
        print("---")

# Example usage
version = 'a7cab93baaee6394628bb7c8fd0bec5d'
file_path = f'/workspace/axolotl/last_run_prepared/{version}/data-00000-of-00002.arrow'

tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.1')

# Decode the top 5 examples
read_and_decode(file_path, tokenizer, num_examples=5)