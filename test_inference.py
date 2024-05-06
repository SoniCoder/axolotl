from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig, LoraConfig, get_peft_model

# Path to your fine-tuned model and tokenizer
model_path = "qlora-out-mistral"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

config = PeftConfig.from_pretrained(model_path)

# Load the fine-tuned model
model = AutoModelForCausalLM.from_pretrained(model_path, load_in_4bit=True, device_map="auto")
# model = AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-v0.1', load_in_4bit=True, device_map="auto")
# model = PeftModel.from_pretrained(model, model_path)
# model = model.merge_and_unload()

# Function to generate text
def generate_text(prompt, max_length=100, model_inp=model):
    # Encode the prompt into tokens
    # input_ids = tokenizer.encode(prompt, return_tensors='pt')
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
    # Generate output from the model
    # output_ids = model_inp.generate(input_ids, max_length=max_length)
    output_ids = model_inp.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],  # Pass attention mask
        max_length=max_length,
    )
    # Decode the output tokens to a string
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text

# Example usage
prompt = "Give three tips for staying healthy"
generated_text = generate_text(prompt)
print("Generated Text:", generated_text)