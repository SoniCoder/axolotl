from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig, LoraConfig, get_peft_model

# Path to your fine-tuned model and tokenizer
model_path = "qlora-out-mistral-fun-call"

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

merged_model_path = "./merged_model"
base_model.save_pretrained(merged_model_path)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.save_pretrained(merged_model_path)