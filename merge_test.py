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

# Function to generate text
def generate_text(prompt, max_length=8000, model_inp=model):
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
# prompt = "Give three tips for staying healthy"
prompt = """
<|im_start|>system
You are a helpful assistant with access to the following functions. Use them if required-                                                                                                                
{                                                                                                                                                                                                         
    "name": "calculate_mean",                                                                                                                                                                           
    "description": "Calculate the mean of a list of numbers",                                                                                                                                               "parameters": {                                                                                                                                                                                       
        "type": "object",                                                                                                                                                                                 
        "properties": {                                                                                                                                                                                   
            "numbers": {                                                                                                                                                                                  
                "type": "array",                                                                                                                                                                          
                "items": {                                                                                                                                                                                
                    "type": "number"                                                                                                                                                                      
                },
                "description": "A list of numbers" 
            }
        },
        "required": [
            "numbers"
        ]
    }
}
<|im_end|>
 <|im_start|>user
Hi, I have a list of numbers and I need to find the mean. Can you help me with that?<|im_end|>
<|im_start|>assistant
Of course, I can help you with that. Please provide me with the list of numbers. <|endoftext|><|im_end|>
<|im_start|>user
The numbers are 1, 2, 3, 1, 7, 4, 6, 3, 8.<|im_end|>
 <|im_start|>assistant
"""
generated_text = generate_text(prompt)
print("Generated Text:", generated_text)