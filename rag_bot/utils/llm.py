from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id_path = "local_models/phi-1_5" 
#microsoft/phi-1_5 for speed than phi-2 but less accurate
#TinyLlama/TinyLlama-1.1B-Chat-v1.0 for faster inference but less accurate
#phi-2 is a large model so it takes more time to load and run in first time 

tokenizer = AutoTokenizer.from_pretrained(model_id_path,local_files_only=True)

llm = AutoModelForCausalLM.from_pretrained(
    model_id_path,
    do_sample=False,
    torch_dtype=torch.float32,
    local_files_only=True,
    device_map={"": "cpu"} 
)


def build_prompt(query, chunks):
    context = "\n\n".join([f"{i+1}. {c['text']}" for i, c in enumerate(chunks)])
    return f"""Answer the question based on the context below. 
If the answer is not present, say "I don't know. ".
Context:
{context}

Question: {query}
Answer:"""

def generate_answer(query, chunks):
    prompt = build_prompt(query, chunks)
    inputs = tokenizer(prompt, return_tensors="pt").to(llm.device)
    print(f"Input tokens: {inputs['input_ids'].shape[1]}")
    outputs = llm.generate(**inputs, max_new_tokens=250, temperature=0.7, top_p=0.9)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.split("Answer:")[-1].strip()
