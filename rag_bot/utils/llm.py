from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "microsoft/phi-2" 
#microsoft/phi-1_5
#TinyLlama/TinyLlama-1.1B-Chat-v1.0

tokenizer = AutoTokenizer.from_pretrained(model_id)

llm = AutoModelForCausalLM.from_pretrained(
    model_id,
    do_sample=False,
    torch_dtype=torch.float32,
  
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
