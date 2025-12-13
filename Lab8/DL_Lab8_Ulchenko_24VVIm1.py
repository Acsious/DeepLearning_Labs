from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "Qwen/Qwen2.5-7B-Instruct-1M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

file_path = 'ENG_article.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    data = file.read().replace('\n', ' ').strip()

print(f"Длина текста: {len(data)} символов")
print(f"Пример текста: {data[:200]}...")

def generate_response(messages, max_new_tokens=512):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1
        )
    generated_ids_ = generated_ids[0][len(model_inputs.input_ids[0]):]
    response = tokenizer.decode(generated_ids_, skip_special_tokens=True)
    return response.strip()

system_prompt = {
    "role": "system",
    "content": "Ты полезный ассистент. Проанализируй предоставленный текст и ответь на вопрос точно, ссылаясь на содержание документа. Если информация отсутствует, укажи это явно. Ответь на русском языке, кратко и структурировано."
}

questions = [
    "В каком году была обозначена проблема взрывающихся градиентов?",
    "Кто в 1891 году разработал метод уничтожающей производной?",
    "Кто предложил цепное правило дифференцирования и в каком году?"
]

results = {}

for i, question in enumerate(questions, 1):
    messages = [
        system_prompt,
        {
            "role": "user",
            "content": f"Текст для анализа:\n{data}\n\nВопрос: {question}"
        }
    ]

    response = generate_response(messages)

    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    results[f"Вопрос {i}"] = {
        "prompt": prompt_text,
        "response": response
    }

    print(f"\n--- Вопрос {i}: {question} ---")
    print(f"Промпт: {prompt_text[:200]}...")
    print(f"Ответ: {response}")

with open('prompts_and_responses.txt', 'w', encoding='utf-8') as f:
    for key, value in results.items():
        f.write(f"{key}\n")
        f.write("Промпт:\n")
        f.write(value['prompt'] + "\n")
        f.write("Ответ:\n")
        f.write(value['response'] + "\n\n")

print("\nПромпты и ответы сохранены в 'prompts_and_responses.txt'")