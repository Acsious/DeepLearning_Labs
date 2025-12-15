from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
import torch
import datetime

model_id = "Qwen/Qwen3-4B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    dtype=torch.float16,
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    temperature=0.3,
    top_p=0.95,
    repetition_penalty=1.2
)

llm = HuggingFacePipeline(pipeline=pipe)
chat_model = ChatHuggingFace(llm=llm)

@tool
def get_current_date() -> str:
    """Возвращает текущую дату в формате ГГГГ-ММ-ДД."""
    return datetime.datetime.now().strftime("%Y-%m-%d")

tools = [get_current_date]
system_prompt = (
    f"Четко следуй инструкциям. "
    f"Если пользователь спрашивает о текущей дате, сегодняшнем дне или любой информации, связанной с датой — "
    f"всегда используй инструмент {tools[0]} для получения точной даты. "
    f"Обязательно вызывай {tools[0]} перед ответом. "
    f"Никогда не придумывай дату самостоятельно. "
    f"В финальном ответе явно укажи результат, полученный от инструмента {tools[0]}."
)

agent_executor = create_react_agent(
    chat_model,
    tools,
    prompt=system_prompt
)

queries = [
    "Какая сегодня дата?",
    "Что за число сегодня?"
]

for query in queries:
    print(f"\n=== Запрос: {query} ===")
    input_message = {"role": "user", "content": query}
    result = agent_executor.invoke({"messages": [input_message]})
    for message in result["messages"]:
        message.pretty_print()