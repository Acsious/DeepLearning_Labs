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
    """Возвращает текущую дату в формате ГГГГ-ММ-ДД.
    Используйте этот инструмент, когда пользователь спрашивает о сегодняшней дате или текущем дне."""
    return datetime.datetime.now().strftime("%Y-%m-%d")

tools = [get_current_date]

system_prompt = (
    "Ты — полезный ассистент. "
    "Если вопрос пользователя касается текущей даты, сегодняшнего дня или необходимости узнать дату — "
    "обязательно используй инструмент get_current_date для получения точной информации. "
    "Никогда не придумывай дату самостоятельно. "
    "После вызова инструмента сообщи пользователю полученную дату четко и кратко."
)

agent_executor = create_react_agent(
    chat_model,
    tools,
    prompt=system_prompt
)

input_message = {
    "role": "user",
    "content": "Какая сегодня дата?"
}

result = agent_executor.invoke({"messages": [input_message]})

print("=== Ответ агента ===")
for message in result["messages"]:
    message.pretty_print()

