import json
import yaml
import time
from datetime import datetime
from langchain_core.messages import AIMessage, HumanMessage

def load_config(file_path = "config.yml"):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)
    
config = load_config()

def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Function '{func.__name__}' executed in {execution_time:.4f} seconds")
        return result
    return wrapper


def save_chat_history_json(chat_history,file_path):
    with open(file_path,"w") as f:
        json_data=[message.dict() for message in chat_history]
        json.dump(json_data,f)

def load_chat_history_json(file_path):
    with open(file_path,"r") as f:
        json_data=json.load(f)
        messages=[HumanMessage(**message) if message["type"]=="human" else AIMessage(**message) for message in json_data]
        return messages
def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H_%M_%S") 