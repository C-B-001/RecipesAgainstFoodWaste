from smolagents import CodeAgent, HfApiModel,load_tool,tool
import datetime
import requests
import pytz
import yaml

from tools.final_answer import FinalAnswerTool
from tools.recipe_search import search_recipe
from tools.storage_search import search_storage

from Gradio_UI import GradioUI


final_answer = FinalAnswerTool()


model = HfApiModel(
max_tokens=2096,
model_id='Qwen/Qwen2.5-Coder-32B-Instruct',
custom_role_conversions=None,
)


with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)

    
agent = CodeAgent(
    model=model,
    tools=[final_answer, search_recipe, search_storage], ## add your tools here (don't remove final answer)
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates
)


GradioUI(agent).launch()