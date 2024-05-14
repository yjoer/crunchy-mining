from dotenv import load_dotenv
from st_pages import Page
from st_pages import show_pages

load_dotenv()

show_pages(
    [
        Page("crunchy_mining/pages/visualization.py", "Visualization"),
        Page("crunchy_mining/pages/evaluation.py", "Model Evaluation"),
        Page("crunchy_mining/pages/interpretation.py", "Model Interpretation"),
    ]
)
