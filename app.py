from st_pages import Page
from st_pages import show_pages

show_pages(
    [
        Page("crunchy_mining/pages/visualization.py", "Visualization"),
        Page("crunchy_mining/pages/interpretation.py", "Model Interpretation"),
    ]
)
