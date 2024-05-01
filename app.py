from st_pages import Page
from st_pages import show_pages

show_pages(
    [
        Page("pages/visualization.py", "Visualization"),
        Page("pages/interpretation.py", "Model Interpretation"),
    ]
)
