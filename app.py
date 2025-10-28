import streamlit as st

from app_pages.multi_page import MultiPage
from app_pages.page1 import page1_body
from app_pages.page2 import page2_body
from app_pages.page3 import page3_body
from app_pages.page4 import page4_body
from app_pages.page5 import page5_body
from app_pages.page6 import page6_body

# Create an instance
app = MultiPage(app_name="Housing Price Analysis")

# Add your app pages here using .add_page()
app.app_page("Page 1: Project Summary", page1_body)
app.app_page("Page 2: Data Exploration Study", page2_body)
app.app_page("Page 3: ML Model Description", page3_body)
app.app_page("Page 4: Inherited Houses Appraisal", page4_body)
app.app_page("Page 5: Generalized Predictor", page5_body)
app.app_page("Page 6: Hypothesis & Conclusions", page6_body)

app.run()
