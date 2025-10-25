import streamlit as st



from app_pages.multi_page import MultiPage

from app_pages.page1 import page1_body
from app_pages.page2 import page2_body
from app_pages.page3 import page3_body
from app_pages.page4 import page4_body
from app_pages.page5 import page5_body


# Create an instance
app = MultiPage(app_name="Housing Price Analysis")

# Add your app pages here using .add_page()
app.app_page("Page 1", page1_body)
app.app_page("Page 2", page2_body)
app.app_page("Page 3", page3_body)
app.app_page("Page 4", page4_body)
app.app_page("Page 5", page5_body)

app.run()