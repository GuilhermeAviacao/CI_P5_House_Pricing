import streamlit as st

#Summary
def page1_body():
    """ Page 1 contents"""
    st.write("Quick Project Summary")

    #Text based on Readme file

    st.info("""
    **Situation:** \n
             
    Lydia, a Belgian customer, has inherited four houses in the United States, in Ames-Iowa.\n
    While she understands her local real state market, she needs help appraising her assets in America.\n
    The task is to assess what are the drivers of price in the housing market so she can maximize the sale value.\n
    Customer provided a public dataset to feed our models\n
    
    **Objectives:**\n
    1. Understand how house attributes correlate with sale prices
    2. Predict accurate sale prices for her four inherited properties
    3. Simulate prices for any house in Ames, Iowa"
""")


#hypothesis    
