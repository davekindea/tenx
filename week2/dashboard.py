import streamlit as st
from pages import User_Overview,User_Engagement, Experience_Analytics, Satisfaction_Analysi
# Dictionary for page routing
PAGES = {
    "User Overview Analysis": User_Overview,
    "User_Engagement":User_Engagement,
    "Experience_Analytics":Experience_Analytics,
    "Satisfaction_Analysi":Satisfaction_Analysi
   
}

def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))
    
    # Display the selected page
    page = PAGES[selection]
    page.app()

if __name__ == "__main__":
    main()
