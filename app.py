import streamlit as st
import sys
import os
import base64
# -- Page Configuration --
st.set_page_config(
    page_title="Main Hub",
    page_icon="🚀",
    layout="wide"
)

# --- Path Setup ---
# This is the magic that allows us to import python files from other folders
# We get the path of the current directory (where app.py is)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Add the paths to your two projects to Python's search path
sys.path.append(os.path.join(ROOT_DIR, 'Detection_app'))
sys.path.append(os.path.join(ROOT_DIR, 'lost_and_found'))

# --- App Selection Logic ---
# Import your project files AS MODULES
try:
    import age as age_detection_app
    import main as lost_and_found_app
except ImportError as e:
    st.error(f"Failed to import a project module. Please check folder names and file paths. Error: {e}")
    st.stop()



@st.cache_data
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        # Show the error in the sidebar since that's where we're trying to display it
        st.sidebar.error(f"Logo file not found: {bin_file}")
        return None

logo_path = "lost_and_found\logo.png"  # Make sure this path is correct
logo_base64 = get_base64_of_bin_file(logo_path)

if logo_base64:
    st.sidebar.markdown(f"""
        <div style="text-align: center; padding-bottom: 20px;">
            <img src="data:image/png;base64,{logo_base64}" width="150">
        </div>
        """,
        unsafe_allow_html=True
    )


# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
app_choice = st.sidebar.radio(
    "Choose your application",
    ('Age Detection', 'Lost and Found')
)

# --- Page Display Logic ---
# if app_choice == 'Home':
#     st.title("Welcome to the Main Application Hub")
#     st.markdown("Please select an application from the sidebar to begin.")
#     # You can add more details or images here

if app_choice == 'Age Detection':
    # We will create this 'run_app' function in the next step
    age_detection_app.run_app()

elif app_choice == 'Lost and Found':
    # We will also create this 'run_app' function in the next step
    lost_and_found_app.run_app()