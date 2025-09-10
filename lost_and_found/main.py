# app.py
import streamlit as st
import psycopg2
import os
from dotenv import load_dotenv
from urllib.parse import quote_plus
import base64
# --- 1. Configuration and Setup (No changes) ---

load_dotenv()

try:
    db_user = os.getenv('DB_USER')
    db_password_raw = os.getenv('DB_PASSWORD')
    db_host = os.getenv('DB_HOST')
    db_port = os.getenv('DB_PORT')
    db_name = os.getenv('DB_NAME')
    db_password_encoded = quote_plus(db_password_raw)
    DB_CONNECTION_STRING = (
        f"postgresql://{db_user}:{db_password_encoded}"
        f"@{db_host}:{db_port}/{db_name}"
    )
except (TypeError, AttributeError):
    st.error("One or more database environment variables are not set. Please check your .env file.")
    st.stop()


# --- 2. Database Functions (THIS IS THE UPDATED PART) ---

def get_db_connection():
    try:
        return psycopg2.connect(DB_CONNECTION_STRING)
    except psycopg2.OperationalError as e:
        st.error(f"Database connection failed. Check credentials and ensure the DB is running. Error: {e}")
        return None

def find_matching_items(product_type, brand, color):
    """
    Searches the database using a scoring system.
    A row gets +1 point for each field that matches.
    Returns rows with a score of 2 or more.
    """
    conn = get_db_connection()
    if conn is None:
        return []

    try:
        with conn.cursor() as cur:
            # This dictionary maps the user input to the database column names
            fields = {
                "product_type": product_type,
                "brand": brand,
                "color": color
            }
            
            score_parts = []
            params = []

            # Dynamically build the scoring logic based on which fields the user filled out
            for column, value in fields.items():
                if value:  # Only add to score if the user provided input for this field
                    # The CASE statement is like an if/else in SQL. It returns 1 if it matches, 0 if not.
                    score_parts.append(f'(CASE WHEN "{column}" ILIKE %s THEN 1 ELSE 0 END)')
                    params.append(f"%{value}%")
            
            # If the user provided fewer than 2 inputs, we can't meet the "at least two" rule.
            if len(params) < 2:
                # We return a special string to inform the UI to show a message.
                return "NOT_ENOUGH_INPUTS"
            
            # Combine all the CASE statements with a '+' to get the total match_score
            score_calculation = " + ".join(score_parts)

            # The final query selects all data PLUS our calculated score
            # It then filters to keep only rows with a score >= 2
            # Finally, it sorts the results so the best matches (highest score) appear first
            final_query = f"""
                SELECT 
                    "ID", name, product_type, brand, model, color, image_url,
                    ({score_calculation}) AS match_score
                FROM 
                    scrapingdata."Darshini"
                WHERE 
                    ({score_calculation}) >= 2
                ORDER BY 
                    match_score DESC, "ID";
            """
            
            # The parameters need to be duplicated because they are used twice in the query
            cur.execute(final_query, tuple(params + params))
            
            columns = [desc[0] for desc in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]
    finally:
        if conn:
            conn.close()


@st.cache_data
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        st.error(f"Logo file not found: {bin_file}. Please ensure it's in the correct folder.")
        return None

# --- 3. Streamlit User Interface (No changes) ---

def run_app():

    logo_path = "lost_and_found/logo.png"
    logo_base64 = get_base64_of_bin_file(logo_path)

    # 2. THIS IS THE MISSING PART: You need to actually display the logo and title.
    if logo_base64:
        st.markdown(f"""
            <div style="display: flex; align-items: center;">
                <img src="data:image/png;base64,{logo_base64}" width="70" style="position: relative; top: 5px; margin-right: 15px;">
                <h1>Find Your Lost Item</h1>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        # This is a fallback in case the logo can't be found
        st.title("🔎 Find Your Lost Item")
    st.markdown("Enter at least two of the following fields to find matching items.")

    with st.form("search_form"):
        st.subheader("Enter Item Details")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            product_type = st.text_input("Product Type", help="E.g., 'Mobile', 'Laptop', 'Watch'")
        with col2:
            brand = st.text_input("Brand", help="E.g., 'Apple', 'Samsung', 'Dell'")
        with col3:
            color = st.text_input("Color", help="E.g., 'Black', 'Silver', 'Blue'")
        
        submitted = st.form_submit_button("Search for Item", use_container_width=True)

    # --- 4. Search and Display Logic (Slightly updated to handle the warning) ---

    if submitted:
        with st.spinner("Searching the database..."):
            results = find_matching_items(product_type, brand, color)
            st.session_state.results = results
            st.session_state.searched = True

    if st.session_state.get('searched', False):
        st.divider()
        results = st.session_state.get('results', [])
        
        # Check for our special string to show a specific warning
        if results == "NOT_ENOUGH_INPUTS":
            st.warning("⚠️ Please fill in at least two fields to start a search.")
        elif not results:
            st.info("No items found that match at least two of your search criteria.")
        else:
            st.success(f"Found {len(results)} matching items! The best matches are shown first.")
            cols = st.columns(3)
            for i, item in enumerate(results):
                with cols[i % 3]:
                    with st.container(border=True):
                        st.image(item['image_url'], use_container_width=True)
                        st.subheader(f"{item['brand']} {item['model']}")
                        st.write(f"**Match Score:** {item['match_score']} / {len(results[0]['match_score'].__class__.__mro__[1].__dict__)-1 if results else 0}") # A bit complex, simplified below
                        st.write(f"**Type:** {item['product_type']}")
                        st.write(f"**Color:** {item['color']}")
                        st.write(f"**Reported by:** {item['name']}")
                        if st.button("This is Mine!", key=f"claim_{item['ID']}", use_container_width=True):
                            st.balloons()
                            st.success(f"Claim initiated for Item ID #{item['ID']}!")