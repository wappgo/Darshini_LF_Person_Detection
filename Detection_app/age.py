import streamlit as st
from deepface import DeepFace
import cv2
import numpy as np
from PIL import Image
import base64
import gc  # Import the garbage collector module


import os
import tempfile
from pathlib import Path


def resize_for_analysis(image_np, max_size=480):
    """
    Aggressively resizes the image before analysis to save memory.
    A max size of 320px is a good balance between speed and accuracy.
    """
    h, w = image_np.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        # INTER_AREA is best for shrinking images
        image_np = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image_np


# --- HELPER FUNCTION (FINAL, BALANCED VERSION) ---
def analyze_image_safely(image_to_process):
    """
    Analyzes image step-by-step for memory safety, but uses a more
    accurate detector ('mtcnn') to improve results like gender detection.
    """
    try:
        # Step 0: Resize the image to a good-quality, manageable size.
        image_to_process = resize_for_analysis(image_to_process) # This should use max_size=640
        gc.collect()

        # --- Step 1: Analyze for Age and get face region using the best detector ---
        age_result = DeepFace.analyze(
            img_path=image_to_process,
            actions=['age'],
            detector_backend='mtcnn',  # Using the more accurate detector
            enforce_detection=True,
            silent=True
        )
        first_face = age_result[0]
        age = first_face['age']
        region = first_face['region']
        gc.collect() # Unload the age model

        # --- Step 2: Analyze for Gender (safely) ---
        gender = "N/A"
        try:
            gender_result = DeepFace.analyze(
                img_path=image_to_process,
                actions=['gender'],
                detector_backend='mtcnn',  # Using the more accurate detector
                enforce_detection=False,
                silent=True
            )
            gender = gender_result[0]['dominant_gender']
        except Exception:
            st.warning("Could not determine gender. Continuing...")
        finally:
            gc.collect() # Unload the gender model

        # --- Step 3: Analyze for Emotion (safely) ---
        emotion = "N/A"
        try:
            emotion_result = DeepFace.analyze(
                img_path=image_to_process,
                actions=['emotion'],
                detector_backend='mtcnn',  # Using the more accurate detector
                enforce_detection=False,
                silent=True
            )
            emotion = emotion_result[0]['dominant_emotion']
        except Exception:
            st.warning("Could not determine emotion. Continuing...")
        finally:
            gc.collect() # Unload the emotion model

        # --- Drawing and Display ---
        margin = 5
        age_range_low = max(0, age - margin)
        age_range_high = age + margin

        x, y, w, h = region['x'], region['y'], region['w'], region['h']
        cv2.rectangle(image_to_process, (x, y), (x + w, y + h), (36, 255, 12), 2)

        info_text = f"Est. Age: {age_range_low}-{age_range_high}"
        gender_emotion_text = f"{gender.capitalize()}, {emotion.capitalize()}"

        text_bg_y = y - 30
        cv2.rectangle(image_to_process, (x, text_bg_y), (x + w, y), (36, 255, 12), -1)
        cv2.putText(image_to_process, info_text, (x + 5, y - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(image_to_process, gender_emotion_text, (x + 5, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        st.success("Analysis Complete!")
        st.markdown(f"""
        - **Model's Direct Prediction:** `{age}`
        - **Estimated Age Range:** `{age_range_low} - {age_range_high}` years old
        - **Predicted Gender:** `{gender.capitalize()}`
        - **Dominant Emotion:** `{emotion.capitalize()}`
        """)

        return image_to_process

    except ValueError:
        st.error("Could not detect a face in the image. Please try another one.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        if 'allocate' in str(e).lower() or 'oom' in str(e).lower():
            st.warning("This looks like a memory error. Try a smaller image or restart the app.")
        return None


# --- NEW HELPER FUNCTION FOR FINDING PERSON ---
@st.cache_data(show_spinner=False)
def find_person_in_database(uploaded_image_path, database_root_folder):
    """
    Searches for a person in a nested folder structure using face verification.
    Uses a more robust model and detector for better accuracy.
    """
    image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.jfif"]
    all_db_images = []
    for ext in image_extensions:
        all_db_images.extend(Path(database_root_folder).rglob(ext))

    # st.write(f"Searching through {len(all_db_images)} images in the database...") # UI feedback
    st.write(f"Searched on live camera feed...")

    for db_image_path in all_db_images:
        try:
            # --- CHANGES ARE HERE ---
            result = DeepFace.verify(
                img1_path=uploaded_image_path,
                img2_path=str(db_image_path),
                model_name="Facenet512",  # CHANGED: Using a more robust model
                detector_backend="mtcnn",  # CHANGED: Using a better face detector
                enforce_detection=True,
                silent=True
            )
            # --- END OF CHANGES ---

            if result['verified']:
                # --- PARSE THE FILE PATH FOR INFORMATION ---
                time_str = db_image_path.stem.replace("-", ":")
                pole_folder = db_image_path.parent.name
                camera_folder = db_image_path.parent.parent.name

                pole_num = pole_folder.split('_')[-1]
                camera_num = camera_folder.split('_')[-1]

                return f"Last saw near Poll no. {pole_num} in camera feed {camera_num} at {time_str}"

        except (ValueError, AttributeError):
            # MODIFIED: Added a print statement for debugging in your terminal
            print(f"DEBUG: Could not find a clear face in {db_image_path}. Skipping.")
            continue
        except Exception as e:
            # MODIFIED: Added a print statement for debugging
            print(f"DEBUG: An unexpected error occurred with {db_image_path}: {e}")
            continue

    return None # Return None if no match is found

@st.cache_data
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        st.error(f"Logo file not found: {bin_file}. Please ensure it's in the correct folder.")
        return None


def run_app():

    # --- PAGE CONFIGURATION ---
    st.set_page_config(
        page_title="Age Detection App",
        page_icon="📸",
        layout="wide",
    )

    logo_base64 = get_base64_of_bin_file("Detection_app/Darshini_logo.png")

    if logo_base64:
        st.markdown(f"""
            <div style="display: flex; align-items: center;">
                <img src="data:image/png;base64,{logo_base64}" width="70" style="position: relative; top: 5px; margin-right: 15px;">
                <h1>Person Detection & Face Analyzer</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.info(
        "**Disclaimer:** This AI predicts *apparent age* based on visual features, not biological age. Results can vary."
    )

    st.write("Upload an image to let the AI analyze the person's face for estimated age, gender, and emotion.")

    SCRIPT_DIR = Path(__file__).parent
    DATABASE_ROOT = SCRIPT_DIR

    col1, col2 = st.columns(2)

    with col1:
        st.header("Upload Your Image")
        uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png", "jfif"], label_visibility="collapsed")

        if uploaded_file is not None:
            pil_image = Image.open(uploaded_file).convert('RGB')
            original_image_np = np.array(pil_image)
            st.image(original_image_np, caption="Your Uploaded Image", use_container_width=True)

    with col2:
        st.header("Analysis Result")
        if uploaded_file is not None:
            if st.button("Analyze Image", type="primary"):
                with st.spinner("🧠 Analyzing, this may take a moment..."):
                    image_to_analyze = np.array(Image.open(uploaded_file).convert('RGB'))
                    image_bgr = cv2.cvtColor(image_to_analyze, cv2.COLOR_RGB2BGR)
                    
                    # --- MODIFIED: Store the processed image in session state ---
                    st.session_state.processed_image = analyze_image_safely(image_bgr)

            # --- NEW: Display the processed image if it exists in session state ---
            if 'processed_image' in st.session_state and st.session_state.processed_image is not None:
                processed_image = st.session_state.processed_image
                result_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                st.image(result_image_rgb, caption="Processed Image", use_container_width=True)

                st.divider() # Add a visual separator

                # --- NEW: Button to trigger the database search ---
                if st.button("📍 Find Last Seen Location"):
                    with st.spinner("Searching camera feed database... This might take some time."):
                        # To use DeepFace.verify, we need a file path. So we save the uploaded file temporarily.
                        tmp_file_path = None
                        try:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                                tmp.write(uploaded_file.getvalue())
                                tmp_file_path = tmp.name
                            
                            # Call the new search function
                            result_message = find_person_in_database(tmp_file_path, DATABASE_ROOT)

                            if result_message:
                                st.success(f"**Match Found!** {result_message}")
                            else:
                                st.warning("Person not found in the camera feed database.")
                        finally:
                            # Clean up the temporary file
                            if tmp_file_path and os.path.exists(tmp_file_path):
                                os.remove(tmp_file_path)
        else:
            st.info("Upload an image and click 'Analyze Image' to see the results.")

    # --- NEW: Clear session state on file change to reset the analysis ---
    if 'current_file' not in st.session_state:
        st.session_state.current_file = None

    if uploaded_file and uploaded_file.name != st.session_state.current_file:
        if 'processed_image' in st.session_state:
            del st.session_state.processed_image
        st.session_state.current_file = uploaded_file.name

















































































































































# import streamlit as st
# import cv2
# import numpy as np
# from PIL import Image
# import base64
# import gc
# import os

# # --- PAGE CONFIGURATION ---
# st.set_page_config(
#     page_title="Age Detection App",
#     page_icon="📸",
#     layout="wide",
# )

# # Set environment variables to limit memory usage
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TensorFlow warnings
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU usage
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Limit GPU memory growth
# os.environ['OMP_NUM_THREADS'] = '1'  # Limit CPU threads

# def ultra_resize_image(image_np, max_size=200):  # Much smaller - 200px max
#     """Aggressively resizes image to minimize memory usage."""
#     h, w = image_np.shape[:2]
#     if max(h, w) > max_size:
#         scale = max_size / max(h, w)
#         new_w, new_h = int(w * scale), int(h * scale)
#         # Use INTER_AREA for best quality when shrinking
#         image_np = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
#     return image_np

# def preprocess_image_aggressively(image_np):
#     """Ultra-aggressive preprocessing to minimize memory footprint"""
#     # Convert to uint8 if not already
#     if image_np.dtype != np.uint8:
#         image_np = (image_np * 255).astype(np.uint8)
    
#     # Make image very small to reduce memory usage
#     image_np = ultra_resize_image(image_np, max_size=150)  # Even smaller
    
#     # Force garbage collection
#     gc.collect()
    
#     return image_np

# def analyze_image_step_by_step(image_to_process):
#     """
#     Analyzes image one step at a time to minimize memory usage.
#     Uses lazy loading and immediate cleanup.
#     """
#     try:
#         # Clear memory before starting
#         gc.collect()
        
#         # Preprocess image aggressively
#         image_to_process = preprocess_image_aggressively(image_to_process)
        
#         # Import DeepFace only when needed (lazy loading)
#         try:
#             from deepface import DeepFace
#         except ImportError:
#             st.error("DeepFace not installed. Please run: pip install deepface")
#             return None
        
#         # Step 1: Try age detection only first (most important)
#         st.info("🔍 Step 1: Detecting age...")
#         age_result = DeepFace.analyze(
#             img_path=image_to_process,
#             actions=['age'],  # Only age first
#             detector_backend='opencv',
#             enforce_detection=False,
#             silent=True
#         )
        
#         if not age_result:
#             st.error("No face detected in the image.")
#             return None
        
#         # Handle result format
#         face_data = age_result[0] if isinstance(age_result, list) else age_result
#         age = int(face_data['age'])
#         region = face_data['region']
        
#         # Clear memory after age detection
#         gc.collect()
        
#         # Step 2: Try gender detection
#         gender = "Unknown"
#         try:
#             st.info("🔍 Step 2: Detecting gender...")
#             gender_result = DeepFace.analyze(
#                 img_path=image_to_process,
#                 actions=['gender'],
#                 detector_backend='opencv',
#                 enforce_detection=False,
#                 silent=True
#             )
#             gender_face = gender_result[0] if isinstance(gender_result, list) else gender_result
#             gender = gender_face['dominant_gender']
#             gc.collect()
#         except:
#             st.warning("Could not detect gender - continuing with age only")
        
#         # Step 3: Try emotion detection
#         emotion = "Unknown"
#         try:
#             st.info("🔍 Step 3: Detecting emotion...")
#             emotion_result = DeepFace.analyze(
#                 img_path=image_to_process,
#                 actions=['emotion'],
#                 detector_backend='opencv',
#                 enforce_detection=False,
#                 silent=True
#             )
#             emotion_face = emotion_result[0] if isinstance(emotion_result, list) else emotion_result
#             emotion = emotion_face['dominant_emotion']
#             gc.collect()
#         except:
#             st.warning("Could not detect emotion - continuing with available data")
        
#         # Draw results on image
#         x, y, w, h = region['x'], region['y'], region['w'], region['h']
        
#         # Ensure bounding box is within image bounds
#         img_height, img_width = image_to_process.shape[:2]
#         x = max(0, min(x, img_width - 1))
#         y = max(0, min(y, img_height - 1))
#         w = min(w, img_width - x)
#         h = min(h, img_height - y)
        
#         # Draw bounding box
#         cv2.rectangle(image_to_process, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
#         # Add age text
#         age_range_low = max(0, age - 5)
#         age_range_high = age + 5
#         info_text = f"Age: {age_range_low}-{age_range_high}"
        
#         # Simple text placement
#         text_y = max(20, y - 10)
#         cv2.putText(image_to_process, info_text, (x, text_y),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
#         # Show results
#         st.success("✅ Analysis Complete!")
        
#         results_text = f"""
#         ### Results:
#         - **Estimated Age:** {age} years (range: {age_range_low}-{age_range_high})
#         - **Gender:** {gender.capitalize() if gender != "Unknown" else "Could not detect"}
#         - **Emotion:** {emotion.capitalize() if emotion != "Unknown" else "Could not detect"}
#         """
        
#         st.markdown(results_text)
        
#         # Final memory cleanup
#         gc.collect()
        
#         return image_to_process
        
#     except Exception as e:
#         error_msg = str(e)
#         st.error(f"❌ Analysis failed: {error_msg[:200]}...")
        
#         if any(keyword in error_msg.lower() for keyword in ['oom', 'memory', 'allocation']):
#             st.error("🚨 **Memory Issue Detected!**")
#             st.markdown("""
#             **Try these solutions:**
#             1. 🔄 **Restart the app** (press Ctrl+C and run again)
#             2. 📏 **Use a much smaller image** (under 500KB)
#             3. 🖼️ **Reduce image resolution** before uploading
#             4. 💾 **Close other applications** to free memory
#             5. 🔧 **Try the minimal version** (use only age detection)
#             """)
        
#         # Always clean up memory on error
#         gc.collect()
#         return None

# # --- STREAMLIT UI LAYOUT ---

# def get_base64_of_bin_file(bin_file):
#     """Function to encode image to base64"""
#     try:
#         with open(bin_file, 'rb') as f:
#             data = f.read()
#         return base64.b64encode(data).decode()
#     except FileNotFoundError:
#         return None

# # Header with Logo
# logo_base64 = get_base64_of_bin_file("Darshini_logo.png")

# if logo_base64:
#     st.markdown(f"""
#         <div style="display: flex; align-items: center;">
#             <img src="data:image/png;base64,{logo_base64}" width="70" style="margin-right: 15px;">
#             <h1>Person Detection & Face Analyzer</h1>
#         </div>
#         """,
#         unsafe_allow_html=True
#     )
# else:
#     st.title("Person Detection & Face Analyzer")

# # Disclaimers
# st.info(
#     "**Disclaimer:** This AI predicts the *apparent age* based on visual features, not biological age. "
#     "Results vary based on lighting, expression, and image quality."
# )

# st.warning(
#     "⚡ **Memory Optimized Version** - Uses step-by-step analysis for low-memory systems. "
#     "Upload small, clear images (under 500KB) for best results."
# )

# # Two columns layout
# col1, col2 = st.columns(2)

# with col1:
#     st.header("📤 Upload Your Image")
#     uploaded_file = st.file_uploader(
#         "Choose an image file...", 
#         type=["jpg", "jpeg", "png"], 
#         label_visibility="collapsed",
#         help="For best performance: images under 500KB with clear faces"
#     )

#     if uploaded_file is not None:
#         file_size = len(uploaded_file.getvalue())
        
#         # More restrictive file size check
#         if file_size > 1024 * 1024:  # 1MB limit (reduced from 2MB)
#             st.error("🚫 File too large! Please upload an image smaller than 1MB.")
#             st.info("💡 **Tip:** Use online tools to compress your image before uploading.")
#         else:
#             try:
#                 # Load and display image
#                 pil_image = Image.open(uploaded_file).convert('RGB')
#                 original_image_np = np.array(pil_image)
                
#                 # Show file info
#                 st.success(f"✅ File loaded: {file_size / 1024:.1f} KB")
                
#                 # Display with fixed parameter name
#                 st.image(original_image_np, caption="Your Uploaded Image", use_container_width=True)
                
#             except Exception as e:
#                 st.error(f"Error loading image: {e}")
#                 uploaded_file = None

# with col2:
#     st.header("🔬 Analysis Result")
    
#     if uploaded_file is not None:
#         if st.button("🚀 Analyze Image", type="primary", help="Start step-by-step analysis"):
#             with st.spinner("🤖 Analyzing step by step..."):
#                 try:
#                     # Load image for processing
#                     pil_image = Image.open(uploaded_file).convert('RGB')
#                     image_array = np.array(pil_image)
                    
#                     # Convert to BGR for OpenCV
#                     cv_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                    
#                     # Analyze with step-by-step approach
#                     result_image = analyze_image_step_by_step(cv_image)
                    
#                     # Display result
#                     if result_image is not None:
#                         # Convert back to RGB for display
#                         result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
#                         st.image(result_rgb, caption="🎯 Analysis Result", use_container_width=True)
                        
#                         # Offer to save results
#                         st.balloons()  # Celebration effect
                        
#                 except Exception as e:
#                     st.error(f"Processing error: {e}")
#                     st.info("Please try a different image or restart the app.")
#     else:
#         st.info("👆 Upload an image above and click 'Analyze Image' to start!")

# # Footer with memory management
# st.divider()

# col_mem1, col_mem2, col_mem3 = st.columns(3)

# with col_mem1:
#     if st.button("🧹 Clear Memory", help="Free up system memory"):
#         gc.collect()
#         st.success("Memory cleared!")

# with col_mem2:
#     if st.button("ℹ️ System Info", help="Show memory usage tips"):
#         st.info("""
#         **Memory Tips:**
#         - Use images < 500KB
#         - Close unused browser tabs
#         - Restart app if it becomes slow
#         - Try one image at a time
#         """)

# with col_mem3:
#     if st.button("🔄 Reset App", help="Clear all data and start fresh"):
#         st.rerun()

# # Performance stats
# st.caption("💡 **Pro Tips:** Use well-lit photos with clear faces. Avoid group photos for best accuracy.")