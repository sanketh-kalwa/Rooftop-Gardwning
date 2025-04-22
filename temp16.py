import streamlit as st
import time
import google.generativeai as genai
from datetime import datetime, timedelta
from io import BytesIO
from urllib.request import urlopen
import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

# Set page title and layout
st.set_page_config(page_title="RoofTop Gardening", layout="wide")

# Apply custom CSS for background image (without white overlay)
page_bg_img = f"""
<style>
    body {{
        background-image: url("https://sl.bing.net/df80MIH7xYq");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    .stApp {{
        padding: 20px;
        border-radius: 10px;
    }}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Login Functionality
def login(username, password):
    valid_users = ["sanketh", "nikhil", "karthik", "shiva"]
    if username in valid_users and password == "rooftop":
        st.session_state.logged_in = True
        st.session_state.username = username
        st.session_state.water_start_time = datetime.now()
        st.session_state.fertilizer_start_time = datetime.now()
        return True
    return False

# Initialize session state for login and timers
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "water_start_time" not in st.session_state:
    st.session_state.water_start_time = None
if "fertilizer_start_time" not in st.session_state:
    st.session_state.fertilizer_start_time = None
if "forum_data" not in st.session_state:
    st.session_state.forum_data = []
if "replying" not in st.session_state:
    st.session_state.replying = {}

# Function to calculate remaining time and progress
def calculate_progress(start_time, total_duration):
    if start_time is None:
        return 0, "Login Required"
    elapsed_time = datetime.now() - start_time
    remaining_time = total_duration - elapsed_time.total_seconds()
    if remaining_time <= 0:
        return 100, "Time to water/fertilize!"
    progress = (elapsed_time.total_seconds() / total_duration) * 100
    return min(progress, 100), f"Time left: {timedelta(seconds=int(remaining_time))}"

# Layout for Reminders and Login Form
col1, col2 = st.columns([3, 1])  # Adjust column widths for layout

# Reminders in the Left Column
with col1:
    st.write("🌿 Reminders")
    water_progress, water_message = calculate_progress(st.session_state.water_start_time, 24 * 3600)  # 24 hours
    fertilizer_progress, fertilizer_message = calculate_progress(st.session_state.fertilizer_start_time, 48 * 3600)  # 48 hours

    # Placeholders for dynamic updates
    water_placeholder = st.empty()
    fertilizer_placeholder = st.empty()

    # Update progress bars and messages
    water_placeholder.progress(water_progress / 100)
    water_placeholder.write(f"💧 Water Reminder: {water_message}")
    fertilizer_placeholder.progress(fertilizer_progress / 100)
    fertilizer_placeholder.write(f"🌱 Fertilizer Reminder: {fertilizer_message}")

# Login Form in the Right Column
with col2:
    if st.session_state.logged_in:
        st.success(f"Welcome, {st.session_state.username}!")
    else:
        with st.expander("🔑 Login"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", placeholder="Enter your password", type="password")
            login_button = st.button("Login")

            if login_button:
                if login(username, password):
                    st.success(f"Welcome, {username}!")
                    st.rerun()  # Rerun the app to update the UI
                else:
                    st.error("Login unsuccessful. Please check your credentials.")

# Sidebar Navigation
st.sidebar.title("🌿 Navigation")
page = st.sidebar.radio("Go to", ["Home", "Chatbot", "Prompts", "Forum"])

# Home Page
if page == "Home":
    st.title("🌿 Welcome to Our RoofTop Gardening Web App!")
    
    st.markdown("""
    RoofTop gardening transforms underutilized rooftop spaces into thriving green areas. 
    This web app serves as your **go-to guide** for starting and maintaining a **cost-effective, sustainable** garden right on your terrace. 

    With easy-to-follow tips and expert recommendations, you can enjoy **fresh, organic produce** while contributing to a greener environment.
    """)

    st.header("🌱 Why RoofTop Gardening?")
    st.markdown("""
    - **Utilize Your Space:** Convert rooftops into lush gardens.
    - **Grow Fresh & Organic:** Enjoy pesticide-free, home-grown produce.
    - **Cost-Effective Solutions:** Gardening tips that don’t break the bank.
    - **Health & Well-being:** Gardening reduces stress and promotes a healthier lifestyle.
    - **Eco-Friendly Choice:** Green spaces help lower urban heat and improve air quality.
    """)

    st.header("🚀 What You’ll Find Here")
    st.markdown("""
    ✅ **Step-by-step gardening guides** ✅ **Best plants for rooftop gardening** ✅ **DIY solutions for low-cost gardening** ✅ **Organic farming techniques** ✅ **Community & expert advice** """)

    st.info("🌍 Start your RoofTop gardening journey today and make a positive impact on your health and the environment!")

# Chatbot Page
elif page == "Chatbot":
    st.title("🤖 Gardening Assistant Chatbot")
    st.markdown("Ask anything about **RoofTop gardening** and get instant responses")

    # Initialize Gemini Model
    def setup_gemini():
        API_KEY = "AIzaSyCZyJ5f0yYJWkxZOp7u-Txo0jgDDBhPB_k"  
        genai.configure(api_key=API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")  
        return model

    try:
        model = setup_gemini()

        # Chat UI
        with st.container():
            st.write("### 🌱 Ask Your Gardening Question Below:")
            user_input = st.text_area("Type your question here...", height=100)

            # Audio Input
            st.write("### 🎤 Or Upload an Audio File:")
            audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

            # Submit button
            if st.button("Generate Response 🌿"):
                if user_input or audio_file:
                    with st.spinner("Thinking... 💡"):
                        try:
                            if audio_file:
                                # Process the audio file
                                audio, _ = librosa.load(audio_file, sr=16000)
                                inputs = processor(audios=audio, return_tensors="pt", padding=True)
                                inputs.input_ids = inputs.input_ids.to("cuda")
                                generate_ids = model.generate(**inputs, max_length=256)
                                generate_ids = generate_ids[:, inputs.input_ids.size(1):]
                                response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                                st.subheader("🤖 AI Response:")
                                st.markdown(f"**{response}**")
                            else:
                                # Process text input
                                response = model.generate_content(user_input)
                                st.subheader("🤖 AI Response:")
                                st.markdown(f"**{response.text}**")
                        except Exception as e:
                            st.error(f"⚠️ Error: {e}. Please check your input and try again.")
                else:
                    st.warning("⚠️ Please enter a question or upload an audio file before submitting.")
    except Exception as e:
        st.error(f"⚠️ Error initializing the Chatbot: {e}. Please ensure your Gemini API key is correctly set.")


# Prompts Page
elif page == "Prompts":
    st.title("📝 RoofTop Gardening Prompts")
    st.markdown("Explore a comprehensive list of prompts to guide your rooftop gardening journey.")
    st.header("Newbie Prompt")
    st.markdown(""" I Have [available area]sqft of land on my terrace at [your area] in [season]. Give me some suggestions rooftop gardening with minimal effort and maximum output. Also provide pot dimensions and soil composition. """)
    # Organized Prompts
    st.header("🌿 How to Design Rooftop Gardening")
    st.markdown("""
    1. How to Design Rooftop Gardening  
    2. What are the key considerations for designing a rooftop garden?  
    3. How can I create a layout for my rooftop garden?  
    4. What types of containers are best for rooftop gardening?  
    5. How do I choose the right plants for my rooftop garden design?  
    6. What are the best materials for building raised beds on a rooftop?  
    7. How can I incorporate vertical gardening into my rooftop design?  
    8. What are some creative ways to use space in a small rooftop garden?  
    9. How can I design a rooftop garden that is aesthetically pleasing?  
    10. What are the best practices for ensuring proper drainage in a rooftop garden?  
    11. How can I create shaded areas in my rooftop garden?  
    """)

    st.header("🌱 Which Crops to Grow in Which Season")
    st.markdown("""
    12. What vegetables can I grow in the spring on my rooftop?  
    13. Which herbs thrive in summer rooftop gardens?  
    14. What are the best fall crops for rooftop gardening?  
    15. How can I grow winter vegetables in a rooftop garden?  
    16. What are the best fruits to grow in a rooftop garden by season?  
    17. How do I choose companion plants for my rooftop garden?  
    18. What are the best crops for container gardening on rooftops?  
    19. How can I extend the growing season in my rooftop garden?  
    20. What are the best microgreens to grow indoors or on a rooftop?  
    21. How do seasonal changes affect plant selection for rooftop gardens?  
    """)

    st.header("🌿 Proper Manure and Preparation Methods")
    st.markdown("""
    22. What types of manure are best for rooftop gardening?  
    23. How do I prepare manure for use in my rooftop garden?  
    24. What is the difference between compost and manure?  
    25. How can I make my own organic manure at home?  
    26. What are the benefits of using manure in rooftop gardening?  
    27. How do I apply manure to my rooftop garden?  
    28. What is the proper ratio of manure to soil for container gardening?  
    29. How can I tell if my manure is ready for use?  
    30. What precautions should I take when using manure in my garden?  
    31. How can I store manure safely for future use?  
    """)

    st.header("💧 Techniques for Manure and Water Management")
    st.markdown("""
    32. What are the best techniques for composting on a rooftop?  
    33. How can I integrate rainwater harvesting into my rooftop garden?  
    34. What are the benefits of using drip irrigation in rooftop gardening?  
    35. How do I set up a simple irrigation system for my rooftop garden?  
    36. What are the best practices for watering plants in containers?  
    37. How can I use greywater in my rooftop garden?  
    38. What are the signs of overwatering in rooftop plants?  
    39. How can I create a self-watering system for my rooftop garden?  
    40. What are the best times of day to water rooftop plants?  
    41. How can I prevent water runoff from my rooftop garden?  
    """)

    st.header("🐛 Pest Management in Rooftop Gardens")
    st.markdown("""
    42. What are common pests in rooftop gardens and how can I manage them?  
    43. How can I use companion planting to deter pests?  
    44. What natural pest control methods are effective for rooftop gardens?  
    45. How do I identify signs of pest infestations in my plants?  
    46. What are the best organic pesticides for rooftop gardening?  
    47. How can I attract beneficial insects to my rooftop garden?  
    48. What are the best practices for maintaining plant health to prevent pests?  
    49. How can I create barriers to protect my rooftop garden from pests?  
    50. What role do birds play in pest management on rooftops?  
    51. How can I use traps to control pests in my rooftop garden?  
    """)

    st.header("🌱 Soil Preparation and Maintenance")
    st.markdown("""
    52. What is the best soil mix for rooftop gardening?  
    53. How do I test the soil quality in my rooftop garden?  
    54. What amendments can I add to improve rooftop garden soil?  
    55. How often should I refresh the soil in my containers?  
    56. What are the signs of nutrient deficiency in rooftop plants?  
    57. How can I improve drainage in my rooftop garden soil?  
    58. What are the best practices for mulching in rooftop gardens?  
    59. How do I prevent soil erosion on my rooftop garden?  
    60. What is the importance of soil pH in rooftop gardening?  
    61. How can I create a soil management plan for my rooftop garden?  
    """)

    st.header("🌍 Sustainable Practices in Rooftop Gardening")
    st.markdown("""
    62. How can I make my rooftop garden more sustainable?  
    63. What are the benefits of using organic fertilizers in rooftop gardening?  
    64. How can I reduce waste in my rooftop garden?  
    65. What are the best practices for recycling materials in rooftop gardening?  
    66. How can I create a pollinator-friendly rooftop garden?  
    67. What are the benefits of using native plants in rooftop gardens?  
    68. How can I incorporate permaculture principles into my rooftop garden?  
    69. What are the best practices for sustainable water management in rooftop gardening?  
    70. How can I create a habitat for wildlife in my rooftop garden?  
    71. What are the benefits of using cover crops in rooftop gardening?  
    """)

    st.header("🍂 Seasonal Care and Maintenance")
    st.markdown("""
    72. How do I prepare my rooftop garden for winter?  
    73. What are the best practices for spring planting in rooftop gardens?  
    74. How can I protect my rooftop garden from summer heat?  
    75. What fall maintenance tasks should I perform in my rooftop garden?  
    76. How do I manage plant growth during seasonal transitions?  
    77. What are the signs that my rooftop garden needs seasonal care?  
    78. How can I extend the growing season with season extenders?  
    79. What are the best practices for harvesting crops from a rooftop garden?  
    80. How do I clean and store gardening tools for seasonal changes?  
    81. What are the benefits of crop rotation in rooftop gardening?  
    """)

    st.header("👥 Community and Education")
    st.markdown("""
    82. How can I get involved in community rooftop gardening projects?  
    83. What resources are available for learning about rooftop gardening?  
    84. How can I share my rooftop gardening experiences with others?  
    85. What are the benefits of joining a rooftop gardening club?  
    86. How can I teach children about rooftop gardening?  
    87. What workshops or classes are available for rooftop gardening enthusiasts?  
    88. How can I connect with local gardeners for advice and support?  
    89. What are the best online forums for rooftop gardening discussions?  
    90. How can I document my rooftop gardening journey?  
    91. What are the benefits of collaborating with local schools on gardening projects?  
    """)

    st.header("🚀 Innovations in Rooftop Gardening")
    st.markdown("""
    92. What are the latest trends in rooftop gardening technology?  
    93. How can I use smart gardening tools in my rooftop garden?  
    94. What are the benefits of hydroponics in rooftop gardening?  
    95. How can I incorporate aquaponics into my rooftop garden?  
    96. What are the advantages of using green roofs in urban areas?  
    97. How can I utilize solar energy for my rooftop garden?  
    98. What are the best apps for managing a rooftop garden?  
    99. How can I use sensors to monitor plant health in my rooftop garden?  
    100. What innovative materials can I use for rooftop gardening?  
    101. How can I create a sustainable rooftop garden that adapts to climate change?  
    """)

# Forum Page
elif page == "Forum":
    st.title("💬 Community Forum")
    
    # Check if the user is logged in
    if not st.session_state.logged_in:
        st.warning("🔒 Please log in to access the Community Forum.")
    else:
        st.markdown("Engage with fellow gardening enthusiasts, ask questions, and share experiences.")

        # Function to format datetime
        def format_datetime(dt):
            return dt.strftime("%Y-%m-%d %H:%M:%S")

        # Form to submit a new discussion
        with st.form(key="forum_form"):
            user_name = st.text_input("Your Name", placeholder="Enter your name")
            post_content = st.text_area("Share your thoughts or ask a question...", height=100)
            submit_button = st.form_submit_button("Post")

            if submit_button and user_name and post_content:
                timestamp = datetime.now()
                new_post = {"user": user_name, "content": post_content, "replies": [], "timestamp": timestamp}
                st.session_state.forum_data.append(new_post)
                st.success("✅ Your post has been added!")
                st.rerun()

        st.write("### 🌿 Community Discussions")
        if st.session_state.forum_data:
            for idx, post in enumerate(st.session_state.forum_data):
                with st.container():
                    st.markdown(f"**📝 {post['user']} says:**")
                    st.info(post["content"])
                    st.caption(f"Posted on: {format_datetime(post['timestamp'])}")

                    # Reply button to toggle reply form
                    reply_key = f"reply_button_{idx}"
                    if st.button("Reply", key=reply_key):
                        st.session_state.replying[idx] = not st.session_state.replying.get(idx, False)
                        st.rerun()

                    # Display reply form if the reply button is clicked
                    if st.session_state.replying.get(idx, False):
                        with st.form(key=f"reply_form_{idx}"):
                            reply_name = st.text_input("Your Name", placeholder="Enter your name", key=f"reply_name_{idx}")
                            reply_content = st.text_area("Your Reply...", height=50, key=f"reply_content_{idx}")
                            reply_submit_button = st.form_submit_button("Submit Reply")

                            if reply_submit_button and reply_name and reply_content:
                                reply_timestamp = datetime.now()
                                new_reply = {"user": reply_name, "content": reply_content, "timestamp": reply_timestamp}
                                st.session_state.forum_data[idx]["replies"].append(new_reply)
                                st.session_state.replying[idx] = False  # Hide reply form after submission
                                st.success("✅ Your reply has been added!")
                                st.rerun()

                    # Display replies
                    if post["replies"]:
                        st.write("**Replies:**")
                        