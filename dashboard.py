import streamlit as st
import requests
import pandas as pd
import time

# Set Page Config
st.set_page_config(
    page_title="AI Smart Retail & Intelligence Platform",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Config
API_BASE_URL = "http://127.0.0.1:8000"

# Custom Styling
st.markdown("""
    <style>
        .main-header {
            font-size: 38px;
            font-weight: 700;
            color: #1E3A8A;
            margin-bottom: 2px;
        }
        .subheader {
            font-size: 18px;
            color: #4B5563;
            margin-bottom: 25px;
        }
        .card {
            background-color: #F3F4F6;
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #2563EB;
            margin-bottom: 15px;
        }
        .status-badge {
            padding: 5px 12px;
            border-radius: 20px;
            font-weight: 600;
            display: inline-block;
        }
    </style>
""", unsafe_allow_html=True)

# App Header
st.markdown('<div class="main-header">🛒 AI-Powered Smart Retail Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Real-time Loyalty Recognition, Checkout Product Scanning, Review Sentiment and Customer FAQ Assistant</div>', unsafe_allow_html=True)

# Define Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "👤 Loyalty & Face Kiosk", 
    "📦 Product Scanner", 
    "💬 Sentiment Analyzer", 
    "🤖 FAQ Support Bot", 
    "📈 Analytics Dashboard"
])

# ==========================================
# TAB 1: Loyalty & Face Recognition
# ==========================================
with tab1:
    st.header("👤 Customer Loyalty Face Recognition")
    st.write("Recognize returning loyalty program members automatically using face classification technology.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Face Snapshot Input")
        photo_option = st.radio("Choose Input Method:", ["Live Webcam Photo", "Upload Profile Image"])
        
        image_bytes = None
        
        if photo_option == "Live Webcam Photo":
            camera_img = st.camera_input("Capture Portrait")
            if camera_img:
                image_bytes = camera_img.getvalue()
        else:
            uploaded_file = st.file_uploader("Upload Portrait", type=["jpg", "png", "jpeg"])
            if uploaded_file:
                image_bytes = uploaded_file.getvalue()

    with col2:
        st.subheader("Recognition Results")
        if image_bytes:
            with st.spinner("Processing face embedding..."):
                try:
                    files = {"file": ("face.jpg", image_bytes, "image/jpeg")}
                    response = requests.post(f"{API_BASE_URL}/vision/recognize-face", files=files)
                    
                    if response.status_code == 200:
                        res = response.json()
                        st.success("Recognition Complete!")
                        
                        # Results Cards
                        st.markdown(f"""
                            <div class="card">
                                <h3>Matched Customer: {res['name']}</h3>
                                <p><strong>Matched Status:</strong> {"Active loyalty member" if res['matched'] else "New Visitor"}</p>
                                <p><strong>Visit Logged:</strong> {"Yes, customer visit updated in database" if res['logged'] else "No"}</p>
                                <p><strong>Detection Engine:</strong> {res['engine'].upper()}</p>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error(f"Error: API returned status code {response.status_code}")
                except Exception as e:
                    st.error(f"Could not connect to API: {e}")
        else:
            st.info("Awaiting input image to run face recognition.")

# ==========================================
# TAB 2: Product Scanner
# ==========================================
with tab2:
    st.header("📦 Checkout Product Classifier")
    st.write("Scan product packaging to automatically identify items for automatic checkout.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Product Packaging Photo")
        uploaded_prod = st.file_uploader("Upload Product Picture", type=["jpg", "png", "jpeg"], key="prod")
        
        if uploaded_prod:
            st.image(uploaded_prod, caption="Selected product photo", use_container_width=True)

    with col2:
        st.subheader("Scanning Results")
        if uploaded_prod:
            with st.spinner("Classifying product category..."):
                try:
                    files = {"file": ("product.jpg", uploaded_prod.getvalue(), "image/jpeg")}
                    response = requests.post(f"{API_BASE_URL}/vision/classify-product", files=files)
                    
                    if response.status_code == 200:
                        res = response.json()
                        st.success("Scanning Successful!")
                        
                        st.markdown(f"""
                            <div class="card">
                                <h3>Predicted Class: <span style="text-transform: capitalize;">{res['category']}</span></h3>
                                <p><strong>Confidence:</strong> {res['confidence']*100:.2f}%</p>
                            </div>
                        """, unsafe_allow_html=True)
                        st.progress(res['confidence'])
                    else:
                        st.error(f"Error: API returned status code {response.status_code}")
                except Exception as e:
                    st.error(f"Could not connect to API: {e}")
        else:
            st.info("Please upload a product photo to classify.")

# ==========================================
# TAB 3: Sentiment Analyzer
# ==========================================
with tab3:
    st.header("💬 Customer Feedback Sentiment Analysis")
    st.write("Paste reviews to determine customer satisfaction and feedback sentiments.")

    review_text = st.text_area("Customer Review / Feedback text:", height=150, placeholder="Write a review here (e.g., 'The support was very helpful and the product arrived quickly!')")
    
    if st.button("Analyze Sentiment", type="primary"):
        if review_text.strip():
            with st.spinner("Analyzing text sentiment..."):
                try:
                    payload = {"text": review_text}
                    response = requests.post(f"{API_BASE_URL}/nlp/analyze-sentiment", json=payload)
                    
                    if response.status_code == 200:
                        res = response.json()
                        sentiment = res['sentiment']
                        confidence = res['confidence']
                        
                        # Color coding based on sentiment
                        if sentiment == "positive":
                            color = "#D1FAE5"
                            text_color = "#065F46"
                            badge_text = "🟢 Positive"
                        elif sentiment == "negative":
                            color = "#FEE2E2"
                            text_color = "#991B1B"
                            badge_text = "🔴 Negative"
                        else:
                            color = "#FEF3C7"
                            text_color = "#92400E"
                            badge_text = "🟡 Neutral"
                            
                        st.markdown(f"""
                            <div style="background-color: {color}; padding: 25px; border-radius: 10px; border-left: 6px solid {text_color}; margin-top: 15px;">
                                <h3 style="color: {text_color}; margin-top: 0px;">Predicted Sentiment: {badge_text}</h3>
                                <p style="color: {text_color}; font-size: 16px; margin-bottom: 0px;">
                                    <strong>Confidence Score:</strong> {confidence * 100:.2f}%
                                </p>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error(f"Error: API returned status code {response.status_code}")
                except Exception as e:
                    st.error(f"Could not connect to API: {e}")
        else:
            st.warning("Please enter review text before submitting.")

# ==========================================
# TAB 4: FAQ Support Bot
# ==========================================
with tab4:
    st.header("🤖 Customer FAQ Assistant")
    st.write("Interact with the store's retail virtual assistant to resolve basic customer inquiries.")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User chat input
    user_input = st.chat_input("Ask a question (e.g. 'What is your refund policy?' or 'Do you accept Apple Pay?')")
    
    if user_input:
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Get bot response
        with st.spinner("Bot is typing..."):
            try:
                payload = {"message": user_input}
                response = requests.post(f"{API_BASE_URL}/chatbot", json=payload)
                
                if response.status_code == 200:
                    res = response.json()
                    bot_reply = res["reply"]
                    bot_intent = res["intent"]
                    
                    reply_with_intent = f"{bot_reply}\n\n*Intended Tag: {bot_intent.upper()}*"
                    
                    with st.chat_message("assistant"):
                        st.markdown(reply_with_intent)
                    st.session_state.messages.append({"role": "assistant", "content": reply_with_intent})
                else:
                    st.error(f"Error: API returned status code {response.status_code}")
            except Exception as e:
                st.error(f"Could not connect to API: {e}")

# ==========================================
# TAB 5: Store Telemetry Dashboard
# ==========================================
with tab5:
    st.header("📈 Store Managers & Executives Dashboard")
    st.write("Aggregated visual metrics from customer visits, product checkout scans, and support interactions.")

    if st.button("Refresh Telemetry Data", type="primary"):
        st.rerun()

    with st.spinner("Fetching database stats..."):
        try:
            response = requests.get(f"{API_BASE_URL}/dashboard/stats")
            
            if response.status_code == 200:
                stats = response.json()
                
                # Telemetry Cards
                st.subheader("Today's Key Performance Indicators")
                t_cols = st.columns(3)
                telemetry = stats["telemetry"]
                
                t_cols[0].metric("Total Store Visits", telemetry["total_visits"], "+12%")
                t_cols[1].metric("Loyalty Customer Visits", telemetry["returning_customers"], "+5%")
                t_cols[2].metric("Average Dwell Time", f"{telemetry['average_dwell_time_minutes']} min", "-3%")
                
                st.markdown("<hr>", unsafe_allow_html=True)
                
                # Charts Breakdown
                col_c1, col_c2 = st.columns(2)
                
                with col_c1:
                    st.subheader("Customer Satisfaction (Sentiment)")
                    sent = stats["sentiment_breakdown"]
                    sent_df = pd.DataFrame({
                        "Sentiment": list(sent.keys()),
                        "Percentage (%)": list(sent.values())
                    })
                    st.bar_chart(sent_df.set_index("Sentiment"))
                    
                with col_c2:
                    st.subheader("Product Category Checkout Sales")
                    sales = stats["product_sales_distribution"]
                    sales_df = pd.DataFrame({
                        "Category": list(sales.keys()),
                        "Market Share (%)": list(sales.values())
                    })
                    st.bar_chart(sales_df.set_index("Category"))
            else:
                st.error(f"Error: API returned status code {response.status_code}")
        except Exception as e:
            st.error(f"Could not connect to API: {e}")
