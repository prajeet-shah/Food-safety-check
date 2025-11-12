import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
import re
import requests
import zipfile
import os
from io import BytesIO
import gdown

# Page configuration
st.set_page_config(
    page_title="Food Adulteration Detector",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e86ab;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 300;
    }
    .risk-high {
        background: linear-gradient(135deg, #ff6b6b, #ee5a52);
        color: white;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
        margin: 20px 0;
    }
    .risk-medium {
        background: linear-gradient(135deg, #ffd93d, #ff9a3d);
        color: white;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(255, 217, 61, 0.3);
        margin: 20px 0;
    }
    .risk-low {
        background: linear-gradient(135deg, #6bcf7f, #4caF50);
        color: white;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(107, 207, 127, 0.3);
        margin: 20px 0;
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        text-align: center;
        margin: 10px 0;
    }
    .feature-card {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 10px 0;
    }
    .stButton>button {
        background: linear-gradient(135deg, #1f77b4, #2e86ab);
        color: white;
        border: none;
        padding: 12px 30px;
        border-radius: 25px;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(31, 119, 180, 0.4);
    }
    .download-progress {
        background: linear-gradient(135deg, #1f77b4, #2e86ab);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

class FoodAdulterationPredictor:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.risky_additives = {
            'high_risk': ['e102', 'e110', 'e122', 'e123', 'e124', 'e129', 'e133', 'e142', 'e151', 'e155',
                         'e211', 'e212', 'e213', 'e214', 'e215', 'e216', 'e217', 'e218', 'e219',
                         'e249', 'e250', 'e251', 'e252', 'e320', 'e321', 'e621', 'e951', 'e954'],
            'medium_risk': ['e104', 'e127', 'e128', 'e950', 'e952', 'e622', 'e623', 'e624', 'e625']
        }
        self.suspicious_patterns = [
            'artificial', 'preservative', 'synthetic', 'hydrogenated', 
            'modified starch', 'high fructose', 'msg', 'aspartame', 'saccharin'
        ]
    
    def download_model_from_drive(self):
        """Download model from Google Drive with progress tracking"""
        model_dir = "deberta_food_safety_model_cpu"
        zip_path = "deberta_food_safety_model_cpu.zip"
        
        # Google Drive file ID - extract from your shareable link
        # Your link: https://drive.google.com/drive/folders/1P6TLy8fT0EZAzFyb4itHQr1XCNwNVoib
        # https://drive.google.com/file/d/1ME_LnrTBdUySpQhMcTO40JC9wcwdc-Ux/view?usp=sharing
        # We need the direct download link for the zip file
        file_id = "1ME_LnrTBdUySpQhMcTO40JC9wcwdc-Ux"
        
        # Create download URL
        download_url = f"https://drive.google.com/uc?id={file_id}"
        
        if not os.path.exists(model_dir):
            st.info("📥 Downloading AI model (1GB)... This may take a few minutes.")
            
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Download with progress
                gdown.download(download_url, zip_path, quiet=False)
                
                # Update progress
                progress_bar.progress(50)
                status_text.text("📦 Extracting model files...")
                
                # Extract zip file
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(".")
                
                # Clean up
                os.remove(zip_path)
                
                progress_bar.progress(100)
                status_text.text("✅ Model downloaded successfully!")
                
            except Exception as e:
                st.error(f"❌ Download failed: {str(e)}")
                return False
        
        return True
    
    @st.cache_resource(show_spinner=False)
    def load_model(_self):
        """Load the DeBERTa model with caching"""
        try:
            # First, download the model if not exists
            if not _self.download_model_from_drive():
                return None, None
            
            model_dir = "deberta_food_safety_model_cpu"
            
            # Verify model files exist
            required_files = ['config.json', 'pytorch_model.bin', 'tokenizer_config.json']
            for file in required_files:
                if not os.path.exists(os.path.join(model_dir, file)):
                    st.error(f"❌ Missing model file: {file}")
                    return None, None
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            
            st.success("✅ AI Model loaded successfully!")
            return model, tokenizer
            
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
            return None, None
    
    def preprocess_text(self, product_name, ingredients, additives, allergens, nutrition_grade):
        """Preprocess input text for the model"""
        combined_text = (
            f"Product: {product_name}. "
            f"Ingredients: {ingredients}. "
            f"Additives: {', '.join(additives)}. "
            f"Allergens: {', '.join(allergens)}. "
            f"Nutrition Grade: {nutrition_grade}"
        )
        return combined_text
    
    def analyze_risk_factors(self, ingredients_text, additives_list):
        """Analyze risk factors in ingredients and additives"""
        found_risky = []
        risk_score = 0
        
        ingredients_lower = ingredients_text.lower()
        
        # Check risky additives
        for additive in additives_list:
            additive_clean = additive.lower().replace('en:', '').replace('e', '')
            if any(risky in additive_clean for risky in self.risky_additives['high_risk']):
                found_risky.append(f"🚨 High-risk additive: {additive}")
                risk_score += 3
            elif any(risky in additive_clean for risky in self.risky_additives['medium_risk']):
                found_risky.append(f"⚠️ Medium-risk additive: {additive}")
                risk_score += 1
        
        # Check suspicious patterns
        for pattern in self.suspicious_patterns:
            if pattern in ingredients_lower:
                found_risky.append(f"🔍 Suspicious ingredient: {pattern}")
                risk_score += 2
        
        # Additive count penalty
        additive_count = len(additives_list)
        if additive_count > 10:
            risk_score += 4
            found_risky.append(f"📊 Very high additive count: {additive_count}")
        elif additive_count > 5:
            risk_score += 2
            found_risky.append(f"📊 High additive count: {additive_count}")
        
        return found_risky, risk_score
    
    def predict(self, product_name, ingredients_text, additives_list, allergens_list, nutrition_grade):
        """Make prediction using the loaded model"""
        if self.model is None or self.tokenizer is None:
            return None, None, [], 0
        
        try:
            # Preprocess input
            combined_text = self.preprocess_text(
                product_name, ingredients_text, additives_list, allergens_list, nutrition_grade
            )
            
            # Tokenize
            inputs = self.tokenizer(
                combined_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                confidence, predicted_class = torch.max(predictions, dim=1)
            
            # Analyze risk factors
            found_risky, risk_score = self.analyze_risk_factors(ingredients_text, additives_list)
            
            return predicted_class.item(), confidence.item(), found_risky, risk_score
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return None, None, [], 0

def main():
    # Initialize predictor
    predictor = FoodAdulterationPredictor()
    
    # Header
    st.markdown('<div class="main-header">🍎 Food Adulteration Detector</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Food Safety Analysis • 98.9% Accuracy</div>', unsafe_allow_html=True)
    
    # Load model (with progress)
    with st.spinner('🚀 Loading AI Model... This might take a few minutes on first run'):
        predictor.model, predictor.tokenizer = predictor.load_model()
    
    # Show model status
    if predictor.model is None:
        st.error("""
        ❌ **Model failed to load.** 
        
        This could be because:
        - The model is still downloading (check the progress above)
        - Network connectivity issues
        - Google Drive download limits
        
        Please wait a moment and refresh the page. If the problem persists, check the Google Drive link.
        """)
        return
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📝 Product Information")
        
        with st.form("product_form"):
            product_name = st.text_input(
                "**Product Name**",
                placeholder="e.g., Chocolate Chip Cookies, Fruit Juice, etc."
            )
            
            ingredients_text = st.text_area(
                "**Ingredients List** *",
                height=150,
                placeholder="Paste the complete ingredients list here...\nExample: wheat flour, sugar, palm oil, cocoa, emulsifier (soya lecithin), artificial flavor"
            )
            
            additives = st.text_area(
                "**Additives (Optional)**",
                height=100,
                placeholder="Enter additives, one per line...\nExample: E102\nE211\nE621"
            )
            
            allergens = st.text_area(
                "**Allergens (Optional)**",
                height=80,
                placeholder="Enter allergens, one per line...\nExample: Milk\nSoy\nWheat"
            )
            
            nutrition_grade = st.selectbox(
                "**Nutrition Grade (Optional)**",
                ["", "A", "B", "C", "D", "E", "Unknown"]
            )
            
            submitted = st.form_submit_button("🔍 Analyze Product Safety", use_container_width=True)
    
    with col2:
        st.markdown("### ℹ️ How It Works")
        
        st.markdown("""
        <div class="feature-card">
        <h4>🧠 AI-Powered Analysis</h4>
        <p>Our DeBERTa v3 model analyzes product information with <b>98.9% accuracy</b> to detect potential adulteration risks.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
        <h4>🚨 Risk Detection</h4>
        <p>Identifies high-risk additives, artificial ingredients, and suspicious patterns that may indicate adulteration.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
        <h4>📊 Comprehensive Scoring</h4>
        <p>Provides detailed risk analysis with confidence scores and specific concerns identified.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick tips
        st.markdown("### 💡 Quick Tips")
        st.info("""
        - **Read labels carefully** - Look for long ingredient lists
        - **Watch for E-numbers** - Especially E100-199 (colors), E200-299 (preservatives)
        - **Choose natural** - Prefer products with recognizable ingredients
        - **Check nutrition grades** - A/B grades generally indicate healthier options
        """)
    
    # Handle form submission
    if submitted:
        if not ingredients_text.strip():
            st.error("⚠️ Please enter the ingredients list to analyze the product.")
        else:
            with st.spinner('🤖 Analyzing product for adulteration risks...'):
                # Process inputs
                additives_list = [additive.strip() for additive in additives.split('\n') if additive.strip()]
                allergens_list = [allergen.strip() for allergen in allergens.split('\n') if allergen.strip()]
                
                # Make prediction
                prediction, confidence, found_risky, risk_score = predictor.predict(
                    product_name, ingredients_text, additives_list, allergens_list, nutrition_grade
                )
                
                if prediction is not None:
                    # Display results
                    st.markdown("---")
                    st.markdown("## 📊 Analysis Results")
                    
                    # Risk level display
                    if prediction == 1 or risk_score >= 5:
                        risk_html = """
                        <div class="risk-high">
                            <h2>🚨 HIGH RISK - POSSIBLY ADULTERATED</h2>
                            <p>This product shows significant signs of potential adulteration. Exercise caution.</p>
                        </div>
                        """
                    elif risk_score >= 2:
                        risk_html = """
                        <div class="risk-medium">
                            <h2>⚠️ MEDIUM RISK - NEEDS CAUTION</h2>
                            <p>This product has some concerning factors. Review carefully.</p>
                        </div>
                        """
                    else:
                        risk_html = """
                        <div class="risk-low">
                            <h2>✅ LOW RISK - LIKELY SAFE</h2>
                            <p>This product appears safe with minimal risk factors.</p>
                        </div>
                        """
                    
                    st.markdown(risk_html, unsafe_allow_html=True)
                    
                    # Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown('<div class="metric-card"><h3>Confidence</h3><h2>{:.1%}</h2></div>'.format(confidence), 
                                   unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown('<div class="metric-card"><h3>Risk Score</h3><h2>{}/10</h2></div>'.format(min(risk_score, 10)), 
                                   unsafe_allow_html=True)
                    
                    with col3:
                        risk_level = "HIGH" if prediction == 1 or risk_score >= 5 else "MEDIUM" if risk_score >= 2 else "LOW"
                        st.markdown('<div class="metric-card"><h3>Risk Level</h3><h2>{}</h2></div>'.format(risk_level), 
                                   unsafe_allow_html=True)
                    
                    with col4:
                        st.markdown('<div class="metric-card"><h3>AI Model</h3><h2>DeBERTa v3</h2></div>', 
                                   unsafe_allow_html=True)
                    
                    # Risk factors
                    if found_risky:
                        st.markdown("### 🚨 Identified Risk Factors")
                        for factor in found_risky:
                            st.write(f"• {factor}")
                    else:
                        st.markdown("### ✅ No Major Risk Factors Found")
                        st.success("No high-risk additives or suspicious ingredients detected.")
                    
                    # Detailed explanation
                    st.markdown("### 📝 Detailed Analysis")
                    
                    if prediction == 1:
                        st.warning("""
                        **AI Detection Alert:** The model has identified patterns consistent with potentially adulterated products.
                        
                        **Recommendations:**
                        - Consider alternative products with simpler ingredient lists
                        - Look for certified organic or natural alternatives
                        - Consult nutritional experts if concerned
                        """)
                    else:
                        st.success("""
                        **AI Safety Assessment:** The product appears to meet standard safety criteria.
                        
                        **Good Practices:**
                        - Continue reading labels for new purchases
                        - Maintain balanced dietary choices
                        - Stay informed about food safety updates
                        """)
                    
                    # Ingredient summary
                    st.markdown("### 🔍 Ingredient Summary")
                    ingredient_count = len([x for x in ingredients_text.split(',') if x.strip()])
                    additive_count = len(additives_list)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Ingredients", ingredient_count)
                    with col2:
                        st.metric("Additives Found", additive_count)

    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='text-align: center; color: #666;'>
            <p>Built with ❤️ using DeBERTa v3 • Accuracy: 98.9%</p>
            <p>Always consult with food safety professionals for critical decisions</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()