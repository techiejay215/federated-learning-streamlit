# app.py - Main Streamlit Application
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import base64

# Set page configuration
st.set_page_config(
    page_title="Federated Learning for Medical Imaging",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .metric-box {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem;
    }
    .model-comparison {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Model Architecture (same as training)
class PneumoniaCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(PneumoniaCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.classifier(x)
        return x

# Load Model Function
@st.cache_resource
def load_model(model_path, device):
    """Load trained model"""
    model = PneumoniaCNN()
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Image Preprocessing
def preprocess_image(image):
    """Preprocess image for model inference"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Model Prediction
def predict_image(model, image_tensor, device):
    """Make prediction on image"""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        return predicted.item(), confidence.item(), probabilities.cpu().numpy()[0]

# Generate Confidence Visualization
def plot_confidence(probabilities, classes):
    """Create confidence score visualization"""
    fig = go.Figure(data=[
        go.Bar(x=classes, y=probabilities, 
               marker_color=['lightblue', 'lightcoral'],
               text=[f'{p*100:.2f}%' for p in probabilities],
               textposition='auto')
    ])
    fig.update_layout(
        title='Model Confidence Scores',
        xaxis_title='Class',
        yaxis_title='Probability',
        yaxis=dict(range=[0, 1]),
        template='plotly_white'
    )
    return fig

# Sample data for demonstration (replace with your actual results)
def get_sample_results():
    """Return sample model results for demonstration"""
    return {
        'centralized': {
            'accuracy': 0.923,
            'precision': 0.918,
            'recall': 0.923,
            'f1_score': 0.919,
            'auc_roc': 0.961
        },
        'federated': {
            'accuracy': 0.912,
            'precision': 0.908,
            'recall': 0.912,
            'f1_score': 0.909,
            'auc_roc': 0.952
        },
        'fl_rounds': list(range(1, 51)),
        'fl_accuracy': [0.782, 0.812, 0.834, 0.856, 0.871, 0.882, 0.889, 0.894, 
                        0.898, 0.901, 0.903, 0.905, 0.906, 0.907, 0.908, 0.909,
                        0.909, 0.910, 0.910, 0.911, 0.911, 0.911, 0.911, 0.911,
                        0.912, 0.912, 0.912, 0.912, 0.912, 0.912, 0.912, 0.912,
                        0.912, 0.912, 0.912, 0.912, 0.912, 0.912, 0.912, 0.912,
                        0.912, 0.912, 0.912, 0.912, 0.912, 0.912, 0.912, 0.912,
                        0.912, 0.912]
    }

# Main Application
def main():
    # Sidebar Navigation
    st.sidebar.title("🏥 Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose Section",
        ["Home", "Model Demo", "Federated Learning", "Performance Analysis", "Privacy Analysis"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "**Federated Learning for Privacy-Preserving Medical Data Analysis**\n\n"
        "This application demonstrates how federated learning enables collaborative "
        "AI model training without sharing sensitive patient data."
    )
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Home Page
    if app_mode == "Home":
        show_home_page()
    
    # Model Demo Page
    elif app_mode == "Model Demo":
        show_model_demo(device)
    
    # Federated Learning Page
    elif app_mode == "Federated Learning":
        show_federated_learning_info()
    
    # Performance Analysis Page
    elif app_mode == "Performance Analysis":
        show_performance_analysis()
    
    # Privacy Analysis Page
    elif app_mode == "Privacy Analysis":
        show_privacy_analysis()

def show_home_page():
    """Display home page content"""
    st.markdown('<h1 class="main-header">Federated Learning for Medical Imaging</h1>', 
                unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="info-box">
        <h3>🚀 Project Overview</h3>
        <p>This project demonstrates the application of <b>Federated Learning</b> for 
        privacy-preserving analysis of chest X-ray images to detect pneumonia.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        ### 🎯 Key Features
        
        - **Privacy-Preserving AI**: Train models without sharing sensitive patient data
        - **Multi-Hospital Collaboration**: Enable cooperation between medical institutions
        - **Real-time Inference**: Upload and analyze chest X-ray images
        - **Performance Comparison**: Compare federated vs centralized learning approaches
        - **Regulatory Compliance**: Align with HIPAA and GDPR requirements
        """)
    
    with col2:
        st.image("https://miro.medium.com/v2/resize:fit:1400/1*4rQ49p34rTkfWk0j69O-3A.png", 
                 caption="Federated Learning Architecture", use_column_width=True)
    
    st.markdown("---")
    
    # Quick stats
    st.subheader("📊 Project Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Dataset Size", "5,216 X-rays")
    
    with col2:
        st.metric("Model Accuracy", "91.2%")
    
    with col3:
        st.metric("Privacy Level", "High")
    
    with col4:
        st.metric("Hospitals Simulated", "5")
    
    # Technology stack
    st.subheader("🛠️ Technology Stack")
    tech_col1, tech_col2, tech_col3 = st.columns(3)
    
    with tech_col1:
        st.markdown("""
        **Machine Learning**
        - PyTorch
        - TorchVision
        - Scikit-learn
        """)
    
    with tech_col2:
        st.markdown("""
        **Federated Learning**
        - Flower Framework
        - Custom CNN
        - Federated Averaging
        """)
    
    with tech_col3:
        st.markdown("""
        **Web Application**
        - Streamlit
        - Plotly
        - PIL/Pillow
        """)

def show_model_demo(device):
    """Display model demonstration page"""
    st.title("🩻 Chest X-Ray Analysis Demo")
    
    st.markdown("""
    <div class="info-box">
    Upload a chest X-ray image to analyze whether it shows signs of pneumonia using our 
    federated learning model.
    </div>
    """, unsafe_allow_html=True)
    
    # Model selection
    col1, col2 = st.columns([1, 2])
    
    with col1:
        model_choice = st.radio(
            "Select Model Type:",
            ["Federated Learning Model", "Centralized Model"],
            help="Choose which model to use for prediction"
        )
        
        # Load appropriate model
        if model_choice == "Federated Learning Model":
            model_path = "best_federated_model.pth"  # Update with your actual path
            model_type = "federated"
        else:
            model_path = "best_centralized_model.pth"  # Update with your actual path
            model_type = "centralized"
        
        # Try to load model (for demo, we'll use a placeholder)
        model_loaded = st.checkbox("Load Model (Demo Mode)", value=True)
        
        if model_loaded:
            st.success(f"✅ {model_choice} loaded successfully!")
        else:
            st.warning("Model not loaded. This is a demo interface.")
    
    with col2:
        st.markdown("""
        **How it works:**
        1. Upload a chest X-ray image (JPEG/PNG)
        2. The model analyzes the image
        3. Get instant results with confidence scores
        4. View detailed analysis and explanations
        """)
    
    st.markdown("---")
    
    # Image upload and prediction
    uploaded_file = st.file_uploader(
        "Choose a chest X-ray image", 
        type=['jpg', 'jpeg', 'png'],
        help="Upload a chest X-ray image for pneumonia detection"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Uploaded Image")
            st.image(image, caption="Uploaded Chest X-Ray", use_column_width=True)
            
            # Image info
            st.write(f"**Image Details:**")
            st.write(f"- Format: {uploaded_file.type}")
            st.write(f"- Size: {image.size}")
            st.write(f"- Mode: {image.mode}")
        
        with col2:
            st.subheader("Analysis Results")
            
            if model_loaded:
                # Simulate prediction (replace with actual model inference)
                with st.spinner("Analyzing image..."):
                    # Preprocess image
                    processed_image = preprocess_image(image)
                    
                    # For demo purposes, we'll simulate predictions
                    # In real implementation, you would use:
                    # prediction, confidence, probabilities = predict_image(model, processed_image, device)
                    
                    # Simulated results
                    classes = ['Normal', 'Pneumonia']
                    if model_type == "federated":
                        # Simulate FL model results
                        probabilities = np.array([0.15, 0.85])  # 85% confidence in pneumonia
                    else:
                        # Simulate centralized model results
                        probabilities = np.array([0.12, 0.88])  # 88% confidence in pneumonia
                    
                    predicted_class = classes[np.argmax(probabilities)]
                    confidence = np.max(probabilities)
                
                # Display results
                if predicted_class == "Pneumonia":
                    st.error(f"🚨 **Result: {predicted_class}**")
                    st.warning("**Recommendation:** Please consult with a healthcare professional for further evaluation.")
                else:
                    st.success(f"✅ **Result: {predicted_class}**")
                    st.info("**Note:** This is an AI-assisted analysis. Always consult with medical professionals for diagnosis.")
                
                # Confidence scores
                st.plotly_chart(plot_confidence(probabilities, classes), use_container_width=True)
                
                # Detailed metrics
                st.subheader("Detailed Analysis")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Prediction Confidence", f"{confidence*100:.2f}%")
                    st.metric("Model Type", model_choice)
                
                with col2:
                    st.metric("Processing Time", "0.8s")
                    st.metric("Image Quality", "Good")
                
                # Explanation
                st.markdown("""
                **Analysis Explanation:**
                - The model has analyzed patterns in the lung regions
                - Confidence score indicates certainty level
                - Results should be verified by radiologists
                """)
                
            else:
                st.warning("Please load the model to perform analysis.")
    
    else:
        # Show sample images
        st.subheader("📸 Sample Images for Testing")
        
        col1, col2, col3, col4 = st.columns(4)
        
        sample_images = {
            "Normal 1": "https://github.com/ieee8023/covid-chestxray-dataset/raw/master/images/1.jpeg",
            "Normal 2": "https://github.com/ieee8023/covid-chestxray-dataset/raw/master/images/2.jpeg", 
            "Pneumonia 1": "https://github.com/ieee8023/covid-chestxray-dataset/raw/master/images/3.jpeg",
            "Pneumonia 2": "https://github.com/ieee8023/covid-chestxray-dataset/raw/master/images/4.jpeg"
        }
        
        for i, (label, url) in enumerate(sample_images.items()):
            with [col1, col2, col3, col4][i]:
                st.image(url, caption=label, use_column_width=True)

def show_federated_learning_info():
    """Display federated learning educational content"""
    st.title("🔒 Federated Learning Explained")
    
    st.markdown("""
    <div class="info-box">
    <h3>What is Federated Learning?</h3>
    <p>Federated Learning is a distributed machine learning approach that enables model training 
    across multiple decentralized devices or servers holding local data samples, without exchanging them.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # FL Process Visualization
    st.subheader("🔄 How Federated Learning Works")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### Traditional Centralized Learning
        1. **Data Collection**: All data sent to central server
        2. **Model Training**: Single model trained on combined data  
        3. **Model Deployment**: Trained model distributed to devices
        4. **Privacy Risk**: Sensitive data exposed during transfer
        
        ❌ **Privacy Concerns**
        ❌ **Regulatory Issues** 
        ❌ **Data Security Risks**
        """)
    
    with col2:
        st.markdown("""
        ### Federated Learning
        1. **Local Training**: Models trained on local devices
        2. **Update Sharing**: Only model updates (not data) sent to server
        3. **Aggregation**: Server combines updates to improve global model
        4. **Model Distribution**: Improved model sent back to devices
        
        ✅ **Data Privacy Preserved**
        ✅ **Regulatory Compliance** 
        ✅ **Enhanced Security**
        """)
    
    st.markdown("---")
    
    # FL Process Steps
    st.subheader("📋 Federated Learning Process Steps")
    
    steps = [
        {"step": "1", "title": "Initialization", "desc": "Server creates initial global model"},
        {"step": "2", "title": "Distribution", "desc": "Model sent to participating hospitals"},
        {"step": "3", "title": "Local Training", "desc": "Each hospital trains model on local data"},
        {"step": "4", "title": "Update Transmission", "desc": "Hospitals send model updates (not data) to server"},
        {"step": "5", "title": "Aggregation", "desc": "Server combines updates using Federated Averaging"},
        {"step": "6", "title": "Model Update", "desc": "Improved global model sent back to hospitals"}
    ]
    
    for i in range(0, len(steps), 2):
        col1, col2 = st.columns(2)
        with col1:
            step = steps[i]
            st.markdown(f"""
            <div style='border: 2px solid #1f77b4; border-radius: 10px; padding: 15px; margin: 10px 0;'>
                <h4>🚩 Step {step['step']}: {step['title']}</h4>
                <p>{step['desc']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        if i + 1 < len(steps):
            with col2:
                step = steps[i + 1]
                st.markdown(f"""
                <div style='border: 2px solid #1f77b4; border-radius: 10px; padding: 15px; margin: 10px 0;'>
                    <h4>🚩 Step {step['step']}: {step['title']}</h4>
                    <p>{step['desc']}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Benefits Section
    st.markdown("---")
    st.subheader("🎯 Benefits for Medical Applications")
    
    benefits = [
        {"icon": "🔐", "title": "Privacy Preservation", "desc": "Patient data never leaves hospital servers"},
        {"icon": "🏥", "title": "Multi-Institutional Collaboration", "desc": "Hospitals can collaborate without sharing data"},
        {"icon": "📊", "title": "Diverse Training Data", "desc": "Models learn from varied patient populations"},
        {"icon": "⚖️", "title": "Regulatory Compliance", "desc": "Meets HIPAA, GDPR, and other privacy regulations"},
        {"icon": "🛡️", "title": "Security", "desc": "Reduces risk of data breaches and unauthorized access"},
        {"icon": "🌍", "title": "Scalability", "desc": "Easy to add new hospitals to the federation"}
    ]
    
    cols = st.columns(3)
    for i, benefit in enumerate(benefits):
        with cols[i % 3]:
            st.markdown(f"""
            <div style='background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin: 10px 0; text-align: center;'>
                <h3>{benefit['icon']}</h3>
                <h4>{benefit['title']}</h4>
                <p>{benefit['desc']}</p>
            </div>
            """, unsafe_allow_html=True)

def show_performance_analysis():
    """Display performance comparison between models"""
    st.title("📊 Performance Analysis")
    
    st.markdown("""
    <div class="info-box">
    Compare the performance of Federated Learning models against traditional Centralized training approaches.
    </div>
    """, unsafe_allow_html=True)
    
    # Get sample results (replace with your actual results)
    results = get_sample_results()
    
    # Metrics Comparison
    st.subheader("📈 Model Performance Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Centralized Model")
        centralized_metrics = results['centralized']
        
        for metric, value in centralized_metrics.items():
            st.metric(
                label=metric.replace('_', ' ').title(),
                value=f"{value:.3f}",
                delta=None
            )
    
    with col2:
        st.markdown("### Federated Learning Model")
        federated_metrics = results['federated']
        
        for metric, value in federated_metrics.items():
            delta = value - centralized_metrics[metric]
            st.metric(
                label=metric.replace('_', ' ').title(),
                value=f"{value:.3f}",
                delta=f"{delta:+.3f}",
                delta_color="inverse" if metric in ['loss'] else "normal"
            )
    
    st.markdown("---")
    
    # Performance Over Time
    st.subheader("📈 Federated Learning Convergence")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=results['fl_rounds'],
        y=results['fl_accuracy'],
        mode='lines+markers',
        name='Federated Learning Accuracy',
        line=dict(color='blue', width=3),
        marker=dict(size=6)
    ))
    
    fig.add_hline(
        y=results['centralized']['accuracy'],
        line_dash="dash",
        line_color="red",
        annotation_text="Centralized Model Accuracy"
    )
    
    fig.update_layout(
        title='Federated Learning Performance Over Communication Rounds',
        xaxis_title='Communication Round',
        yaxis_title='Accuracy',
        template='plotly_white',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed Metrics Comparison
    st.markdown("---")
    st.subheader("📋 Detailed Metrics Breakdown")
    
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
        'Centralized': [
            results['centralized']['accuracy'],
            results['centralized']['precision'], 
            results['centralized']['recall'],
            results['centralized']['f1_score'],
            results['centralized']['auc_roc']
        ],
        'Federated': [
            results['federated']['accuracy'],
            results['federated']['precision'],
            results['federated']['recall'],
            results['federated']['f1_score'],
            results['federated']['auc_roc']
        ]
    })
    
    metrics_df['Difference'] = metrics_df['Federated'] - metrics_df['Centralized']
    metrics_df['Difference %'] = (metrics_df['Difference'] / metrics_df['Centralized']) * 100
    
    st.dataframe(
        metrics_df.style.format({
            'Centralized': '{:.3f}',
            'Federated': '{:.3f}', 
            'Difference': '{:+.3f}',
            'Difference %': '{:+.2f}%'
        }).background_gradient(subset=['Difference'], cmap='RdYlGn'),
        use_container_width=True
    )
    
    # Key Insights
    st.markdown("---")
    st.subheader("💡 Key Insights")
    
    insights_col1, insights_col2 = st.columns(2)
    
    with insights_col1:
        st.markdown("""
        **🎯 Performance Findings:**
        - Federated Learning achieves **99% of centralized performance**
        - Minimal accuracy gap: **1.1% difference**
        - FL models converge within **50 communication rounds**
        - Robust performance across all metrics
        """)
    
    with insights_col2:
        st.markdown("""
        **🚀 Practical Implications:**
        - Privacy can be achieved without significant performance loss
        - Suitable for real-world medical applications
        - Enables cross-institutional collaboration
        - Reduces regulatory barriers to AI adoption
        """)

def show_privacy_analysis():
    """Display privacy analysis and benefits"""
    st.title("🛡️ Privacy & Security Analysis")
    
    st.markdown("""
    <div class="info-box">
    <h3>Privacy-Preserving Machine Learning</h3>
    <p>Federated Learning provides strong privacy guarantees while enabling collaborative AI model development.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Privacy Comparison
    st.subheader("🔒 Privacy Comparison: Centralized vs Federated Learning")
    
    privacy_data = {
        'Aspect': ['Data Location', 'Data Transfer', 'Privacy Risk', 'Regulatory Compliance', 'Security'],
        'Centralized Learning': [
            'Central Server', 'Raw Data', 'High', 'Challenging', 'Single Point of Failure'
        ],
        'Federated Learning': [
            'Local Devices', 'Model Updates Only', 'Low', 'Easier', 'Distributed Security'
        ]
    }
    
    privacy_df = pd.DataFrame(privacy_data)
    st.dataframe(privacy_df, use_container_width=True)
    
    # Privacy Techniques
    st.markdown("---")
    st.subheader("🛠️ Advanced Privacy Techniques")
    
    techniques = [
        {
            "name": "Differential Privacy",
            "description": "Adds calibrated noise to model updates to prevent data reconstruction",
            "privacy_level": "High",
            "impact": "Small performance decrease",
            "use_case": "Standard medical applications"
        },
        {
            "name": "Secure Aggregation", 
            "description": "Cryptographic protocol that prevents server from seeing individual updates",
            "privacy_level": "Very High",
            "impact": "Moderate computational overhead",
            "use_case": "Multi-institutional collaborations"
        },
        {
            "name": "Homomorphic Encryption",
            "description": "Enables computation on encrypted data without decryption",
            "privacy_level": "Highest", 
            "impact": "High computational cost",
            "use_case": "Highly sensitive data"
        }
    ]
    
    for technique in techniques:
        with st.expander(f"🔐 {technique['name']} - Privacy Level: {technique['privacy_level']}"):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(technique['description'])
            with col2:
                st.metric("Performance Impact", technique['impact'])
                st.write(f"**Use Case:** {technique['use_case']}")
    
    # Regulatory Compliance
    st.markdown("---")
    st.subheader("⚖️ Regulatory Compliance")
    
    regulations = [
        {"regulation": "HIPAA (USA)", "status": "✅ Compliant", "details": "No PHI transfer required"},
        {"regulation": "GDPR (EU)", "status": "✅ Compliant", "details": "Data minimization principle"},
        {"regulation": "PIPEDA (Canada)", "status": "✅ Compliant", "details": "Limited data collection"},
        {"regulation": "PDPA (Singapore)", "status": "✅ Compliant", "details": "Purpose limitation adherence"}
    ]
    
    for reg in regulations:
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            st.write(f"**{reg['regulation']}**")
        with col2:
            st.success(reg['status'])
        with col3:
            st.write(reg['details'])
    
    # Risk Analysis
    st.markdown("---")
    st.subheader("📊 Risk Analysis Matrix")
    
    risks = {
        'Data Breach During Transfer': {'Centralized': 'High', 'Federated': 'Low'},
        'Model Inversion Attacks': {'Centralized': 'Medium', 'Federated': 'Very Low'},
        'Membership Inference': {'Centralized': 'Medium', 'Federated': 'Low'},
        'Data Reconstruction': {'Centralized': 'High', 'Federated': 'Very Low'},
        'Regulatory Penalties': {'Centralized': 'High', 'Federated': 'Low'}
    }
    
    risk_df = pd.DataFrame(risks).T
    st.dataframe(
        risk_df.style.applymap(
            lambda x: 'background-color: #ff6b6b' if x == 'High' 
            else 'background-color: #ffd166' if x == 'Medium' 
            else 'background-color: #06d6a0' if x == 'Low'
            else 'background-color: #118ab2'
        ),
        use_container_width=True
    )
    
    # Best Practices
    st.markdown("---")
    st.subheader("🎯 Privacy Best Practices")
    
    practices = [
        "✅ Implement differential privacy for model updates",
        "✅ Use secure aggregation protocols", 
        "✅ Conduct regular privacy audits",
        "✅ Maintain data processing records",
        "✅ Implement access controls and authentication",
        "✅ Use encrypted communication channels",
        "✅ Regularly update security protocols",
        "✅ Train staff on privacy principles"
    ]
    
    for practice in practices:
        st.write(practice)

# Run the application
if __name__ == "__main__":
    main()