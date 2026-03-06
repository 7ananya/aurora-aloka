#final code 

import streamlit as st
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------

st.set_page_config(
    page_title="Aurora Āloka",
    layout="wide"
)

# -------------------------------------------------
# GLOBAL STYLE
# -------------------------------------------------

st.markdown("""
<style>

.stApp {
background:
radial-gradient(circle at 20% 30%, rgba(0,255,200,0.15), transparent 40%),
radial-gradient(circle at 80% 40%, rgba(120,0,255,0.15), transparent 40%),
radial-gradient(circle at 50% 90%, rgba(0,150,255,0.15), transparent 40%),
linear-gradient(180deg,#020617,#000000);
color:white;
}

/* star field */

.stApp:before{
content:"";
position:fixed;
top:0;
left:0;
width:100%;
height:100%;
background-image:
radial-gradient(white 1px, transparent 1px),
radial-gradient(white 1px, transparent 1px),
radial-gradient(white 2px, transparent 2px);
background-size: 80px 80px,120px 120px,200px 200px;
background-position: 0 0,40px 40px,100px 50px;
opacity:0.2;
z-index:-1;
}

/* TITLE */

.title{
font-size:140px;
font-weight:900;
text-align:center;
margin-top:30px;

background:linear-gradient(
90deg,
#7ef9ff,
#8a7dff,
#d16bff,
#7ef9ff
);

background-size:300%;

-webkit-background-clip:text;
-webkit-text-fill-color:transparent;

letter-spacing:6px;

animation:glowFlow 6s linear infinite;
}

@keyframes glowFlow{
0%{background-position:0%}
100%{background-position:300%}
}

/* subtitle */

.subtitle{
text-align:center;
opacity:0.75;
font-size:28px;
margin-bottom:60px;
letter-spacing:2px;
}

/* glass card */

.card{
background:rgba(255,255,255,0.05);
padding:30px;
border-radius:16px;
border:1px solid rgba(255,255,255,0.1);
backdrop-filter:blur(12px);
}

/* uploader */

[data-testid="stFileUploader"]{
background:rgba(255,255,255,0.04);
border:1px solid rgba(255,255,255,0.15);
border-radius:16px;
padding:25px;
backdrop-filter:blur(8px);
transition:0.3s;
}

[data-testid="stFileUploader"]:hover{
border:1px solid #7ef9ff;
box-shadow:0 0 25px rgba(126,249,255,0.25);
}

[data-testid="stFileUploader"] button{
background:linear-gradient(90deg,#7ef9ff,#8a7dff);
border:none;
color:black;
font-weight:600;
border-radius:10px;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# HEADER
# -------------------------------------------------

st.markdown(
"""
<p class="title">Aurora Āloka</p>
<p class="subtitle">Illuminating Vision with AI</p>
""",
unsafe_allow_html=True
)

# -------------------------------------------------
# CLASSES
# -------------------------------------------------

classes = [
'airplane','automobile','bird','cat','deer',
'dog','frog','horse','ship','truck'
]

# -------------------------------------------------
# INFO
# -------------------------------------------------

st.markdown(
"""
**Supported object classes**

airplane • automobile • bird • cat • deer  
dog • frog • horse • ship • truck
"""
)

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------

@st.cache_resource
def load_model():

    model = models.resnet18(pretrained=False)

    model.fc = nn.Linear(model.fc.in_features,10)

    model.load_state_dict(
        torch.load("aurora_aloka_resnet18.pth",map_location="cpu")
    )

    model.eval()

    return model

model = load_model()

# -------------------------------------------------
# TRANSFORM
# -------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# -------------------------------------------------
# UPLOADER
# -------------------------------------------------

st.markdown("### Upload Image")

uploaded_file = st.file_uploader(
"",
type=["jpg","jpeg","png"],
label_visibility="collapsed"
)

# -------------------------------------------------
# PREDICTION
# -------------------------------------------------

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    col1,col2 = st.columns([1,1])

    with col1:

        st.markdown("### Input Image")

        st.image(image,width=360)

    img = transform(image).unsqueeze(0)

    with st.spinner("Analyzing image..."):

        with torch.no_grad():

            outputs = model(img)

            probs = torch.softmax(outputs,dim=1)

    probs = probs.numpy()[0]

    top = np.argsort(probs)[::-1][:3]

    prediction = classes[top[0]]

    confidence = probs[top[0]]*100

    with col2:

        st.markdown(
        f"""
        <div class="card">
        <h3>Prediction</h3>
        <p style="opacity:0.6;">Detected Object</p>
        <h1 style="margin-top:-10px;">{prediction}</h1>
        </div>
        """,
        unsafe_allow_html=True
        )

        st.markdown("### Confidence")

        st.progress(int(confidence))

        st.write(f"{confidence:.2f}%")

        st.markdown("### Top Predictions")

        for i in top:

            st.write(f"{classes[i]} — {probs[i]*100:.2f}%")

    # -------------------------------------------------
    # CHART
    # -------------------------------------------------

    df = pd.DataFrame({
        "Class":classes,
        "Probability":probs
    })

    fig = px.bar(
        df,
        x="Probability",
        y="Class",
        orientation="h",
        color="Probability",
        color_continuous_scale="viridis"
    )

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(fig,use_container_width=True)

# -------------------------------------------------
# FOOTER
# -------------------------------------------------

st.markdown("---")

st.markdown(
"<center>Aurora Āloka — Computer Vision Demo<br>Created by Ananya Terwadkar</center>",
unsafe_allow_html=True
)