import streamlit as st
import nutrition
from langchain.agents import AgentType, initialize_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_google_genai import GoogleGenerativeAI
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import speech_recognition as sr
import warnings

warnings.filterwarnings('ignore')

#from langchain_community.llms import Cohere

#from secret_key import cohereapi_key
#import os
#os.environ['COHERE_API_KEY'] = cohereapi_key

from secret_key import googleapi_key,serpapi_key
import os
os.environ['GOOGLE_API_KEY'] = googleapi_key
os.environ['SERPAPI_API_KEY'] = serpapi_key

llm = GoogleGenerativeAI(model="gemini-2.0-flash")

# AI agent
tools = load_tools(["serpapi"], llm=llm)

# AI agent initialization
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handle_parsing_errors = True)


st.title("Smart AI Nutrition Assistant :sunglasses:")
st.subheader("by :blue[_Deepneel Majumdar_]",divider="blue")
st.subheader("created on :green[_langchain framework_], powered by :orange[_Gemini_] LLM",divider="blue")
st.sidebar.subheader(":green[Select your Health Choices]",divider="blue")

health_goal = st.sidebar.selectbox("Choose your health goal",['Fitness','Body building','Weight loss','Mental health','Lifestyle'])

medical_conditions = st.sidebar.selectbox("Select your medical condition",['Diabetic','Obesity','Heart disease','Kidney disease','Liver disease','Healthy'])

fitness_routines = st.sidebar.selectbox("Pick your fitness routine",['Endurance training','Powerlifting','General fitness','Functional Training','Yoga','Crossfit','Sports performance','Body recomposition'])

preferences = st.sidebar.selectbox("Elect your food preference",['Vegetarian','Vegan','Non-vegetarian','Gluten-free','Ketogenic','Diabetic-friendly','Dairy-free','Low-carb'])

button1 = st.sidebar.button("_Create my Nutrition Meal Plan_",key="b1")

main_body = st.empty()

if button1:
    main_body.text("Processing ......Please wait......✅✅✅")
    output = nutrition.diet_planner(health_goal,medical_conditions,fitness_routines,preferences)
    st.write(output['chain4'])

st.sidebar.subheader(":red[Chat with AI assistant]",divider="blue")
stquery = st.sidebar.text_input(label="Type here",placeholder="Enter your query!")
button2 = st.sidebar.button("_Submit_",key="b2")

if button2:
    main_body.text("Processing ......Please wait......✅✅✅")
    st.write(agent.run(stquery))


## Translate Speech to Text

# Initialize recognizer
recognizer = sr.Recognizer()

st.sidebar.subheader(":orange[Click to speak with AI bot]",divider="blue")
button4 = st.sidebar.button("_Click for Speech to Text_",key="b4")

if button4:
    # Use the microphone as source
    with sr.Microphone() as source:
        #print("Listening... Speak now!")
        st.write("Listening... Speak now!")
        # Optional: adjust for ambient noise
        recognizer.adjust_for_ambient_noise(source)
    
        # Listen to the audio
        audio = recognizer.listen(source)

        try:
            # Convert speech to text using Google's API
            text = recognizer.recognize_google(audio)
            #print("You said:", text)
            st.header("You said :: " + text)
            # AI agent
            tools = load_tools(["serpapi"], llm=llm)
            # AI agent initialization
            agent2 = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handle_parsing_errors = True)
            if text:
                main_body.text("Processing ......Please wait......✅✅✅")
                st.write(agent2.run("Tell me about"+text))

        except sr.UnknownValueError:
            print("Sorry, I could not understand the audio.")
        except sr.RequestError as e:
            print(f"Could not request results; {e}")


## Image upload and recognition

# Load BLIP model and processor
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_model()

# For Streamlit UI
st.sidebar.subheader(":green[Upload your image for recognition]",divider="blue")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])


if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    #st.image(image, caption="Uploaded Image", use_column_width=True)

    
    caption=""
    if st.sidebar.button("_Generate Caption_"):
        with st.spinner("Generating caption..."):
            inputs = processor(image, return_tensors="pt")
            with torch.no_grad():
                out = model.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)
            st.success("Caption:")
            st.write(f"> {caption}")

    stquery_against_image=st.sidebar.text_input(label="Post your query against your uploaded image",placeholder="Enter your query!")
    button3 = st.sidebar.button("_Submit to AI_",key="b3")
    if button3:
        llm = GoogleGenerativeAI(model="gemini-2.0-flash")
        # AI agent
        tools = load_tools(["serpapi"], llm=llm)
        # AI agent initialization
        agent1 = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handle_parsing_errors = True)
        main_body.text("Processing ......Please wait......✅✅✅")
        st.write(agent1.run(caption+"."+stquery_against_image))






