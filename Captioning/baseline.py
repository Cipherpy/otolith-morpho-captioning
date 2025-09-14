import base64
import os
from PIL import Image
from io import BytesIO
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the ChatOpenAI model
model = ChatOpenAI(model="gpt-4o")

# Path to your local TIFF image
image_path = "/home/cmlre/Desktop/otolith/pterygotrigla hemisticta.tif"

# Convert TIFF to JPEG and encode in base64
def convert_tif_to_base64(image_path):
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=90)
            base64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return base64_str
    except Exception as e:
        print(f"Error processing TIFF image: {e}")
        return None

# Get base64 encoded image
image_data = convert_tif_to_base64(image_path)
if not image_data:
    print("Failed to process image.")
    exit()

# Create a message with the base64-encoded image and an updated prompt
message = HumanMessage(
    content=[
        {
            "type": "text",
            "text": (
                "**Otolith Identification**\n\n"
                "You are provided with an otolith image. Please analyze it step by step"
                "to identify the following features:\n"
                "1. **Notch**\n"
                "2. **Rostrum & Anti-Rostrum**\n"
                "3. **Overall Shape** (elliptical, oval, trapezoidal, etc.)\n"
                "4. **Orientation** (anterior, posterior, dorsal, ventral sides)\n"
                "5. **Excisura** (if present)\n"
                "6. **Sulcus** (distinguish **cauda** and **ostium**)\n\n"
                "Explain **how** each observed feature contributes to the taxonomic identification of the otolith. "
                
                "Provide your reasoning detailing the sequence of morphological clues "
                "you used to narrow down possible species or families.\n\n"
                "**Finally**, offer a higher-level taxonomic identification:\n"
                "• **Kingdom**: \n"
                "• **Phylum**: \n"
                "• **Class**: \n"
                "• **Order**, **Family**, **Genus**, **Species** (as far as can be determined from visible features).\n\n"
                "Make sure your response:\n"
                "• Focuses primarily on the morphological details listed above.\n"
                "• Includes a clear, step-by-step explanation in how you arrive at the identification.\n"
                "• Stays concise but thorough—avoiding irrelevant details.\n"
            ),
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
        },
    ],
)

# Send the message to the model and get the response
response = model.invoke([message])

print("\n🔹 **Otolith Identification (with Chain-of-Thought):**\n")
print(response.content)
