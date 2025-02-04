#%%
from openai import OpenAI
from dotenv import load_dotenv
import os
import cv2  # OpenCV
import base64
from PIL import Image
import numpy as np
import io
import matplotlib.pyplot as plt

# Load the .env file and get the API key
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# Path to your local video file
video_path = '/home/ubuntu/virtual_folder/audiocraft/images_detector/videos/casa_patio.mp4'

# Function to process a video frame
def process_frame(frame):
    # Convert from BGR (OpenCV) to RGB (PIL)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Create a PIL image from the array
    img = Image.fromarray(rgb_frame)
    # Resize the image to smaller dimensions (adjust as needed)
    img = img.resize((100, 50))
    # Convert the PIL image back to a NumPy array
    img_array = np.array(img)
    # Compress the image to PNG format
    success, buffer = cv2.imencode(".png", img_array)
    if not success:
        raise ValueError("Failed to encode frame to PNG format.")
    # Encode the PNG image to a base64 string
    encoded_string = base64.b64encode(buffer).decode('utf-8')
    return encoded_string

# Open the video file using OpenCV
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError(f"Cannot open video file: {video_path}")

# Calculate video duration in seconds using frame count and FPS
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
video_duration = frame_count / fps
print(f"Video duration: {video_duration:.2f} seconds")

# Define the time interval (in seconds) between frames to compare
step_interval = 3

# Prepare a list of starting times (in seconds) for each step
steps = list(range(0, int(video_duration - step_interval) + 1, step_interval))
print(f"Processing steps at times: {steps}")

# Create an output directory for the plots if it doesn't exist
output_dir = "output_plots"
os.makedirs(output_dir, exist_ok=True)

# Process each frame pair from the video using the base code logic
for t in steps:
    # Seek to the previous frame at time t seconds
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ret, frame_prev = cap.read()
    if not ret:
        print(f"Failed to read frame at {t} seconds.")
        continue

    # Seek to the current frame at time t+step_interval seconds
    cap.set(cv2.CAP_PROP_POS_MSEC, (t + step_interval) * 1000)
    ret, frame_curr = cap.read()
    if not ret:
        print(f"Failed to read frame at {t + step_interval} seconds.")
        continue

    # Process both frames to get base64-encoded images
    try:
        base64_frame_prev = process_frame(frame_prev)
        base64_frame_curr = process_frame(frame_curr)
        print(f"Frames at {t} sec and {t + step_interval} sec processed successfully.")
    except Exception as e:
        print(f"Error processing frames at {t} sec and {t + step_interval} sec: {e}")
        continue

    # Define the prompt message for this pair of frames
    prompt_message = [
        (
            f"Decide if the following frames from the video are in the same place/context. "
            "For example, if one frame is taken indoors and the other outdoors, or if they are from "
            "different locations, respond with either 'different' or 'not different'. If you consider "
            "the frames are different but in the same context, classify them as 'not different'."
        ),
        {"image": base64_frame_prev, "resize": 768},
        {"image": base64_frame_curr, "resize": 768},
    ]
    PROMPT_MESSAGES = [{"role": "user", "content": prompt_message}]

    # Define parameters for the OpenAI API call
    params = {
        "model": "gpt-4o",  # Replace with your actual model name if needed
        "messages": PROMPT_MESSAGES,
        "max_tokens": 500,
    }

    # Call the API to compare the two frames
    try:
        result = client.chat.completions.create(**params)
        response_text = result.choices[0].message.content
        print(f"API response for frames at {t} sec and {t + step_interval} sec:", response_text)
    except Exception as e:
        print(f"API call failed at {t} sec: {e}")
        response_text = "API call failed."

    # Plot the two frames side by side with the API's response below them
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Convert the base64 strings back to images for plotting
    img_prev = Image.open(io.BytesIO(base64.b64decode(base64_frame_prev.encode('utf-8'))))
    img_curr = Image.open(io.BytesIO(base64.b64decode(base64_frame_curr.encode('utf-8'))))

    ax[0].imshow(img_prev)
    ax[0].axis('off')
    ax[0].set_title(f"Frame at {t} sec")

    ax[1].imshow(img_curr)
    ax[1].axis('off')
    ax[1].set_title(f"Frame at {t + step_interval} sec")

    # Add the API response text below the images
    plt.figtext(0.5, 0.01, response_text, wrap=True, horizontalalignment='center', fontsize=12)
    plt.subplots_adjust(bottom=0.2)

    # Save the plot to a file instead of showing it
    output_filepath = os.path.join(output_dir, f"frames_{t}_{t+step_interval}.png")
    plt.savefig(output_filepath)
    plt.close(fig)
    print(f"Saved plot to {output_filepath}")

# ---- Additional: Compare the Final Two Frames ----
# Sometimes the final pair of frames is not captured by the steps.
# We'll compare the penultimate frame and the final frame.
final_time = video_duration  # Time (in seconds) for the final frame
penultimate_time = max(video_duration - step_interval, 0)  # Ensure non-negative

print(f"Comparing final two frames at {penultimate_time:.2f} sec and {final_time:.2f} sec.")

# Seek to the penultimate frame
cap.set(cv2.CAP_PROP_POS_MSEC, penultimate_time * 1000)
ret, frame_penultimate = cap.read()
if not ret:
    print(f"Failed to read penultimate frame at {penultimate_time} seconds.")
else:
    # Seek to the final frame
    cap.set(cv2.CAP_PROP_POS_MSEC, final_time * 1000)
    ret, frame_final = cap.read()
    if not ret:
        print(f"Failed to read final frame at {final_time} seconds.")
    else:
        try:
            base64_frame_penultimate = process_frame(frame_penultimate)
            base64_frame_final = process_frame(frame_final)
            print("Final two frames processed successfully.")
        except Exception as e:
            print(f"Error processing final two frames: {e}")

        # Define the prompt message for the final two frames
        prompt_message = [
            (
                f"Decide if the following final frames from the video are in the same place/context. "
                "For example, if one frame is taken indoors and the other outdoors, or if they are from "
                "different locations, respond with either 'different' or 'not different'. If you consider "
                "the frames are different but in the same context, classify them as 'not different'."
            ),
            {"image": base64_frame_penultimate, "resize": 768},
            {"image": base64_frame_final, "resize": 768},
        ]
        PROMPT_MESSAGES = [{"role": "user", "content": prompt_message}]

        params = {
            "model": "gpt-4o",  # Replace with your actual model name if needed
            "messages": PROMPT_MESSAGES,
            "max_tokens": 500,
        }

        try:
            result = client.chat.completions.create(**params)
            response_text = result.choices[0].message.content
            print(f"API response for final two frames:", response_text)
        except Exception as e:
            print(f"API call failed for final two frames: {e}")
            response_text = "API call failed."

        # Plot and save the comparison for the final two frames
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        img_penultimate = Image.open(io.BytesIO(base64.b64decode(base64_frame_penultimate.encode('utf-8'))))
        img_final = Image.open(io.BytesIO(base64.b64decode(base64_frame_final.encode('utf-8'))))
        ax[0].imshow(img_penultimate)
        ax[0].axis('off')
        ax[0].set_title(f"Frame at {penultimate_time:.2f} sec")
        ax[1].imshow(img_final)
        ax[1].axis('off')
        ax[1].set_title(f"Frame at {final_time:.2f} sec")
        plt.figtext(0.5, 0.01, response_text, wrap=True, horizontalalignment='center', fontsize=12)
        plt.subplots_adjust(bottom=0.2)
        output_filepath = os.path.join(output_dir, f"frames_{penultimate_time:.2f}_{final_time:.2f}_final.png")
        plt.savefig(output_filepath)
        plt.close(fig)
        print(f"Saved final frames plot to {output_filepath}")

# Release the video capture when done
cap.release()
# %%
