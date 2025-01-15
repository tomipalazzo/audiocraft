
 # %%

import librosa
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import numpy as np
from scipy.io.wavfile import write

# Load the MusicGen model and processor
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

# Set up parameters
text_prompt = "A dramatic orchestral piece."  # Your text prompt
chunk_duration = 30  # Duration of each chunk in seconds
sampling_rate = 32000  # Model's expected sampling rate
iterations = 5  # Number of chunks to generate (total duration = iterations * chunk_duration)

# Start with an initial audio prompt or use an empty array
generated_audio = None

for i in range(iterations):
    # Prepare inputs
    inputs = processor(
        audio=generated_audio if generated_audio is not None else np.zeros((sampling_rate * chunk_duration,)),
        sampling_rate=sampling_rate,
        text=[text_prompt],
        padding=True,
        return_tensors="pt",
    )

    # Generate audio
    output = model.generate(
        **inputs,
        do_sample=True,
        guidance_scale=3,
        max_new_tokens=chunk_duration * sampling_rate // 64,  # Adjust token count for chunk duration
    )

    # Convert to NumPy and append
    new_audio = output[0].cpu().numpy()
    generated_audio = np.concatenate([generated_audio, new_audio]) if generated_audio is not None else new_audio

    print(f"Generated chunk {i + 1}/{iterations}")

# Save the final output
write("unlimited_audio.wav", sampling_rate, (generated_audio * 32767).astype(np.int16))
print("Unlimited audio saved as 'unlimited_audio.wav'")
