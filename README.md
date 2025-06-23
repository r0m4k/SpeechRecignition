# Speech-to-Text

> End-to-end pipeline that converts long educational videos (demo on **3Blue1Brown’s _“Large Language Models, explained briefly”_**) into well-formatted, timestamped text using the OpenAI Whisper model.

---

## Features

* **Automatic Speech Recognition** powered by `Whisper-large-v3-turbo` via Hugging Face *transformers*.
* **MP4 → MP3 extraction** using `ffmpeg` so you can start from any video.
* **Smart chunking** – splits audio every *N* seconds (default `30 s`) to keep VRAM in check and enable parallel decoding.
* **Timestamped transcripts** saved to `audio_to_text.txt` with neat 80-column wrapping for readability.
* **Simple Python API** (`AudioRecognition` class) and a 5-line `main.py` so you can copy-paste into your own projects.
* **Reproducible example** on a real-world 3Blue1Brown video – perfect as a reference or teaching demo.

---
