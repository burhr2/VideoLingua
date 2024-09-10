# Video Subtitle Processor

## Description

This project is a Streamlit web application that processes video files to add translated subtitles. It allows users to upload a video, select the source and destination languages, and generate a new video with embedded subtitles.

## Features

- Video upload functionality
- Automatic audio extraction from video
- Speech recognition and transcription
- Language detection (optional)
- Translation of subtitles
- Subtitle embedding into the original video
- Progress tracking during processing

## Requirements

- Python 3.11+
- Streamlit
- MoviePy
- SpeechRecognition
- Pydub
- Pillow
- NumPy
- Matplotlib
- googletrans
- langdetect

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/burhr2/VideoLingua.git
   cd VideoLingua
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:
   ```
   streamlit run streamlit_app.py
   ```

2. Open your web browser and go to the URL provided by Streamlit (usually `http://localhost:8501`).

3. Use the web interface to:
   - Upload a video file
   - Select the source language (or use auto-detect)
   - Choose the destination language for translation
   - Specify the output directory
   - Click "Process Video" to start the subtitle generation and embedding process

4. Wait for the processing to complete. The app will display progress and show the final video with subtitles directly in the browser.

## Notes

- Supported video formats: MP4, AVI, MOV
- The app uses Google's speech recognition API for transcription
- Translation is performed using the googletrans library
- Processing time depends on the length of the video and the complexity of the audio

## Contributing

Contributions to improve the Video Subtitle Processor are welcome. Please feel free to submit pull requests or create issues for bugs and feature requests.

## License

This project is licensed under the Apache-2.0 license - see the LICENSE file for details.