import streamlit as st
import os
import logging
from pathlib import Path
import tempfile
import tkinter as tk
from tkinter import filedialog
import moviepy.editor as mp
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence
import srt
from datetime import timedelta
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.font_manager as fm
from googletrans import Translator
from langdetect import detect

def select_directory():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    root.attributes('-topmost', True)  # Bring the dialog to the front
    directory = filedialog.askdirectory()
    root.destroy()
    return directory

def extract_audio_from_video(video_path, audio_path):
    st.text(f"Extracting audio from {video_path}")
    video = mp.VideoFileClip(str(video_path))
    video.audio.write_audiofile(str(audio_path))
    duration = video.duration
    video.close()
    return duration

def translate_text(text, dest_language='sw'):
    translator = Translator()
    translated = translator.translate(text, dest=dest_language)
    return translated.text

def create_srt_subtitles(transcriptions, output_srt, translate=True, dest_language='sw'):
    st.text(f"Creating SRT subtitles: {output_srt}")
    subs = []
    for i, (start, end, text) in enumerate(transcriptions, start=1):
        if translate:
            text = translate_text(text, dest_language)
        sub = srt.Subtitle(
            index=i,
            start=timedelta(seconds=start),
            end=timedelta(seconds=end),
            content=text
        )
        subs.append(sub)

    with open(output_srt, 'w', encoding='utf-8') as f:
        f.write(srt.compose(subs))

def find_font(font_type):
    system_fonts = fm.findSystemFonts(fontpaths=None, fontext='ttf')
    if font_type == 'arabic':
        fonts = [f for f in system_fonts if 'arabic' in f.lower() or 'noto' in f.lower()]
    elif font_type == 'sans':
        fonts = [f for f in system_fonts if 'sans' in f.lower()]
    else:
        fonts = []
    return fonts[0] if fonts else None

def create_subtitle_clip(text, video_size, font_size=37, font_color="white", bg_color=(0, 0, 0, 150)):
    font_path = find_font('sans')
    if font_path is None:
        logging.warning("No suitable font found. Using default font.")
        font = ImageFont.load_default()
    else:
        font = ImageFont.truetype(font_path, font_size)

    img = Image.new('RGBA', video_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    max_width = int(video_size[0] * 0.9)
    lines = []
    words = text.split()
    current_line = words[0]

    for word in words[1:]:
        test_line = current_line + " " + word
        bbox = draw.textbbox((0, 0), test_line, font=font)
        test_width = bbox[2] - bbox[0]
        if test_width <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word
    lines.append(current_line)

    bbox = draw.textbbox((0, 0), 'Tg', font=font)
    line_height = bbox[3] - bbox[1]
    text_height = len(lines) * line_height

    margin = 20
    bg_bbox = [
        0,
        video_size[1] - text_height - 2*margin,
        video_size[0],
        video_size[1]
    ]
    draw.rectangle(bg_bbox, fill=bg_color)

    y_text = video_size[1] - text_height - margin
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        x_text = (video_size[0] - text_width) // 2
        draw.text((x_text, y_text), line, font=font, fill=font_color)
        y_text += line_height

    return mp.ImageClip(np.array(img))

def add_subtitles_to_video(video_path, srt_path, output_video_path):
    st.text(f"Adding subtitles to video: {video_path}")
    video = mp.VideoFileClip(str(video_path))

    with open(srt_path, 'r', encoding='utf-8') as f:
        subtitles = list(srt.parse(f))

    subtitle_clips = []

    for subtitle in subtitles:
        start_time = subtitle.start.total_seconds()
        end_time = subtitle.end.total_seconds()
        duration = end_time - start_time

        text_clip = (create_subtitle_clip(subtitle.content, video.size)
                     .set_duration(duration)
                     .set_start(start_time))

        subtitle_clips.append(text_clip)

    final_video = mp.CompositeVideoClip([video] + subtitle_clips)
    final_video.write_videofile(str(output_video_path))
    video.close()

def detect_language(text):
    try:
        return detect(text)
    except:
        logging.warning("Could not detect language. Defaulting to 'en'")
        return 'en'

def transcribe_audio(audio_path, src_language=None):
    st.text(f"Transcribing audio from {audio_path}")
    sound = AudioSegment.from_wav(audio_path)
    chunks = split_on_silence(sound, min_silence_len=700, silence_thresh=sound.dBFS-14, keep_silence=500)

    folder_name = Path("audio-chunks")
    folder_name.mkdir(exist_ok=True)

    transcriptions = []
    r = sr.Recognizer()

    current_time = 0
    for i, audio_chunk in enumerate(chunks, start=1):
        chunk_filename = folder_name / f"chunk{i}.wav"
        audio_chunk.export(str(chunk_filename), format="wav")

        chunk_duration = len(audio_chunk) / 1000  # Convert to seconds

        with sr.AudioFile(str(chunk_filename)) as source:
            audio_listened = r.record(source)
            try:
                if src_language:
                    text = r.recognize_google(audio_listened, language=src_language)
                else:
                    text = r.recognize_google(audio_listened)
                transcriptions.append((current_time, current_time + chunk_duration, text))
            except sr.UnknownValueError as e:
                st.error(f"Error in chunk {i}: {str(e)}")

        current_time += chunk_duration
        os.remove(chunk_filename)  # Clean up temporary files

    return transcriptions

def select_output_directory():
    root = Tk()
    root.withdraw()
    folder_selected = filedialog.askdirectory()
    return folder_selected

def process_video(input_video, output_dir, src_language, dest_language, progress_bar):
    temp_audio_path = Path(output_dir) / 'extracted_audio.wav'
    srt_path = Path(output_dir) / f'subtitles_{dest_language}.srt'
    output_video_path = Path(output_dir) / f'video_with_subtitles_{dest_language}.mp4'

    os.makedirs(output_dir, exist_ok=True)

    try:
        progress_bar.progress(0)
        st.text("Extracting audio from video...")
        video_duration = extract_audio_from_video(input_video, temp_audio_path)
        progress_bar.progress(20)

        st.text("Transcribing audio...")
        if src_language != "auto":
            st.text(f"Using specified source language: {src_language}")
            transcriptions = transcribe_audio(str(temp_audio_path), src_language=src_language)
            detected_language = src_language
        else:
            transcriptions = transcribe_audio(str(temp_audio_path))
            if transcriptions:
                detected_language = detect_language(transcriptions[0][2])
                st.text(f"Detected source language: {detected_language}")
            else:
                detected_language = 'en'
                st.warning("No transcriptions available. Defaulting to English as source language.")
        progress_bar.progress(60)

        st.text("Creating subtitles...")
        create_srt_subtitles(transcriptions, srt_path, translate=True, dest_language=dest_language)
        progress_bar.progress(80)

        st.text("Adding subtitles to video...")
        add_subtitles_to_video(input_video, srt_path, output_video_path)
        progress_bar.progress(100)

        return output_video_path

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None
    finally:
        # Clean up temporary audio file
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

def main():
    st.title("Video Subtitle Processor")

    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            input_video_path = tmp_file.name

        src_language = st.selectbox(
            "Select Source Language",
            ["auto", "en", "ar"],
            format_func=lambda x: "Auto Detect (less Accurate)" if x == "auto" else {"en": "English", "ar": "Arabic"}[x]
        )

        dest_language = st.selectbox(
            "Select Destination Language",
            ["en", "ar", "sw", "fr"],
            format_func=lambda x: {"en": "English", "ar": "Arabic", "sw": "Swahili", "fr": "French"}[x]
        )
        # Initialize session state
        if 'output_dir' not in st.session_state:
            st.session_state.output_dir = "output"

        # Create a button to open the directory selector
        if st.button("Select Output Directory"):
            chosen_dir = select_directory()
            if chosen_dir:
                st.session_state.output_dir = chosen_dir

        # Display the selected directory
        output_dir = st.text_input("Output directory path", value=st.session_state.output_dir)
        # output_dir = st.text_input("Enter output directory path", value="output")
        # Use the output_dir variable in your app
        st.write(f"Selected output directory: {output_dir}")

        if st.button("Process Video"):
            if not output_dir:
                st.warning("Please enter an output directory path before processing.")
            else:
                progress_bar = st.progress(0)
                with st.spinner("Processing video..."):
                    output_video_path = process_video(input_video_path, output_dir, src_language, dest_language, progress_bar)
                
                if output_video_path:
                    st.success("Video processing complete!")
                    st.video(str(output_video_path))
                else:
                    st.error("Video processing failed.")

        # Clean up temporary input video file
        os.unlink(input_video_path)

if __name__ == "__main__":
    main()
