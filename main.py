import argparse
import logging
import os
from pathlib import Path

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

def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def extract_audio_from_video(video_path, audio_path):
    logging.info(f"Extracting audio from {video_path}")
    video = mp.VideoFileClip(str(video_path))
    video.audio.write_audiofile(str(audio_path))
    duration = video.duration
    video.close()
    return duration

def transcribe_audio(audio_path, language='ar-AR'):
    logging.info(f"Transcribing audio from {audio_path}")
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
                text = r.recognize_google(audio_listened, language=language)
                transcriptions.append((current_time, current_time + chunk_duration, text))
            except sr.UnknownValueError as e:
                logging.error(f"Error in chunk {i}: {str(e)}")

        current_time += chunk_duration
        os.remove(chunk_filename)  # Clean up temporary files

    return transcriptions

def translate_text(text, dest_language='sw'):
    translator = Translator()
    translated = translator.translate(text, dest=dest_language)
    return translated.text

def create_srt_subtitles(transcriptions, output_srt, translate=True, dest_language='sw'):
    logging.info(f"Creating SRT subtitles: {output_srt}")
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
    logging.info(f"Adding subtitles to video: {video_path}")
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
    logging.info(f"Transcribing audio from {audio_path}")
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
                logging.error(f"Error in chunk {i}: {str(e)}")

        current_time += chunk_duration
        os.remove(chunk_filename)  # Clean up temporary files

    return transcriptions

def main(args):
    setup_logging(args.log_file)

    logging.info("Starting video subtitle processing")

    video_path = Path(args.input_video)
    audio_path = Path(args.output_dir) / 'extracted_audio.wav'
    srt_path = Path(args.output_dir) / f'subtitles_{args.dest_language}.srt'
    output_video_path = Path(args.output_dir) / f'video_with_subtitles_{args.dest_language}.mp4'

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        video_duration = extract_audio_from_video(video_path, audio_path)
        logging.info("Audio extraction complete")

        if args.src_language:
            logging.info(f"Using specified source language: {args.src_language}")
            transcriptions = transcribe_audio(str(audio_path), src_language=args.src_language)
            detected_language = args.src_language
        else:
            transcriptions = transcribe_audio(str(audio_path))
            if transcriptions:
                detected_language = detect_language(transcriptions[0][2])  # Use the first transcription for detection
                logging.info(f"Detected source language: {detected_language}")
            else:
                detected_language = 'en'
                logging.warning("No transcriptions available. Defaulting to English as source language.")

        logging.info("Audio transcription complete")

        create_srt_subtitles(transcriptions, srt_path, translate=True, dest_language=args.dest_language)
        logging.info(f"SRT file created at {srt_path}")

        add_subtitles_to_video(video_path, srt_path, output_video_path)
        logging.info(f"Video with subtitles saved to {output_video_path}")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
    finally:
        # Clean up temporary files
        if os.path.exists(audio_path):
            os.remove(audio_path)
        if os.path.exists('audio-chunks'):
            for file in os.listdir('audio-chunks'):
                os.remove(os.path.join('audio-chunks', file))
            os.rmdir('audio-chunks')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video and add translated subtitles")
    parser.add_argument("--input_video", help="Path to the input video file")
    parser.add_argument("--output_dir", help="Directory to store output files")
    parser.add_argument("--src_language", help="Source language code (e.g., 'en-US', 'fr-FR'). If not provided, language will be auto-detected.")
    parser.add_argument("--dest_language", default="en", help="Destination language code (default: en)")
    parser.add_argument("--log_file", default="video_processor.log", help="Log file path (default: video_processor.log)")
    
    args = parser.parse_args()
    main(args)