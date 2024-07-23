import subprocess
import deepspeech
import numpy as np
import wave
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
from pydub import AudioSegment
import os
from concurrent.futures import ThreadPoolExecutor

class SubtitleGenerator:
    def __init__(self, model_path, scorer_path):
        self.model = deepspeech.Model(model_path)
        self.model.enableExternalScorer(scorer_path)
        self._create_directories()
        self.max_workers = 4  # 병렬 작업을 위한 최대 워커 수

    def _create_directories(self):
        if not os.path.exists('/tmp'):
            os.makedirs('/tmp')
        if not os.path.exists('outputs'):
            os.makedirs('outputs')

    def generate_subtitles(self, video_path, output_video_path):
        video = VideoFileClip(video_path)
        audio_path = "/tmp/temp_audio.wav"
        video.audio.write_audiofile(audio_path)

        audio = AudioSegment.from_wav(audio_path)
        segments = self.split_audio(audio, 60000)  # 60초씩 분할

        subtitles = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for i, segment in enumerate(segments):
                segment_path = f"/tmp/temp_segment_{i}.wav"
                segment.export(segment_path, format="wav")
                futures.append(executor.submit(self.process_segment, segment_path, segment.start_second, segment.end_second))
            
            for future in futures:
                subtitles.append(future.result())

        os.remove(audio_path)
        final_video = self.add_subtitles_to_video(video, subtitles)
        final_video.write_videofile(output_video_path, codec='libx264', audio_codec='aac')

    def split_audio(self, audio, chunk_length_ms):
        chunks = []
        for i in range(0, len(audio), chunk_length_ms):
            chunk = audio[i:i + chunk_length_ms]
            chunk.start_second = i / 1000
            chunk.end_second = (i + chunk_length_ms) / 1000
            chunks.append(chunk)
        return chunks

    def process_segment(self, segment_path, start_time, end_time):
        text = self.recognize_speech(segment_path)
        os.remove(segment_path)
        return (start_time, end_time, text)

    def recognize_speech(self, audio_path):
        with wave.open(audio_path, 'r') as w:
            rate = w.getframerate()
            frames = w.getnframes()
            buffer = w.readframes(frames)
            data16 = np.frombuffer(buffer, dtype=np.int16)
            text = self.model.stt(data16)
        return text

    def add_subtitles_to_video(self, video, subtitles):
        clips = [video]

        for start_time, end_time, text in subtitles:
            if text:
                subtitle_clip = TextClip(
                    text, fontsize=24, color='white', bg_color='black', size=(video.w, 50), method='caption'
                ).set_position(('center', 'bottom'), relative=True).set_start(start_time).set_duration(end_time - start_time)
                clips.append(subtitle_clip)

        final_video = CompositeVideoClip(clips)
        return final_video
