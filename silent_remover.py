from moviepy.editor import VideoFileClip, concatenate_videoclips
import numpy as np
import os

class SilentRemover:
    def __init__(self, silence_threshold=-50.0, min_silence_len=0.5):
        self.silence_threshold = silence_threshold
        self.min_silence_len = min_silence_len
        self.supported_formats = ['mp4', 'mov']
        self._create_directories()

    def _is_supported_format(self, file_path):
        ext = os.path.splitext(file_path)[1][1:].lower()
        return ext in self.supported_formats

    def _create_directories(self):
        if not os.path.exists('/tmp'):
            os.makedirs('/tmp')
        if not os.path.exists('outputs'):
            os.makedirs('outputs')

    def remove_silent_parts(self, video_path, output_path):
        if not self._is_supported_format(video_path):
            raise ValueError(f"Unsupported file format: {video_path}. Supported formats are {', '.join(self.supported_formats)}")

        # 비디오 파일 로드
        video = VideoFileClip(video_path)
        
        # 오디오 추출
        audio = video.audio
        sample_rate = audio.fps
        audio_array = audio.to_soundarray(fps=sample_rate)
        
        # 오디오 에너지 계산
        audio_energy = np.sqrt(np.mean(audio_array ** 2, axis=1))
        audio_energy_db = 20 * np.log10(audio_energy + 1e-10)  # log10(0)을 피하기 위해 작은 값을 더함

        # 에너지를 dB로 변환 후, 묵음 구간을 찾기 위해 필터 적용
        silent_mask = audio_energy_db < self.silence_threshold
        silent_chunks = []
        
        # 최소 묵음 길이 기준 설정
        min_silence_samples = int(self.min_silence_len * sample_rate)

        # 묵음 구간 추출
        for i in range(len(silent_mask)):
            if silent_mask[i]:
                if len(silent_chunks) == 0 or i - silent_chunks[-1][-1] > 1:
                    silent_chunks.append([i, i])
                else:
                    silent_chunks[-1][-1] = i
        
        # 최소 묵음 길이 미만의 구간 제거
        silent_chunks = [chunk for chunk in silent_chunks if chunk[1] - chunk[0] >= min_silence_samples]
        
        # 묵음 구간을 기준으로 비디오 클립 자르기
        chunks = []
        start_idx = 0
        for start, end in silent_chunks:
            start_time = start / sample_rate
            end_time = end / sample_rate
            if start_time - start_idx > 0:
                chunks.append(video.subclip(start_idx, start_time))
            start_idx = end_time

        # 마지막 클립 추가
        if start_idx < video.duration:
            chunks.append(video.subclip(start_idx, video.duration))
        
        # 묵음 구간이 제거된 비디오 클립 결합
        final_clip = concatenate_videoclips(chunks)
        
        # 결과 저장
        final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')

        return output_path  # 처리된 비디오 경로 반환
