from silent_remover import SilentRemover
from subtitle_generator import SubtitleGenerator

def main():
    video_path = "inputs/input_video.mp4"
    processed_video_path = "outputs/processed_video.mp4"
    output_video_path = "outputs/output_video_with_subtitles.mp4"

    # DeepSpeech 모델 경로 설정
    model_path = 'deepspeech-0.9.3-models.pbmm'
    scorer_path = 'deepspeech-0.9.3-models.scorer'

    silent_remover = SilentRemover(silence_threshold=-50.0, min_silence_len=0.5)
    try:
        processed_video_path = silent_remover.remove_silent_parts(video_path, processed_video_path)
        print(f"Processed video saved to {processed_video_path}")

        subtitle_generator = SubtitleGenerator(model_path, scorer_path)
        subtitle_generator.generate_subtitles(processed_video_path, output_video_path)
        print(f"Subtitled video saved to {output_video_path}")

    except ValueError as e:
        print(e)

if __name__ == "__main__":
    main()
