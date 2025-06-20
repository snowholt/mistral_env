import os
import logging
from beautyai_inference.config.config_manager import AppConfig
from beautyai_inference.services.model.lifecycle_service import ModelLifecycleService
from beautyai_inference.inference_engines.transformers_engine import TransformersEngine
from transformers import WhisperProcessor, WhisperForConditionalGeneration

def test_whisper_large_v3_turbo_arabic():
    """Test the Whisper Large V3 Turbo Arabic model with a simple MP3 file."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Load the model configuration
    app_config = AppConfig()
    app_config.models_file = "beautyai_inference/config/model_registry.json"
    app_config.load_model_registry()

    model_config = app_config.model_registry.get_model("whisper-large-v3-turbo-arabic")

    if not model_config:
        logger.error("Model configuration for 'whisper-large-v3-turbo-arabic' not found.")
        return

    # Initialize the lifecycle service
    lifecycle_service = ModelLifecycleService()

    # Load the model
    success, error_msg = lifecycle_service.load_model(model_config)

    if not success:
        logger.error(f"Failed to load model: {error_msg}")
        return

    logger.info("Model loaded successfully.")

    # Test the model with a user-specified audio file
    audio_file_path = input("Enter the path to the audio file (MP3, OGG, WAV): ")

    if not os.path.exists(audio_file_path):
        logger.error(f"Audio file not found: {audio_file_path}")
        return

    # Initialize the engine
    engine = TransformersEngine(model_config)
    engine.load_model()

    # Ensure the tokenizer has a chat template
    if not hasattr(engine.tokenizer, 'chat_template') or engine.tokenizer.chat_template is None:
        logger.info("Setting default chat template for the tokenizer.")
        engine.tokenizer.chat_template = "<s>[INST] {input} [/INST]"

    # Load the processor and model
    processor = WhisperProcessor.from_pretrained(model_config.model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_config.model_id)

    # Preprocess the audio file
    import torchaudio

    try:
        waveform, sample_rate = torchaudio.load(audio_file_path)
        logger.info(f"Audio file loaded: {audio_file_path}, Sample rate: {sample_rate}")

        # Prepare input features
        input_features = processor(waveform.squeeze().numpy(), sampling_rate=sample_rate, return_tensors="pt").input_features

        # Perform transcription
        predicted_ids = model.generate(input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        logger.info(f"Transcription: {transcription}")
    except Exception as e:
        logger.error(f"Error during transcription: {e}")

if __name__ == "__main__":
    test_whisper_large_v3_turbo_arabic()
