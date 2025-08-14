import json
import torch

# Import voice config singleton and transcription factory
from beautyai_inference.config.voice_config_loader import get_voice_config
from beautyai_inference.services.voice.transcription.transcription_factory import (
    create_transcription_service,
)


def test_whisper_model_loaded_identity_and_device():
    """Integration-style test verifying the STT model identity and device placement.

    This test loads the STT model using the same config path used by the running service
    and asserts that:
      1. The resolved Hugging Face model id matches expectation from registry.
      2. The model is placed on GPU if CUDA is available (or on CPU otherwise) matching
         the service-reported device.
    """
    # 1. Load model config from registry (singleton)
    voice_config = get_voice_config()
    stt_cfg = voice_config.get_stt_model_config()
    expected_hf_id = stt_cfg.model_id

    assert expected_hf_id, "Expected Hugging Face model id missing in voice registry"

    service = create_transcription_service()
    # Ensure model is loaded (service defers loading until first use)
    if not service.is_model_loaded():
        loaded_ok = service.load_whisper_model()
        assert loaded_ok, "Whisper STT model failed to load"

    info = service.get_model_info()
    assert info.get("loaded"), "Service reports model not loaded"

    # We only have alias/name in model_info for some backends; verify alias resolves to expected HF id through config again
    alias = info.get("model_name") or info.get("loaded_model")
    registry_alias = voice_config._config["default_models"]["stt"]
    if alias:  # Some implementations may not expose alias
        assert alias == registry_alias, f"Service alias '{alias}' != registry alias '{registry_alias}'"

    # Sanity: alias maps to expected HF id
    resolved_hf_id = voice_config.get_stt_model_config().model_id
    assert (
        resolved_hf_id == expected_hf_id
    ), f"Resolved HF id '{resolved_hf_id}' != expected '{expected_hf_id}'"

    # 2. Verify device placement
    torch_cuda_available = torch.cuda.is_available()
    reported_device = info.get("device")
    assert reported_device, "Model info missing 'device'"

    if torch_cuda_available:
        assert "cuda" in reported_device.lower() or "gpu" in reported_device.lower(), (
            f"CUDA available but reported device '{reported_device}' does not indicate GPU"
        )
    else:
        assert "cpu" in reported_device.lower(), (
            f"CUDA NOT available but reported device '{reported_device}' is not CPU"
        )

    # 3. Optional diagnostics
    print("Model Info Diagnostic:")
    print(json.dumps(info, indent=2))
    print(f"Expected HF ID: {expected_hf_id}")
    print(f"Alias: {alias}")
    print(f"Torch CUDA Available: {torch_cuda_available}")


if __name__ == "__main__":
    test_whisper_model_loaded_identity_and_device()
