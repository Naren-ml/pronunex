"""
Model Loading for Forced Alignment.

Handles singleton model loading for alignment models.
"""

import logging
import torch
import torchaudio

logger = logging.getLogger(__name__)

# Singleton model instances (expensive to load)
_aligner_bundle = None
_aligner_model = None
_aligner_tokenizer = None


def get_forced_alignment_model():
    """
    Load the forced alignment model using torchaudio's MMS_FA bundle.
    
    MMS_FA (Massively Multilingual Speech Forced Alignment) provides
    accurate frame-level alignment for speech.
    
    Falls back to wav2vec2-base-960h for older torchaudio versions.
    """
    global _aligner_bundle, _aligner_model, _aligner_tokenizer
    
    if _aligner_model is None:
        logger.info("Loading forced alignment model...")
        
        try:
            # Try MMS_FA bundle (torchaudio >= 2.1)
            _aligner_bundle = torchaudio.pipelines.MMS_FA
            _aligner_model = _aligner_bundle.get_model()
            _aligner_tokenizer = _aligner_bundle.get_tokenizer()
            _aligner_model.eval()
            logger.info("MMS_FA forced alignment model loaded")
            
        except AttributeError:
            # Fallback: wav2vec2 base model for older versions
            logger.warning("MMS_FA unavailable, using wav2vec2 fallback")
            _aligner_bundle = None
            
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
            
            _aligner_model = Wav2Vec2ForCTC.from_pretrained(
                "facebook/wav2vec2-base-960h"
            )
            _aligner_tokenizer = Wav2Vec2Processor.from_pretrained(
                "facebook/wav2vec2-base-960h"
            )
            _aligner_model.eval()
            logger.info("Wav2Vec2 fallback model loaded")
    
    return _aligner_bundle, _aligner_model, _aligner_tokenizer


def is_mms_available() -> bool:
    """Check if MMS_FA bundle is being used."""
    bundle, _, _ = get_forced_alignment_model()
    return bundle is not None
