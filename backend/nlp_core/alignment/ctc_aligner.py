"""
CTC-based Forced Alignment.

Core CTC alignment logic for character-level timestamp extraction.
"""

import logging
from typing import List
import torch
import torchaudio

from .models import get_forced_alignment_model
from .utils import AlignedToken

logger = logging.getLogger(__name__)


def perform_ctc_forced_alignment(
    waveform: torch.Tensor,
    transcript: str,
    sample_rate: int = 16000
) -> List[AlignedToken]:
    """
    Perform CTC-based forced alignment between audio and text.
    
    Uses emission probabilities from the model and dynamic time
    warping to find the optimal alignment path.
    
    Args:
        waveform: Audio waveform tensor
        transcript: Text transcript to align
        sample_rate: Audio sample rate
    
    Returns:
        List of AlignedToken with character-level timestamps
    """
    bundle, model, tokenizer = get_forced_alignment_model()
    
    # Normalize transcript
    transcript = transcript.upper().strip()
    
    with torch.no_grad():
        if bundle is not None:
            return _mms_alignment(
                waveform, transcript, model, tokenizer, sample_rate
            )
        else:
            return _wav2vec_alignment(
                waveform, transcript, model, tokenizer, sample_rate
            )


def _mms_alignment(
    waveform: torch.Tensor,
    transcript: str,
    model,
    tokenizer,
    sample_rate: int
) -> List[AlignedToken]:
    """
    Alignment using MMS_FA pipeline.
    
    MMS_FA tokenizer.dictionary format:
    {'-': 0, 'a': 1, 'i': 2, 'e': 3, ... 'z': 23, ...}
    
    Note: The dictionary maps characters to indices directly.
    Index 0 is '-' (separator). Blank is NOT in the dictionary
    but is handled by torchaudio.functional.forced_align internally.
    """
    try:
        # Get emission probabilities
        emissions, _ = model(waveform)
        
        # MMS_FA uses lowercase characters in its vocabulary
        # Dictionary: {'-': 0, 'a': 1, 'i': 2, ...}
        # CTC dimension: 29 (indices 0-28)
        # IMPORTANT: Index 0 is separator AND CTC blank
        # We cannot have index 0 in targets, so we remove spaces entirely
        transcript_clean = transcript.lower().replace(" ", "")
        
        # Filter out any non-alphabetic characters
        transcript_clean = ''.join(c for c in transcript_clean if c.isalpha())
        
        # Get the dictionary from tokenizer
        if hasattr(tokenizer, 'dictionary'):
            char_to_idx = tokenizer.dictionary
            logger.debug(f"MMS_FA dictionary has {len(char_to_idx)} entries")
        else:
            logger.error("MMS_FA tokenizer has no dictionary attribute")
            return _fallback_uniform_alignment(transcript, waveform, sample_rate)
        
        # Get CTC dimension to validate token indices
        ctc_dim = emissions.shape[2]  # Shape: [batch, frames, vocab_size]
        logger.debug(f"CTC dimension: {ctc_dim}, transcript: '{transcript_clean[:30]}...'")
        
        # Convert characters to token IDs
        # Dictionary indices are 0-28, CTC blank is 0
        # We use dictionary indices directly, but skip index 0 (separator)
        tokens = []
        for char in transcript_clean:
            if char in char_to_idx:
                idx = char_to_idx[char]
                # Index 0 is the CTC blank in forced_align, cannot be used
                if idx == 0:
                    continue  # Skip separator character
                tokens.append(idx)
            # Skip unknown characters silently
        
        # Validate we have tokens
        if not tokens:
            logger.error("No valid tokens generated from transcript")
            return _fallback_uniform_alignment(transcript, waveform, sample_rate)
        
        # Validate token indices are within CTC dimension
        max_token = max(tokens)
        if max_token >= ctc_dim:
            logger.error(f"Token index {max_token} exceeds CTC dim {ctc_dim}")
            return _fallback_uniform_alignment(transcript, waveform, sample_rate)
        
        logger.debug(f"Token indices: {tokens[:10]}... (len={len(tokens)}, min={min(tokens)}, max={max_token})")
        
        # Perform forced alignment
        aligned_tokens, scores = torchaudio.functional.forced_align(
            emissions,
            targets=torch.tensor([tokens], dtype=torch.int32),
            input_lengths=torch.tensor([emissions.shape[1]]),
            target_lengths=torch.tensor([len(tokens)]),
            blank=0
        )
        
        # Convert frame indices to timestamps
        frame_duration = waveform.shape[1] / sample_rate / emissions.shape[1]
        
        results = []
        token_list = list(transcript_clean)
        
        for i, (token_idx, score) in enumerate(zip(aligned_tokens[0], scores[0])):
            if i < len(token_list):
                char = token_list[i]
                start_frame = token_idx.item()
                
                if i + 1 < len(aligned_tokens[0]):
                    end_frame = aligned_tokens[0][i + 1].item()
                else:
                    end_frame = emissions.shape[1]
                
                results.append(AlignedToken(
                    token=char,
                    start=round(start_frame * frame_duration, 3),
                    end=round(end_frame * frame_duration, 3),
                    score=score.item()
                ))
        
        logger.info(f"MMS_FA alignment successful for {len(results)} tokens")
        return results
        
    except Exception as e:
        logger.error(f"MMS_FA alignment failed: {str(e)}")
        # Fall back to simple uniform distribution
        return _fallback_uniform_alignment(transcript, waveform, sample_rate)


def _wav2vec_alignment(
    waveform: torch.Tensor,
    transcript: str,
    model,
    processor,
    sample_rate: int
) -> List[AlignedToken]:
    """
    Fallback alignment using wav2vec2 CTC predictions.
    
    Uses a greedy approach: match predicted characters to expected
    text based on emission probabilities and frame positions.
    """
    # Process audio
    input_values = processor(
        waveform.squeeze().numpy(),
        return_tensors="pt",
        sampling_rate=sample_rate
    ).input_values
    
    # Get emissions (log probabilities)
    with torch.no_grad():
        outputs = model(input_values)
        logits = outputs.logits
        log_probs = torch.log_softmax(logits, dim=-1)
    
    # Get predicted character indices
    pred_ids = torch.argmax(log_probs, dim=-1)[0]
    
    # Calculate frame duration
    num_frames = logits.shape[1]
    audio_duration = waveform.shape[1] / sample_rate
    frame_duration = audio_duration / num_frames
    
    # Extract character segments
    char_segments = _extract_char_segments(
        pred_ids, log_probs, processor, frame_duration
    )
    
    # Match to transcript
    transcript_chars = list(transcript.replace(" ", ""))
    aligned = _align_segments_to_transcript(
        char_segments, transcript_chars, frame_duration
    )
    
    return aligned


def _extract_char_segments(
    pred_ids: torch.Tensor,
    log_probs: torch.Tensor,
    processor,
    frame_duration: float
) -> List[dict]:
    """
    Extract character segments with their frame ranges.
    """
    char_segments = []
    current_char_id = None
    start_frame = 0
    num_frames = len(pred_ids)
    
    for frame_idx, char_id in enumerate(pred_ids):
        char_id = char_id.item()
        
        if char_id != current_char_id:
            if current_char_id is not None and current_char_id != 0:
                char = processor.decode([current_char_id])
                if char.strip():
                    mid_frame = (start_frame + frame_idx) // 2
                    prob = log_probs[0, mid_frame, current_char_id].item()
                    
                    char_segments.append({
                        'char': char,
                        'start_frame': start_frame,
                        'end_frame': frame_idx,
                        'prob': prob
                    })
            
            current_char_id = char_id
            start_frame = frame_idx
    
    # Handle last segment
    if current_char_id is not None and current_char_id != 0:
        char = processor.decode([current_char_id])
        if char.strip():
            mid_frame = (start_frame + num_frames) // 2
            prob = log_probs[0, mid_frame, current_char_id].item()
            
            char_segments.append({
                'char': char,
                'start_frame': start_frame,
                'end_frame': num_frames,
                'prob': prob
            })
    
    return char_segments


def _align_segments_to_transcript(
    detected_segments: List[dict],
    transcript_chars: List[str],
    frame_duration: float
) -> List[AlignedToken]:
    """
    Align detected character segments to expected transcript.
    
    Uses greedy matching with fallback interpolation.
    """
    if not detected_segments or not transcript_chars:
        return []
    
    results = []
    seg_idx = 0
    
    last_end = detected_segments[-1]['end_frame']
    total_duration = last_end * frame_duration
    
    for char_idx, expected_char in enumerate(transcript_chars):
        expected_upper = expected_char.upper()
        matched = False
        
        # Look for matching segment
        search_end = min(seg_idx + 5, len(detected_segments))
        
        for i in range(seg_idx, search_end):
            seg = detected_segments[i]
            
            if seg['char'].upper() == expected_upper:
                start = seg['start_frame'] * frame_duration
                end = seg['end_frame'] * frame_duration
                score = min(1.0, max(0.0, 0.5 + seg['prob']))
                
                results.append(AlignedToken(
                    token=expected_char,
                    start=round(start, 3),
                    end=round(end, 3),
                    score=score
                ))
                seg_idx = i + 1
                matched = True
                break
        
        if not matched:
            # Interpolate position
            progress = char_idx / len(transcript_chars)
            char_duration = total_duration / len(transcript_chars)
            
            results.append(AlignedToken(
                token=expected_char,
                start=round(progress * total_duration, 3),
                end=round(progress * total_duration + char_duration, 3),
                score=0.5
            ))
    
    return results


def _fallback_uniform_alignment(
    transcript: str,
    waveform: torch.Tensor,
    sample_rate: int
) -> List[AlignedToken]:
    """
    Fallback to uniform distribution when alignment fails.
    
    Distributes time evenly across all characters in the transcript.
    Consistent with MMS_FA: removes spaces and non-alphabetic characters.
    """
    logger.warning("Using uniform fallback alignment")
    
    # Match MMS_FA preprocessing: lowercase, remove spaces and punctuation
    transcript_clean = transcript.lower().replace(" ", "")
    transcript_clean = ''.join(c for c in transcript_clean if c.isalpha())
    chars = list(transcript_clean)
    
    if not chars:
        return []
    
    audio_duration = waveform.shape[1] / sample_rate
    char_duration = audio_duration / len(chars)
    
    results = []
    for i, char in enumerate(chars):
        results.append(AlignedToken(
            token=char,
            start=round(i * char_duration, 3),
            end=round((i + 1) * char_duration, 3),
            score=0.5
        ))
    
    return results

