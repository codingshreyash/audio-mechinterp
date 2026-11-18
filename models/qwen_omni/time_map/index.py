"""Source-matched row indexing for pinned Qwen2.5-Omni audio encoder.

These are tensor/chunk indices, not forced word boundaries. The formulas mirror
Transformers commit e7e568dc7673ac99a87660c7c50e84830268285c in
modeling_qwen2_5_omni.py: chunk_and_pad_features, get_audio_cu_seqlens,
get_pool_indices, and Qwen2_5OmniAudioEncoder.forward.
"""

from __future__ import annotations


def build_row_index(
    *, mel_frames: int, audio_token_positions: list[int], n_window: int,
    hop_length: int, sample_rate: int,
) -> dict:
    if mel_frames < 2:
        raise ValueError("At least two valid mel frames are needed")
    if n_window <= 0 or hop_length <= 0 or sample_rate <= 0:
        raise ValueError("Window, hop, and sample rate must be positive")
    chunk_size = 2 * n_window
    chunks: list[dict] = []
    encoder_rows: list[dict] = []
    offset = 0
    encoder_offset = 0
    while offset < mel_frames:
        length = min(chunk_size, mel_frames - offset)
        encoded_length = (length - 1) // 2 + 1
        chunk_id = len(chunks)
        chunks.append({
            "chunk_index": chunk_id,
            "mel_frame_start": offset,
            "mel_frame_end_exclusive": offset + length,
            "encoder_row_start": encoder_offset,
            "encoder_row_end_exclusive": encoder_offset + encoded_length,
            "nominal_start_seconds": offset * hop_length / sample_rate,
            "nominal_end_seconds": (offset + length) * hop_length / sample_rate,
        })
        for local_row in range(encoded_length):
            nominal_center = offset + 2 * local_row
            encoder_rows.append({
                "row_index": encoder_offset + local_row,
                "chunk_index": chunk_id,
                "row_in_chunk": local_row,
                "nominal_center_mel_frame": nominal_center,
                "nominal_center_seconds": nominal_center * hop_length / sample_rate,
                "pre_attention_conv_receptive_mel_start": max(offset, nominal_center - 2),
                "pre_attention_conv_receptive_mel_end_exclusive": min(offset + length, nominal_center + 3),
                "attention_scope_mel_start": offset,
                "attention_scope_mel_end_exclusive": offset + length,
            })
        offset += length
        encoder_offset += encoded_length

    # Pinned get_pool_indices uses the full single-audio conv length after
    # concatenating per-chunk valid rows. Full chunks have even conv length.
    full_conv_length = (mel_frames - 1) // 2 + 1
    if len(encoder_rows) != full_conv_length:
        raise ValueError("Per-chunk conv rows do not match source full length")
    pooled_count = (full_conv_length - 2) // 2 + 1
    if pooled_count != len(audio_token_positions):
        raise ValueError(
            f"Projected rows {pooled_count} != processor audio placeholders {len(audio_token_positions)}"
        )
    projected_rows = []
    for row in range(pooled_count):
        a, b = 2 * row, 2 * row + 1
        first, second = encoder_rows[a], encoder_rows[b]
        if first["chunk_index"] != second["chunk_index"]:
            raise ValueError("Pooling crosses an attention chunk; source mapping needs review")
        chunk = chunks[first["chunk_index"]]
        projected_rows.append({
            "row_index": row,
            "audio_placeholder_token_position": int(audio_token_positions[row]),
            "encoder_row_pair": [a, b],
            "chunk_index": first["chunk_index"],
            "nominal_center_mel_frame": (first["nominal_center_mel_frame"] + second["nominal_center_mel_frame"]) / 2,
            "attention_scope_mel_start": chunk["mel_frame_start"],
            "attention_scope_mel_end_exclusive": chunk["mel_frame_end_exclusive"],
        })
    return {
        "mel_frames": mel_frames,
        "mel_hop_samples": hop_length,
        "sample_rate": sample_rate,
        "nominal_mel_frame_period_seconds": hop_length / sample_rate,
        "n_window": n_window,
        "mel_frames_per_full_chunk": chunk_size,
        "encoder_row_count": len(encoder_rows),
        "projected_row_count": len(projected_rows),
        "chunks": chunks,
        "encoder_rows": encoder_rows,
        "projected_rows": projected_rows,
        "timing_warning": "Rows are model indices, not word timestamps. The conv receptive span precedes self-attention; each post-layer encoder row can depend on its entire chunk. Nominal seconds use mel hop only and are not independent speech alignment.",
    }
