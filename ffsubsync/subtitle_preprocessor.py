"""Subtitle preprocessing for improved alignment accuracy."""

import re
import logging
from typing import List, Optional
from ffsubsync.generic_subtitles import GenericSubtitle

logger = logging.getLogger(__name__)

# Patterns for non-dialogue content
NON_DIALOGUE_PATTERNS = [
    # Music symbols
    r'^[♪♫🎵🎶\s]+$',
    # Bracketed content (sound effects, descriptions)
    r'^\s*[\[\(（【][^\]\)）】]*[\]\)）】]\s*$',
    # Italic/styled sound effects like <i>[music]</i>
    r'^\s*<[^>]+>\s*[\[\(（【][^\]\)）】]*[\]\)）】]\s*</[^>]+>\s*$',
    # Pure ellipsis or dashes
    r'^[\.\-\s…—]+$',
    # Single repeated characters
    r'^(.)\1{2,}$',
]

# Compiled patterns for efficiency
_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE | re.UNICODE) for p in NON_DIALOGUE_PATTERNS]

# Keywords that indicate non-dialogue (case-insensitive)
NON_DIALOGUE_KEYWORDS = [
    '♪', '♫', '🎵', '🎶',
    '[music]', '[singing]', '[song]',
    '[applause]', '[laughter]', '[cheering]',
    '[gunshot]', '[explosion]', '[thunder]',
    '[silence]', '[inaudible]',
    '(music)', '(singing)', '(song)',
    '(applause)', '(laughter)', '(cheering)',
    # Chinese equivalents
    '[音乐]', '[掌声]', '[笑声]', '[枪声]', '[爆炸声]',
    '（音乐）', '（掌声）', '（笑声）',
]


def is_non_dialogue(content: str) -> bool:
    """Check if subtitle content is non-dialogue (sound effect, music, etc.)."""
    content = content.strip()
    
    if not content:
        return True
    
    # Check patterns
    for pattern in _COMPILED_PATTERNS:
        if pattern.match(content):
            return True
    
    # Check keywords
    content_lower = content.lower()
    for keyword in NON_DIALOGUE_KEYWORDS:
        if keyword.lower() in content_lower:
            # Only if the keyword is a significant part of the content
            if len(keyword) > len(content) * 0.5:
                return True
            # Or if content is mostly the keyword
            if content_lower.strip('[]()（）【】<>/').strip() == keyword.lower().strip('[]()（）【】'):
                return True
    
    return False


def merge_short_subtitles(
    subs: List[GenericSubtitle],
    min_duration: float = 0.3,
    max_gap: float = 0.3
) -> List[GenericSubtitle]:
    """Merge subtitles that are too short or have small gaps.
    
    Args:
        subs: List of subtitles
        min_duration: Minimum duration in seconds for a subtitle
        max_gap: Maximum gap in seconds between subtitles to merge
    
    Returns:
        List of merged subtitles
    """
    if not subs:
        return subs
    
    merged = []
    current = None
    
    for sub in subs:
        duration = sub.end.total_seconds() - sub.start.total_seconds()
        
        if current is None:
            current = sub
            continue
        
        gap = sub.start.total_seconds() - current.end.total_seconds()
        current_duration = current.end.total_seconds() - current.start.total_seconds()
        
        # Merge if gap is small and either subtitle is short
        if gap <= max_gap and (current_duration < min_duration or duration < min_duration):
            # Extend current subtitle
            from datetime import timedelta
            new_end = sub.end
            new_content = current.content + " " + sub.content
            # Create new subtitle with extended range
            current = GenericSubtitle(
                current.index,
                current.start,
                new_end,
                new_content
            )
        else:
            merged.append(current)
            current = sub
    
    if current is not None:
        merged.append(current)
    
    return merged


def preprocess_subtitles(
    subs: List[GenericSubtitle],
    filter_non_dialogue: bool = True,
    merge_short: bool = True,
    min_duration: float = 0.3,
    max_gap: float = 0.3,
    min_keep_ratio: float = 0.3
) -> List[GenericSubtitle]:
    """Preprocess subtitles for improved alignment.
    
    Args:
        subs: List of subtitles
        filter_non_dialogue: Whether to filter out non-dialogue content
        merge_short: Whether to merge short subtitles
        min_duration: Minimum duration for short subtitle merging
        max_gap: Maximum gap for short subtitle merging
        min_keep_ratio: Minimum ratio of subtitles to keep (safety check)
    
    Returns:
        Preprocessed list of subtitles
    """
    if not subs:
        return subs
    
    original_count = len(subs)
    processed = list(subs)
    
    # Step 1: Filter non-dialogue content
    if filter_non_dialogue:
        processed = [s for s in processed if not is_non_dialogue(s.content)]
        filtered_count = original_count - len(processed)
        if filtered_count > 0:
            logger.info("Filtered %d non-dialogue subtitles", filtered_count)
    
    # Safety check: if too many filtered, use original
    if len(processed) < original_count * min_keep_ratio:
        logger.warning(
            "Too many subtitles filtered (%d/%d), using original subtitles",
            original_count - len(processed), original_count
        )
        return list(subs)
    
    # Step 2: Merge short subtitles
    if merge_short and len(processed) > 1:
        before_merge = len(processed)
        processed = merge_short_subtitles(processed, min_duration, max_gap)
        merged_count = before_merge - len(processed)
        if merged_count > 0:
            logger.info("Merged %d short subtitle segments", merged_count)
    
    return processed
