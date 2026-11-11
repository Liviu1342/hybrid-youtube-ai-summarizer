"""Tests for YouTube URL parsing."""

import pytest

from utils import extract_video_id, get_transcript, TranscriptError
from youtube_transcript_api import YouTubeTranscriptApi, VideoUnavailable

VIDEO_ID = "dQw4w9WgXcQ"


@pytest.mark.parametrize(
    "url",
    [
        f"https://www.youtube.com/watch?v={VIDEO_ID}",
        f"https://www.youtube.com/watch?v={VIDEO_ID}&t=30s",          # start time
        f"https://www.youtube.com/watch?feature=share&v={VIDEO_ID}",  # v is not first
        f"https://youtu.be/{VIDEO_ID}?si=abc123",                     # share link
        f"https://www.youtube.com/shorts/{VIDEO_ID}",
        f"https://www.youtube.com/embed/{VIDEO_ID}",
        f"https://m.youtube.com/watch?v={VIDEO_ID}",                  # mobile
        f"youtube.com/watch?v={VIDEO_ID}",                            # no https://
        f"  https://youtu.be/{VIDEO_ID}  ",                           # extra spaces
    ],
)
def test_extracts_id_from_valid_urls(url: str) -> None:
    assert extract_video_id(url) == VIDEO_ID


@pytest.mark.parametrize(
    "url",
    [
        f"https://example.com/watch?v={VIDEO_ID}",  # not YouTube
        "https://www.youtube.com/watch?v=short",    # ID too short
        "https://www.youtube.com/",                 # no ID at all
        "not a link",
        "",
    ],
)
def test_returns_none_for_invalid_urls(url: str) -> None:
    assert extract_video_id(url) is None

def test_get_transcript_raises_for_unavailable_video(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_unavailable(self, video_id, languages):
        raise VideoUnavailable(video_id)
    monkeypatch.setattr(YouTubeTranscriptApi, "fetch", raise_unavailable)
    with pytest.raises(TranscriptError, match="does not exist or is private"):
        get_transcript("dQw4w9WgXcQ")