"""Helper functions: YouTube transcript extraction and AI calls."""

import re
from urllib.parse import urlparse, parse_qs

from requests.exceptions import RequestException  # installed together with the library
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    CouldNotRetrieveTranscript,
)

# A YouTube video ID is always exactly 11 characters: letters, digits, "-" and "_"
VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")


class TranscriptError(Exception):
    """Error with a user-friendly message that can be shown directly in the UI."""


def extract_video_id(url: str) -> str | None:
    """Extract the video ID from the most common YouTube URL formats.

    Supported examples:
        https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=30s
        https://youtu.be/dQw4w9WgXcQ?si=abc
        https://www.youtube.com/shorts/dQw4w9WgXcQ

    Returns the 11-character ID, or None if the URL is not a valid YouTube link.
    """
    url = url.strip()                      # remove accidentally copied whitespace
    if "://" not in url:                   # also accept "youtube.com/..." without https
        url = "https://" + url
    parsed = urlparse(url)                 # split the URL into its components

    # Normalize "www.youtube.com" and "m.youtube.com" (mobile) to "youtube.com"
    host = parsed.netloc.lower().removeprefix("www.").removeprefix("m.")
    video_id = None

    if host == "youtu.be":
        # Short link: the ID is the path itself -> "/dQw4w9WgXcQ"
        video_id = parsed.path.lstrip("/").split("/")[0]
    elif host == "youtube.com":
        if parsed.path == "/watch":
            # parse_qs turns "v=abc&t=30s" into {"v": ["abc"], "t": ["30s"]}
            video_id = parse_qs(parsed.query).get("v", [None])[0]
        elif parsed.path.startswith(("/shorts/", "/embed/", "/live/")):
            # "/shorts/abc".split("/") -> ["", "shorts", "abc"]
            video_id = parsed.path.split("/")[2]

    # Final check: make sure the result looks like a real video ID
    if video_id and VIDEO_ID_PATTERN.match(video_id):
        return video_id
    return None


def get_transcript(video_id: str, languages: tuple[str, ...] = ("ro", "en")) -> str:
    """Download a video's subtitles and join them into a single text.

    Languages are tried in the given order: Romanian first, then English.
    Raises TranscriptError with a clear message if the transcript can't be fetched.
    """
    try:
        fetched = YouTubeTranscriptApi().fetch(video_id, languages=languages)
    # Catch the SPECIFIC errors first, then the general one (order matters!)
    except TranscriptsDisabled:
        raise TranscriptError("Subtitles are disabled for this video.") from None
    except NoTranscriptFound:
        raise TranscriptError(f"No subtitles found in: {', '.join(languages)}.") from None
    except VideoUnavailable:
        raise TranscriptError("The video does not exist or is private.") from None
    except CouldNotRetrieveTranscript as e:
        # e.g. YouTube temporarily blocking requests (IpBlocked)
        raise TranscriptError(f"YouTube refused the request ({type(e).__name__}).") from e
    except RequestException as e:
        raise TranscriptError("Could not connect to YouTube. Check your internet connection.") from e

    # The transcript arrives as timed "snippets"; we keep only the text
    return " ".join(snippet.text.replace("\n", " ") for snippet in fetched)