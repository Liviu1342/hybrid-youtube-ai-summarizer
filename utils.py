"""Funcții ajutătoare: extragerea transcriptului YouTube și apelul către AI."""

import re
from urllib.parse import urlparse, parse_qs

# Un ID YouTube are mereu exact 11 caractere: litere, cifre, "-" și "_"
VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")


def extract_video_id(url: str) -> str | None:
    """Extrage ID-ul unui videoclip YouTube din diverse formate de link.

    Exemple acceptate:
        https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=30s
        https://youtu.be/dQw4w9WgXcQ?si=abc
        https://www.youtube.com/shorts/dQw4w9WgXcQ

    Returnează ID-ul (11 caractere) sau None dacă linkul nu e valid.
    """
    url = url.strip()                      # eliminăm spațiile copiate din greșeală
    if "://" not in url:                   # acceptăm și "youtube.com/..." fără https
        url = "https://" + url
    parsed = urlparse(url)                 # desfacem URL-ul în bucăți

    # "www.youtube.com" și "m.youtube.com" (mobil) devin simplu "youtube.com"
    host = parsed.netloc.lower().removeprefix("www.").removeprefix("m.")
    video_id = None

    if host == "youtu.be":
        # Linkul scurt: ID-ul e chiar calea -> "/dQw4w9WgXcQ"
        video_id = parsed.path.lstrip("/").split("/")[0]
    elif host == "youtube.com":
        if parsed.path == "/watch":
            # parse_qs transformă "v=abc&t=30s" în {"v": ["abc"], "t": ["30s"]}
            video_id = parse_qs(parsed.query).get("v", [None])[0]
        elif parsed.path.startswith(("/shorts/", "/embed/", "/live/")):
            # "/shorts/abc".split("/") -> ["", "shorts", "abc"]
            video_id = parsed.path.split("/")[2]

    # Validare finală: ne asigurăm că arată ca un ID real
    if video_id and VIDEO_ID_PATTERN.match(video_id):
        return video_id
    return None