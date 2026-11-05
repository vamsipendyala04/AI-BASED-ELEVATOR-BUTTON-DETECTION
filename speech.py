"""Cross-platform text-to-speech helper using pyttsx3."""

import pyttsx3

_engine = pyttsx3.init()


def speak(message: str) -> None:
    """Speak a message aloud and wait until playback finishes."""
    _engine.say(message)
    _engine.runAndWait()
