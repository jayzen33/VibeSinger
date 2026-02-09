"""YingMusic-Singer: Zero-shot Singing Voice Synthesis and Editing.

A unified framework for zero-shot singing voice synthesis (SVS) and editing,
driven by annotation-free melody guidance.

Example:
    >>> from singer import YingSinger
    >>> singer = YingSinger(device="cuda")
    >>> audio = singer.inference(
    ...     timbre_audio_path="reference.wav",
    ...     timbre_audio_content="hello world",
    ...     melody_audio_path="melody.wav",
    ...     lyrics="new lyrics here",
    ... )
"""

__version__ = "0.1.0"
__author__ = "GiantAI Lab"

from .model import YingSinger

__all__ = ["YingSinger", "__version__"]
