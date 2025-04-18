import os
import logging
import hashlib
from gtts import gTTS

logger = logging.getLogger(__name__)

class SpeechGenerator:
    """
    A utility class to generate and manage speech audio files for gestures.
    """
    
    def __init__(self, audio_dir='static/audio'):
        """Initialize the speech generator with a directory for audio files."""
        self.audio_dir = audio_dir
        
        # Create audio directory if it doesn't exist
        if not os.path.exists(self.audio_dir):
            os.makedirs(self.audio_dir)
            logger.info(f"Created audio directory: {self.audio_dir}")
    
    def generate_speech_for_gesture(self, gesture_name):
        """
        Generate an audio file for a gesture if it doesn't already exist.
        
        Args:
            gesture_name: The name of the gesture (e.g., "thumbs_up")
            
        Returns:
            str: The URL path to the audio file
        """
        # Format the gesture name for speech (convert snake_case to spoken words)
        spoken_text = self._format_for_speech(gesture_name)
        
        # Generate a unique filename based on the spoken text
        filename = self._get_audio_filename(spoken_text)
        filepath = os.path.join(self.audio_dir, filename)
        
        # Check if audio file already exists
        if not os.path.exists(filepath):
            try:
                # Generate speech audio file
                tts = gTTS(text=spoken_text, lang='en', slow=False)
                tts.save(filepath)
                logger.info(f"Generated audio for '{gesture_name}': {filepath}")
            except Exception as e:
                logger.error(f"Error generating speech for '{gesture_name}': {e}")
                return None
        
        # Return the URL path to the audio file (relative to static directory)
        return f"/audio/{filename}"
    
    def get_speech_for_all_gestures(self, gestures):
        """
        Generate speech files for a list of gestures and return a mapping.
        
        Args:
            gestures: List of gesture dictionaries with 'name' key
            
        Returns:
            dict: Mapping of gesture names to audio URLs
        """
        audio_map = {}
        
        for gesture in gestures:
            gesture_name = gesture['name']
            audio_url = self.generate_speech_for_gesture(gesture_name)
            if audio_url:
                audio_map[gesture_name] = audio_url
        
        return audio_map
    
    def _format_for_speech(self, gesture_name):
        """Format a gesture name for speech output (convert snake_case to normal text)."""
        # Replace underscores with spaces and capitalize words
        words = gesture_name.split('_')
        formatted_text = ' '.join(word.capitalize() for word in words)
        
        # Customize the spoken text for better clarity
        if gesture_name == "thumbs_up":
            return "Thumbs Up detected"
        elif gesture_name == "thumbs_down":
            return "Thumbs Down detected"
        else:
            return f"{formatted_text} gesture detected"
    
    def _get_audio_filename(self, text):
        """Create a unique filename for the given text."""
        # Create a hash of the text to ensure unique filenames
        text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        safe_text = ''.join(c if c.isalnum() else '_' for c in text.lower())[:20]
        return f"{safe_text}_{text_hash}.mp3"