import random
import os

class TutorAgent:
    def __init__(self, words_file_path=None):
        """
        Initialize the TutorAgent.
        :param words_file_path: Path to the words.txt file.
        """
        if words_file_path is None:
            # Default path relative to this file
            words_file_path = os.path.join(os.path.dirname(__file__), "words.txt")
        
        self.words = self._load_words(words_file_path)
        
        # Initialize mistake_count dictionary for letters A-Z (all start at 0)
        self.mistake_count = {chr(i): 0 for i in range(ord('A'), ord('Z') + 1)}
        
    def _load_words(self, path):
        """Loads words from a text file, one per line."""
        if not os.path.exists(path):
            print(f"Warning: Words file not found at {path}. Using a default list.")
            return ["HELLO", "WORLD", "CAT", "DOG", "APPLE", "ASL", "SIGN", "TUTOR"]
        
        with open(path, "r") as f:
            words = [line.strip().upper() for line in f if line.strip()]
        return words

    def update_performance(self, letter, is_correct):
        """
        Updates the error count for a specific letter.
        :param letter: The character to update (A-Z).
        :param is_correct: Boolean indicating if the user got it right.
        """
        letter = letter.upper()
        if letter in self.mistake_count:
            if not is_correct:
                # If incorrect, increment the mistake count
                self.mistake_count[letter] += 1
            else:
                # If correct, slightly decrease the count (min 0)
                self.mistake_count[letter] = max(0, self.mistake_count[letter] - 0.5)

    def get_next_word(self):
        """
        The AI Logic: Find weak letters and pick a word containing them.
        """
        # Find letters with error counts > 0
        weak_letters = [l for l, count in self.mistake_count.items() if count > 0]
        
        if not weak_letters:
            # If no weak letters, return a random word
            return random.choice(self.words)
        
        # Sort weak letters by error count (descending)
        weak_letters.sort(key=lambda l: self.mistake_count[l], reverse=True)
        
        # Try to find words containing the most "difficult" letters first
        for weak_letter in weak_letters:
            candidate_words = [word for word in self.words if weak_letter in word]
            if candidate_words:
                # Pick a random candidate word that contains the weak letter
                return random.choice(candidate_words)
        
        # Fallback to random word if no word contains the weak letters
        return random.choice(self.words)

    def get_stats(self):
        """Returns the current error stats."""
        return self.mistake_count
