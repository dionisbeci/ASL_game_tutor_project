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
        
        # Initialize performance_score dictionary for letters A-Z (all start at 0)
        self.performance_score = {chr(i): 0 for i in range(ord('A'), ord('Z') + 1)}
        
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
        Updates the performance score for a specific letter.
        :param letter: The character to update (A-Z).
        :param is_correct: Boolean indicating if the user got it right.
        """
        letter = letter.upper()
        if letter in self.performance_score:
            if not is_correct:
                # If incorrect, decrease score
                self.performance_score[letter] -= 1
            else:
                # If correct, increase score
                self.performance_score[letter] += 1

    def get_next_word(self):
        """
        The AI Logic: Find weak letters (lowest scores) and pick a word containing them.
        """
        # Sort letters by score (ascending: worst performance first)
        sorted_letters = sorted(self.performance_score.items(), key=lambda x: x[1])
        
        # Take the top 5 weakest letters
        weak_letters = [l for l, score in sorted_letters[:5]]
        
        # Try to find words containing the weak letters
        for weak_letter in weak_letters:
            candidate_words = [word for word in self.words if weak_letter in word]
            if candidate_words:
                return random.choice(candidate_words)
        
        # Fallback
        return random.choice(self.words)

    def get_stats(self):
        """Returns the current performance stats."""
        return self.performance_score
