from src.tutor_logic import TutorAgent
import os

# Ensure the paths are correct relative to the test script
agent = TutorAgent(words_file_path="src/words.txt")

print("--- ASL Tutor Logic Test ---")
first_word = agent.get_next_word()
print("First random word:", first_word)

# Simulate the user failing 'A' multiple times
print("\nSimulating user failing 'A' 3 times...")
agent.update_performance('A', False)
agent.update_performance('A', False)
agent.update_performance('A', False)

# Simulate the user failing 'Z' once
print("Simulating user failing 'Z' 1 time...")
agent.update_performance('Z', False)

next_word = agent.get_next_word()
print("Next recommended word (should contain 'A' or 'Z'):", next_word)

# Show stats
print("\n--- Current Stats (Non-zero errors) ---")
stats = agent.get_stats()
for letter, errors in stats.items():
    if errors > 0:
        print(f"Letter {letter}: {errors} mistakes")

# Test if it prefers the highest error (A should be prioritized over Z)
print("\nFetching more words to see priority...")
for i in range(3):
    print(f"Word {i+1}:", agent.get_next_word())
