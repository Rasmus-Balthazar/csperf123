#Script made using Claude AI and minimally modified by me
import random
import argparse

def generate_random_words_file(output_file, num_words, words_per_line, wordlist_file=None, seed=None):
    """
    Generate a file containing random words.
    
    Args:
        output_file (str): Path to the output file
        num_words (int): Total number of words to generate
        words_per_line (int): Number of words per line
        wordlist_file (str, optional): Path to a custom wordlist file
        seed (int, optional): Random seed for deterministic output
    """
    # Set seed if provided
    if seed is None:
        seed = random.randint(1, 1000000)
    
    random.seed(seed)
    print(f"Using seed: {seed}")
    
    # Default word list - used if no wordlist file is provided
    default_word_list = [
        "apple", "banana", "carrot", "dog", "elephant", "flower", "guitar", 
        "house", "island", "jungle", "kite", "lemon", "mountain", "notebook", 
        "ocean", "piano", "queen", "river", "sunset", "tree", "umbrella", 
        "violin", "window", "xylophone", "yellow", "zebra", "book", "computer", 
        "desk", "earth", "fire", "glass", "hat", "ice", "jacket", "key", 
        "lamp", "moon", "night", "orange", "paper", "quiet", "road", "sun", 
        "time", "university", "village", "water", "xray", "yard", "zipper"
    ]
    
    # Load custom wordlist if provided
    if wordlist_file:
        try:
            with open(wordlist_file, 'r') as f:
                word_list = set()
                for line in f:
                    # Ignore empty lines and comments
                    if line.strip() and not line.startswith('#'):
                        word_list.update(line.strip().split())
                # Convert set back to list
                word_list = list(word_list)
            if not word_list:
                print(f"Warning: Wordlist file '{wordlist_file}' is empty. Using default wordlist instead.")
                word_list = default_word_list
            else:
                print(f"Loaded {len(word_list)} words from '{wordlist_file}'")
        except FileNotFoundError:
            print(f"Warning: Wordlist file '{wordlist_file}' not found. Using default wordlist instead.")
            word_list = default_word_list
    else:
        word_list = default_word_list
        print(f"Using default wordlist with {len(word_list)} words")
    
    with open(output_file, 'w') as f:
        # Write words to file
        for i in range(1, num_words + 1):
            word = random.choice(word_list)
            f.write(word)
            
            # Add space or newline
            if i % words_per_line == 0:
                f.write('\n')
            else:
                f.write(' ')
        
        # Ensure the file ends with a newline
        if num_words % words_per_line != 0:
            f.write('\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a file with random words')
    parser.add_argument('output_file', type=str, help='Output file path')
    parser.add_argument('--num_words', type=int, default=100, help='Number of words to generate (default: 100)')
    parser.add_argument('--words_per_line', type=int, default=10, help='Number of words per line (default: 10)')
    parser.add_argument('--wordlist', type=str, help='Path to a custom wordlist file (one word per line)')
    parser.add_argument('--seed', type=int, help='Random seed for deterministic output')
    
    args = parser.parse_args()
    
    generate_random_words_file(args.output_file, args.num_words, args.words_per_line, 
                             args.wordlist, args.seed)