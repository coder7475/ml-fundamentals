import re

def normalize_bangla_text(text):
    # Step 1: Remove invisible characters
    text = text.replace('\u200d', '')  # Zero-width joiner
    text = text.replace('\u200c', '')  # Zero-width non-joiner
    text = text.replace('\u00a0', ' ') # Non-breaking space to regular space
    
    # Step 2: Normalize visually similar characters
    text = re.sub(r'[য়]', 'য়', text)   # Normalize 'য়' to 'য়'
    text = re.sub(r'[র‍]', 'র', text)   # Remove ZWJ from 'র‍' if used wrongly
    text = re.sub(r'[ৎ]', 'ত্', text)   # Rare cases where 'ৎ' needs to be decomposed
    text = re.sub(r'[ড়]', 'র়', text)   # Normalize dotted র
    text = re.sub(r'[ঢ়]', 'ঢ়', text)   # Normalize dotted ঢ
    text = re.sub(r'[ঙ‍]', 'ঙ', text)   # Remove ZWJ after ঙ if it exists

    # Step 3: Normalize vowel signs and nukta forms
    text = re.sub(r'[\u09c7\u09c8]', '\u09c7', text)  # Normalize e-kar and ai-kar variants
    text = re.sub(r'[\u09cb\u09cc]', '\u09cb', text)  # Normalize o-kar and au-kar variants
    
    # Optional: remove duplicate diacritics (common from faulty OCR or typing)
    text = re.sub(r'([ািীুূেৈোৌ])\1+', r'\1', text)   # Collapse repeated vowel signs
    
    return text

if __name__ == "__main__": 
    # Sample Bangla text
    bangla_text = "আমি বাংলায় বই পড়তে ভালোবাসি। এটি খুবই আনন্দদায়ক! কিন্তু, কিছু বইয়ের দাম বেশি।"
    print(bangla_text)
    print(normalize_bangla_text(bangla_text))