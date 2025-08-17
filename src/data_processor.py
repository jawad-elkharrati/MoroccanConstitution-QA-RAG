"""data_processor.py: Handles loading, cleaning, and preprocessing of the Moroccan Constitution text."""

import re
import json

CONSTITUTION_TXT_PATH = r"C:\Users\dell\Downloads\constitution_qa_project\Morocco_Constitution_2011.txt"
PROCESSED_DATA_PATH = r"C:\Users\dell\Downloads\constitution_qa_project\processed_constitution_chunks.json"

def load_raw_text(file_path: str) -> str:
    """Loads raw text from a file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        print(f"Successfully loaded raw text from {file_path}")
        return raw_text
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return ""
    except Exception as e:
        print(f"An error occurred while loading {file_path}: {e}")
        return ""

def initial_clean_text(text: str) -> str:
    """Performs initial cleaning of the text."""
    text = re.sub(r"PDF generated:.*\n", "", text)
    text = re.sub(r"constituteproject\.org\n", "", text)
    text = re.sub(r"Morocco 2011\s+Page \d+\n", "", text)
    text = re.sub(r"Table of contents[\s\S]*?(?=Preamble)", "", text, flags=re.IGNORECASE)
    text = text.replace("\f", "\n")
    text = re.sub(r"\n\s*\n", "\n\n", text)
    text = re.sub(r" +", " ", text)
    text = text.strip()
    print("Initial text cleaning performed.")
    return text

def segment_into_articles(text: str) -> list[dict]:
    """Segments the constitution text into articles and extracts metadata."""
    chunks = []
    current_title = "Preamble"
    title_pattern = re.compile(r"^Title (\w+): (.*)$", re.MULTILINE)
    article_pattern = re.compile(r"^(Article (?:\w+|\d+))\s*([\s\S]*?)(?=(^Article (?:\w+|\d+))|(^Title \w+:)|\Z)", re.MULTILINE)

    preamble_match = re.search(r"^(Preamble)([\s\S]*?)(?=^Title One: General Provisions)", text, re.MULTILINE | re.IGNORECASE)
    if preamble_match:
        preamble_text = preamble_match.group(2).strip()
        chunks.append({
            "id": "preamble",
            "text": preamble_text,
            "metadata": {"title": "Preamble", "article_number": "Preamble"}
        })
        print("Processed Preamble.")

    last_pos = preamble_match.end() if preamble_match else 0

    for match in article_pattern.finditer(text):
        article_id_full = match.group(1).strip()
        article_text = match.group(2).strip()

        possible_title_text = text[last_pos:match.start()]
        title_search = title_pattern.search(possible_title_text)
        if title_search:
            current_title_name = title_search.group(1).strip()
            current_title_desc = title_search.group(2).strip()
            current_title = f"Title {current_title_name}: {current_title_desc}"
            print(f"Found new section: {current_title}")

        if article_text:
            chunks.append({
                "id": article_id_full.lower().replace(" ", "_"),
                "text": article_text,
                "metadata": {"title": current_title, "article_number": article_id_full}
            })
        last_pos = match.end()

    print(f"Segmented text into {len(chunks)} chunks (Preamble + Articles).")
    return chunks

def save_processed_data(data: list[dict], file_path: str):
    """Saves the processed data to a JSON file."""
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"Successfully saved processed data to {file_path}")
    except Exception as e:
        print(f"An error occurred while saving data to {file_path}: {e}")

if __name__ == "__main__":
    print("Starting data processing...")
    raw_constitution_text = load_raw_text(CONSTITUTION_TXT_PATH)
    if raw_constitution_text:
        cleaned_text = initial_clean_text(raw_constitution_text)
        segmented_articles = segment_into_articles(cleaned_text)
        if segmented_articles:
            save_processed_data(segmented_articles, PROCESSED_DATA_PATH)
            print(f"Processed {len(segmented_articles)} chunks.")
        else:
            print("No articles were segmented.")
    else:
        print("Could not load raw constitution text. Processing aborted.")
    print("Data processing finished.")