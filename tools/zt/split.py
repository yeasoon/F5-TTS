
import re
def split_text_by_chapters(input_path, output_prefix=""):
    # Read the whole text
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Regular expression for matching "Chapter 1", "Chapter 2", etc.
    # Case-insensitive, and allows optional spaces
    pattern = re.compile(r'(第.*章\s)', re.IGNORECASE)

    # Split while keeping the chapter headings
    parts = pattern.split(content)

    # Merge each heading with its content
    chapters = []
    for i in range(1, len(parts), 2):
        heading = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        chapters.append((heading, body))

    # Save each chapter to a file
    for i, (heading, body) in enumerate(chapters, start=1):
        output_file = f"/data/tts/zt/split/{i}.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"{heading}\n\n{body}")
        print(f"Saved: {output_file}")

if __name__ == "__main__":
    split_text_by_chapters("/data/tts/zt/zt.txt")