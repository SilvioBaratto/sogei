# sogei

extractor_api:
  extract_title.py:
import os
import glob
import re
import statistics
import string

import fitz  # PyMuPDF
import pdfplumber

def _line_in_table(
        line_bbox, 
        table_bboxes
    ) -> bool:
    """
    Check if a line intersects with any table bounding box.
    """
    x0, y0, x1, y1 = line_bbox
    for (tx0, ttop, tx1, tbottom) in table_bboxes:
        # Boxes overlap if they do not lie completely outside each other
        if not (x1 < tx0 or x0 > tx1 or y1 < ttop or y0 > tbottom):
            return True
    return False

def _mark_lines_in_tables(
        lines, 
        table_bboxes
    ) -> None:
    """
    Marks lines that intersect with any table bounding box as 'in_table'.
    """

    # Segna ogni riga che si trova all'interno di una tabella.
    # Controlla se il bounding box di una riga si sovrappone con il bounding box di una tabella.
    for line in lines:
        if _line_in_table(line['bbox'], table_bboxes):
            line['in_table'] = True

def _extract_page_lines(
        page
    ) -> list:
    """
    Extract text lines and associated metadata from a PDF page.
    """
    text_data = page.get_text("dict")
    lines = []
    for block in text_data.get("blocks", []):
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue

            span_texts = [span["text"].strip() for span in spans if span["text"].strip()]
            line_text = " ".join(span_texts)

            if not line_text:
                continue

            font_sizes = [span["size"] for span in spans]
            avg_font_size = statistics.mean(font_sizes) if font_sizes else 0
            font_names = [span["font"] for span in spans]
            bbox = line["bbox"]

            lines.append({
                'text': line_text,
                'size': avg_font_size,
                'fontnames': font_names,
                'bbox': bbox
            })
    return lines

def _ends_with_punctuation(
        text
    ) -> bool:
    """
    Check if a text ends with punctuation.
    """
    return bool(re.match(r'.*[.?!:;]$', text.strip()))

def _is_likely_title_format(
        line_dict
    ) -> bool:
    """
    Determine if a line is likely to be a title based on its formatting (bold or uppercase ratio).
    """
    text = line_dict['text']
    bold = any(('Bold' in fn or 'Black' in fn or 'Heavy' in fn) for fn in line_dict['fontnames'])
    letters = [c for c in text if c.isalpha()]
    uppercase_ratio = sum(1 for c in letters if c.isupper()) / len(letters) if letters else 0
    return bold or uppercase_ratio > 0.3

def _find_main_title(lines, page_height, page_width) -> dict:
    """
    Attempt to find the main title from the first page by looking for the largest text lines
    near the top of the page and merging consecutive lines that likely belong together.
    """
    lines_sorted = sorted(lines, key=lambda x: x['size'], reverse=True)
    top_cutoff = page_height * 0.6  # Consider only lines in the top 60% of the page

    # Filter lines that are reasonably sized and positioned near the top of the page
    candidate_lines = []
    for ln in lines_sorted:
        txt = ln['text'].strip()
        if len(txt) > 5 and not _ends_with_punctuation(txt):  # Avoid too-short or punctuated lines
            top_y = ln['bbox'][1]
            if top_y < top_cutoff and _is_likely_title_format(ln):
                candidate_lines.append(ln)

    # Refine merging logic
    merged_lines = []
    if candidate_lines:
        merged_text = candidate_lines[0]['text']
        merged_bbox = list(candidate_lines[0]['bbox'])  # Convert to list for merging
        for i in range(1, len(candidate_lines)):
            current_line = candidate_lines[i]
            previous_line = candidate_lines[i - 1]

            # Calculate vertical and horizontal gaps
            vertical_gap = current_line['bbox'][1] - previous_line['bbox'][3]

            # Ensure gaps are within reasonable limits to avoid unrelated lines
            if (
                0 < vertical_gap < 20  # Vertical gap tolerance: lines must be close
                and current_line['size'] == previous_line['size']  # Ensure font sizes match
                and set(current_line['fontnames']) == set(previous_line['fontnames'])  # Font consistency
            ):
                # Merge text and expand bounding box
                merged_text += f" {current_line['text']}"
                merged_bbox[2] = max(merged_bbox[2], current_line['bbox'][2])  # Update right edge
                merged_bbox[3] = current_line['bbox'][3]  # Update bottom edge
            else:
                # Save the merged line and start a new group
                merged_lines.append({
                    'text': merged_text.strip(),
                    'bbox': merged_bbox,
                    'size': previous_line['size'],
                    'fontnames': previous_line['fontnames'],
                })
                merged_text = current_line['text']
                merged_bbox = list(current_line['bbox'])

        # Add the last merged line
        merged_lines.append({
            'text': merged_text.strip(),
            'bbox': merged_bbox,
            'size': candidate_lines[-1]['size'],
            'fontnames': candidate_lines[-1]['fontnames'],
        })

    # Return the merged line with the largest font size as the main title
    if merged_lines:
        return max(merged_lines, key=lambda x: x['size'])
    elif candidate_lines:
        return candidate_lines[0]
    return None

def _is_bullet_point(text: str) -> bool:
    """
    Detect if the text is a standalone bullet.
    """
    bullet_characters = ['-', '•', '▪', '●']
    return text.strip() in bullet_characters

def _is_potential_bullet_line(text: str) -> bool:
    """
    Detect if a line looks like a bullet line.
    Examples:
    - PHPStone download
    - • Google Chrome
    """
    bullet_pattern = r'^[\u2022\u25AA\u2023\u25B8\u2219\-]\s*.+$'
    return bool(re.match(bullet_pattern, text.strip()))

def _is_section_number(text) -> bool:
    """
    Check if text represents a section number (e.g., "1.2.3", "2.", "2").
    """
    pattern = r'^\d+(\.\d+)*\.?$'
    return bool(re.match(pattern, text.strip()))

def _clean_text(text) -> str:
    """
    Clean extracted text by removing non-printable characters and excessive whitespace.
    """
    text = ''.join([c if c.isprintable() else ' ' for c in text]).strip()
    text = ' '.join(text.split())
    return text.strip()

import re

def _is_part_of_paragraph(current_line, prev_line, next_line):
    """
    Determines if a line is part of a paragraph.
    It is considered part of a paragraph if:
      - It is close in vertical position to the line before or after.
      - It contains too many words.
      - It is not a section number like 1.1.1 or 2.3.4.
      - It is not a bold, short title.
    """
    # 1. Check if it's a section number (like 1.1.1 Ambito di applicabilità)
    section_pattern = r'^\d+(\.\d+)*\s*.+$'
    if re.match(section_pattern, current_line['text']):
        return False  # Section numbers are NOT paragraphs

    # 2. Calculate the distance from the previous and next lines
    prev_distance = current_line['bbox'][1] - prev_line['bbox'][3] if prev_line else float('inf')
    next_distance = next_line['bbox'][1] - current_line['bbox'][3] if next_line else float('inf')

    # 3. If the distance between lines is small, it is part of a paragraph
    close_to_others = prev_distance < 20 or next_distance < 20

    # 4. If the line has more than 10 words, it is likely part of a paragraph
    is_long_line = len(current_line['text'].split()) > 10

    return close_to_others

def _is_potential_title(text, font_size, fontnames, median_font_size, size_factor=1.0, bbox=None, prev_line=None, next_line=None, page_width=None) -> bool:
    """
    Check if text is a potential title based on its format and size.
    Titles should:
    - Be bold and larger than a specified threshold
    - Exclude specific non-title phrases, bullet point items, and patterns
    - Check if it's near the top of the page or in a prominent position
    - Exclude lines that look like part of lists or bullet points
    """
    # Exclude specific phrases
    excluded_phrases = [
        "Identificativo Documento:", "Ambiti :", 
        "Classificazione di riservatezza :", "Destinatari :",
        "Documento e informazioni per circolazione e uso esclusivamente interni", 
    ]
    
    if text.strip() in excluded_phrases:
        return False

    # Exclude text that matches specific ID-like patterns
    id_pattern = r'^ID:\s*[A-Za-z0-9_-]+$'  # Matches "ID: xyz", "ID: IS-01", etc.
    if re.match(id_pattern, text.strip()):
        return False

    # Exclude lines that follow bullet point symbols (like "-", "•", "▪")
    if _is_bullet_point(text) or _is_potential_bullet_line(text):
        return False

    # Exclude lines with specific patterns indicating bullet-style lists
    bullet_point_pattern = r'^[\u2022\u25AA\u2023\u25B8\u2219\-]\s*.+$'
    if re.match(bullet_point_pattern, text.strip()):
        return False

    # Check if it looks like a section number
    section_pattern = r'^\d+(\.\d+)*\s*.+$'
    is_section_number = bool(re.match(section_pattern, text))

    # Check if the font is bold
    is_bold = any('Bold' in font or 'Black' in font or 'Heavy' in font for font in fontnames)

    # Check if the font size is sufficiently large
    is_large = font_size >= median_font_size * size_factor

    # Check if the line is "wide" enough to be a title (avoids short text like "• R")
    if bbox:
        width = bbox[2] - bbox[0]
        if width < 50:  # Arbitrary threshold
            return False
        
    # Check if the current line is part of a paragraph
    if prev_line or next_line:
        if _is_part_of_paragraph({'bbox': bbox, 'text': text}, prev_line, next_line):
            return False

    # Title is valid if it's a section number or bold and large
    return is_section_number or (is_bold and is_large)

def _find_title_candidates(lines, median_font_size, page_number, size_factor=1.0) -> list:
    """
    Identify candidate title lines based on font size, formatting, and proximity to the top.
    """
    candidates = []
    skip_next_line = False

    for i, ln in enumerate(lines):
        if skip_next_line:
            skip_next_line = False
            continue

        # Extract metadata from the line
        text = ln['text'].strip()
        font_size = ln['size']
        fontnames = ln['fontnames']
        bbox = ln['bbox']

        prev_line = lines[i-1] if i > 0 else None
        next_line = lines[i+1] if i + 1 < len(lines) else None

        # Check if the line is sufficiently large and looks like a title
        is_large = font_size >= median_font_size * size_factor
        looks_like_title = _is_potential_title(text, font_size, fontnames, median_font_size, size_factor, bbox, prev_line, next_line)

        # Check if the previous line is a bullet point and if this line follows it
        if i > 0:
            previous_line_text = lines[i-1]['text'].strip()
            if _is_bullet_point(previous_line_text) or _is_potential_bullet_line(previous_line_text):
                # If the previous line is a bullet, this is not a title
                continue

        # Merge section number with the next line if needed
        if _is_section_number(text) and i + 1 < len(lines):
            next_line = lines[i + 1]
            if not _is_section_number(next_line['text']):
                merged_text = f"{text} {next_line['text']}"
                merged_text = _clean_text(merged_text)
                merged_line = {
                    'text': merged_text,
                    'size': max(font_size, next_line['size']),
                    'fontnames': fontnames + next_line['fontnames'],
                    'bbox': ln['bbox']
                }

                if (merged_line['size'] >= median_font_size * size_factor or
                        _is_potential_title(merged_line['text'], merged_line['size'], merged_line['fontnames'], median_font_size, size_factor, merged_line['bbox'])):
                    candidates.append(merged_line)

                skip_next_line = True
                continue

        if is_large and looks_like_title:
            candidates.append(ln)
            
    return candidates

def _merge_connected_titles(titles) -> list:
    """
    Post-processes the extracted titles to:
    1. Merge section numbers (e.g. '1.') with their following title line.
    2. Merge titles that are very close vertically (less than 16 units apart) and on the same page.
    """
    merged_titles = []
    i = 0
    while i < len(titles):
        current_title = titles[i]

        # Check if the current title is a section number (e.g., "1." or "1.2.")
        if re.match(r'^\d+(\.\d+)*\.$', current_title['text']) and (i + 1) < len(titles):
            next_title = titles[i + 1]
            # Merge the current title (section number) with the next one
            merged_text = f"{current_title['text']} {next_title['text']}"
            merged_bbox = [
                min(current_title['bbox'][0], next_title['bbox'][0]),
                min(current_title['bbox'][1], next_title['bbox'][1]),
                max(current_title['bbox'][2], next_title['bbox'][2]),
                max(current_title['bbox'][3], next_title['bbox'][3]),
            ]

            merged_entry = {
                'page': current_title['page'],
                'text': merged_text,
                'bbox': tuple(merged_bbox),
            }
            # Move past the next title
            i += 2
        else:
            # If not a section heading, start with the current title as a base
            merged_entry = current_title
            i += 1

        # Now, try to merge with subsequent lines that are close vertically
        # Keep merging if:
        # - The next title is on the same page
        # - The vertical distance between the merged_entry and the next title is < 16
        # - The next title is not a section heading itself (no need to merge again)
        while i < len(titles):
            next_title = titles[i]
            
            # Check if on the same page
            if next_title['page'] != merged_entry['page']:
                break

            # Calculate vertical distance
            # We'll consider the vertical gap as the difference between the top of the next line 
            # and the bottom of the current merged_entry.
            current_bottom = merged_entry['bbox'][3]
            next_top = next_title['bbox'][1]
            vertical_gap = next_top - current_bottom

            # Check if next title looks like a section heading
            is_section_heading = bool(re.match(r'^\d+(\.\d+)*\.$', next_title['text']))

            if vertical_gap < 16 and not is_section_heading:
                # Merge them
                merged_text = f"{merged_entry['text']} {next_title['text']}"
                merged_bbox = (
                    min(merged_entry['bbox'][0], next_title['bbox'][0]),
                    min(merged_entry['bbox'][1], next_title['bbox'][1]),
                    max(merged_entry['bbox'][2], next_title['bbox'][2]),
                    max(merged_entry['bbox'][3], next_title['bbox'][3]),
                )

                merged_entry = {
                    'page': merged_entry['page'],
                    'text': merged_text,
                    'bbox': merged_bbox
                }

                i += 1
            else:
                # Either not close enough or is a section heading, so stop merging
                break

        merged_titles.append(merged_entry)

    return merged_titles

def extract_titles_from_pdf(pdf_path) -> list:
    """
    Extracts titles from a PDF file, ignoring text lines that are part of tables, 
    and identifying likely title candidates based on font size, formatting, and position.
    Skips pages containing "Indice Generale".
    
    Returns:
    - A list of dictionaries where each dictionary contains:
      - 'page': Page number where the title appears
      - 'text': The extracted title text
      - 'bbox': The bounding box (x0, y0, x1, y1) of the title
    """
    with pdfplumber.open(pdf_path) as plumber_doc:
        doc = fitz.open(pdf_path)  # For detecting the position and coordinates of the text elements
        titles = []  # To store the extracted titles

        for page_number, page in enumerate(doc, start=1):
            try:
                plumber_page = plumber_doc.pages[page_number - 1]

                # Check if the page contains "Indice Generale"
                page_text = plumber_page.extract_text()
                if "Indice generale" in page_text:
                    continue  # Skip this page if it contains "Indice Generale"

                # Identify tables on the page
                tables = plumber_page.find_tables()
                table_bboxes = [t.bbox for t in tables] if tables else []

                # Extract all text lines and associated metadata from the page
                lines = _extract_page_lines(page)

                # Mark lines that intersect with tables
                _mark_lines_in_tables(lines, table_bboxes)

                # Filter out lines inside tables
                lines = [l for l in lines if not l.get('in_table', False)]
                if not lines:
                    continue

                # Sort lines by their vertical and horizontal positions
                lines.sort(key=lambda l: (l['bbox'][1], l['bbox'][0]))

                # Calculate the median font size for the current page
                font_sizes = [l['size'] for l in lines]
                median_font_size = statistics.median(font_sizes) if font_sizes else 0

                # Detect the main title on the first page
                if page_number == 1:
                    main_title_line = _find_main_title(lines, page.rect.height, page.rect.width)
                    if main_title_line:
                        clean_main_title = _clean_text(main_title_line['text'])
                        if clean_main_title:
                            titles.append({
                                'page': page_number,
                                'text': clean_main_title,
                                'bbox': main_title_line['bbox']
                            })
                    continue  # Skip further processing for the first page

                # Detect other candidate titles for pages other than the first
                candidate_lines = _find_title_candidates(lines, median_font_size, page_number)

                # De-duplicate and clean candidate titles
                unique_candidates = {(cl['text'], tuple(cl['bbox'])): cl for cl in candidate_lines}
                filtered_candidates = []
                for (text, bbox), cl in unique_candidates.items():
                    clean_text_val = _clean_text(text)
                    if clean_text_val:
                        filtered_candidates.append({
                            'page': page_number,
                            'text': clean_text_val,
                            'bbox': cl['bbox']
                        })

                # Sort candidates by position on the page
                filtered_candidates.sort(key=lambda c: (c['bbox'][1], c['bbox'][0]))
                titles.extend(filtered_candidates)
            except Exception as e:
                print(f"Error processing page {page_number}: {e}")

        # Sort titles by page number and their positions on the page
        titles.sort(key=lambda t: (t['page'], t['bbox'][1], t['bbox'][0]))

        # Post-process titles to merge connected lines
        titles = _merge_connected_titles(titles)

        return titles

# --- Main Processing ---

def main():
    """Main function to orchestrate the processing of PDF files."""
    align_working_directory()
    pdf_files = find_files_in_data_directory("pdf")

    if not pdf_files:
        print("No PDF files found to process.")
        return

    for pdf_path in pdf_files:
        print(f"\nProcessing PDF: {pdf_path}\n")
        titles = extract_titles_from_pdf(pdf_path)
        if titles:
            for title in titles:
                print(f"Page {title['page']}: {title['text']}")
        else:
            print(f"No titles found in {pdf_path}")


def align_working_directory():
    """Aligns the working directory with the script's directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)


def find_files_in_data_directory(extension) -> list:
    """Finds files with the given extension in the 'data' directory."""
    return glob.glob(os.path.join(".", "data", f"*.{extension}"))

  html_to_md.py:
import re
from bs4 import BeautifulSoup

def html_table_to_markdown(html_table_str):
    """
    Converts a string containing a single <table>...</table>
    into Markdown table syntax.
    
    :param html_table_str: String containing an HTML <table> with optional <thead>, <tbody>, <tr>, <th>, <td>.
    :return: A string with the table in Markdown format.
    """
    soup = BeautifulSoup(html_table_str, "html.parser")

    # Find the actual <table> tag (there should be exactly one in the string)
    table = soup.find("table")
    if not table:
        return html_table_str  # Return unchanged if no <table> found

    # Collect rows
    rows = table.find_all("tr")
    
    # Convert each row into a list of cell texts
    table_data = []
    for row in rows:
        # Find both <th> and <td> cells
        cells = row.find_all(["th", "td"])
        row_data = [cell.get_text(strip=True) for cell in cells]
        table_data.append(row_data)

    # Now convert that 2D list into Markdown:
    # 1) If you want the first row to be a header, treat it specially.
    #    If the table doesn't have a header row, you can omit that logic.
    #    For demonstration, we'll always treat the first row as a header if it's not empty.
    
    if len(table_data) > 0:
        header = table_data[0]
        # If the first row is entirely empty, you might want to skip it or handle differently.
        # We'll do a quick check:
        if any(cell.strip() for cell in header):
            # We have a non-empty first row. Use it as header.
            body = table_data[1:]
            # Build the header line
            header_line = "| " + " | ".join(header) + " |"
            # Build the separator line (---)
            separator_line = "| " + " | ".join("---" for _ in header) + " |"
            
            markdown_lines = [header_line, separator_line]
            
            # Build each body row
            for row_cells in body:
                row_line = "| " + " | ".join(row_cells) + " |"
                markdown_lines.append(row_line)
        else:
            # The first row is empty, so we treat everything as body
            markdown_lines = []
            for row_cells in table_data:
                row_line = "| " + " | ".join(row_cells) + " |"
                markdown_lines.append(row_line)
    else:
        # No rows at all
        return ""

    return "\n".join(markdown_lines)


def convert_html_tables_in_markdown(md_file_path, output_file_path=None):
    """
    Reads a Markdown file, finds all <html><body><table>...</table></body></html> blocks,
    converts them to Markdown tables, and replaces them in the text.
    
    :param md_file_path: Path to the input Markdown file.
    :param output_file_path: Path to write the output (optional). If None, just returns the string.
    :return: The converted Markdown string (if output_file_path is None).
    """
    with open(md_file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Regex to find <html><body><table>...</table></body></html> blocks.
    # This is a simplistic approach; for well-formed HTML in the snippet, it should work.
    # If the HTML can contain nested elements or newlines, consider re.DOTALL and more robust patterns.
    pattern = r"<html><body><table.*?</table></body></html>"
    
    # Use re.DOTALL so that '.' matches newlines as well
    matches = re.findall(pattern, text, flags=re.DOTALL)

    # For each match, convert to Markdown
    for match in matches:
        md_table = html_table_to_markdown(match)
        # Replace the original HTML snippet with the Markdown table
        text = text.replace(match, md_table)

    # If requested, write to file
    if output_file_path:
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        return text


# ------------------------------
# Example Usage:
# ------------------------------
if __name__ == "__main__":
    # Suppose we have a test.md file containing a snippet of Markdown plus an HTML table block
    # e.g.:
    #
    # Some markdown text
    #
    # <html><body><table>
    # <tr><td></td><td></td></tr>
    # <tr><td>Keepass</td><td>Link</td></tr>
    # ...
    # </table></body></html>
    #
    # More markdown text

    input_md_file = "test.md"
    output_md_file = "converted_test.md"

    # Convert in-place and write result to output_md_file
    convert_html_tables_in_markdown(input_md_file, output_md_file)

    # If you just want the converted string without writing:
    # result = convert_html_tables_in_markdown(input_md_file, output_file_path=None)
    # print(result)

  services.py:
import os
import re
import json
from typing import List, Dict
from magic_pdf.data.data_reader_writer import FileBasedDataReader, FileBasedDataWriter
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.config.enums import SupportedPdfParseMethod
from extractor_api.extract_title import extract_titles_from_pdf

# Updated unwanted substrings to be removed (partial matching)
unwanted_substrings = [
    # 1. Header Lines Starting with "# Documento"
    # Matches:
    # - # Documento ed informazioni ad uso esclusivamente interno
    # - # Documento e informazioni per circolazione e uso esclusivamente interno
    r"#\s*Documento\s+(?:ed|e)\s+informazioni(?:\s+per\s+circolazione\s+e\s+uso)?\s+esclusivamente\s+intern(?:o|i).*?(?=\n|$)",

    # 2. Non-Header Lines Starting with "Documento"
    # Matches:
    # - Documento ed informazioni ad uso esclusivamente interno
    # - Documento e informazioni per circolazione e uso esclusivamente interni
    r"Documento\s+(?:ed|e)\s+informazioni(?:\s+per\s+circolazione\s+e\s+uso)?\s+esclusivamente\s+intern(?:o|i).*?(?=\n|$)",

    # 3. Specific Header Line with Different Wording
    # Matches:
    # - # Documento ed informazioni ad uso esclusivamente interno
    # - # Documento e informazioni ad uso esclusivamente interno
    r"#\s*Documento\s+(?:ed|e)\s+informazioni\s+ad\s+uso\s+esclusivamente\s+intern(?:o|i).*?(?=\n|$)",

    # 4. Specific Non-Header Line with Different Wording
    # Matches:
    # - Documento ed informazioni ad uso esclusivamente interno
    # - Documento e informazioni ad uso esclusivamente interno
    r"Documento\s+(?:ed|e)\s+informazioni\s+ad\s+uso\s+esclusivamente\s+intern(?:o|i).*?(?=\n|$)",

    # 5. Long Agency Lines with Optional Components and Variations
    r"Agenzia\s+delle\s+Entrate\s+Divisione\s+Risorse\s+[-–]\s+Direzione\s+Centrale\s+Tecnologie\s+(?:e|\\ominus)\s+Innovazione\s+Settore\s+Infrastrutture\s+(?:e|\\ominus)\s+Sicurezza\s+[-–]\s+Ufficio\s+Sicurezza\s+Informatica.*?(?=\n|$)",

    # 6. Via Giorgione Lines with Flexible Formatting
    r"Via\s+Giorgione,\s+\d+\s+[-–]\s+\d+\s+Roma\s+[-–]\s+Tel\.\s+\d{2}\s+\d{8}\s+-email:.*?(?=\n|$)",

    # 7. Image Markdown Strings (Case-Insensitive)
    r'!\[.*?\]\(.*?\.(?:png|jpg|jpeg|gif|bmp|svg|tiff|webp)\)',
]

def init_app():
    """Initialization logic for the application if needed."""
    pass

def extract_pdf_content(
    pdf_file_path,
    output_dir,
    method='txt',  # Default to text-based extraction
    lang="it",  # Specify language for OCR (optional)
):
    """Extract content from a PDF and save only the necessary results to the output directory."""
    
    # Ensure output directory exists
    image_output_dir = os.path.join(output_dir, "images")
    os.makedirs(image_output_dir, exist_ok=True)

    # Set up readers and writers
    reader = FileBasedDataReader("")
    image_writer = FileBasedDataWriter(image_output_dir)
    md_writer = FileBasedDataWriter(output_dir)

    # Read PDF content
    pdf_bytes = reader.read(pdf_file_path)
    titles = extract_titles_from_pdf(pdf_file_path)
    ds = PymuDocDataset(pdf_bytes)

    # Perform inference based on classification
    if ds.classify() == SupportedPdfParseMethod.OCR:
        infer_result = ds.apply(doc_analyze, ocr=True, lang=lang)
        pipe_result = infer_result.pipe_ocr_mode(image_writer)
    else:
        infer_result = ds.apply(doc_analyze, ocr=False, lang=lang)
        pipe_result = infer_result.pipe_txt_mode(image_writer)

    # Save outputs
    pdf_basename = os.path.splitext(os.path.basename(pdf_file_path))[0]
    markdown_file_path = os.path.join(output_dir, f"{pdf_basename}.md")
    content_list_path = os.path.join(output_dir, f"{pdf_basename}_content_list.json")
    qdrant_output_path = os.path.join(output_dir, f"{pdf_basename}_qdrant.json")

    # Dump only the necessary extracted content
    pipe_result.dump_md(md_writer, f"{pdf_basename}.md", image_output_dir)
    pipe_result.dump_content_list(md_writer, f"{pdf_basename}_content_list.json", image_output_dir)

    # Draw only the layout output
    pipe_result.draw_layout(os.path.join(output_dir, f"{pdf_basename}_layout.pdf"))

    # Save extracted titles and content in Qdrant-friendly format
    qdrant_data = {
        "markdown_file": markdown_file_path,
        "content_list": content_list_path,
        "titles": titles
    }

    with open(qdrant_output_path, 'w', encoding='utf-8') as json_file:
        json.dump(qdrant_data, json_file, ensure_ascii=False, indent=4)

    # Read the generated Markdown content
    try:
        with open(markdown_file_path, 'r', encoding='utf-8') as md_file:
            markdown_content = md_file.read()
    except FileNotFoundError:
        print(f"Error: Markdown file {markdown_file_path} not found.")
        return markdown_file_path, content_list_path, titles

    # Remove unwanted substrings using regex
    for pattern in unwanted_substrings:
        markdown_content = re.sub(pattern, '', markdown_content, flags=re.DOTALL | re.MULTILINE)

    # Optionally, you can clean up extra blank lines resulting from removals
    markdown_content = re.sub(r'\n{3,}', '\n\n', markdown_content)

    # Write the cleaned Markdown content back to the file
    with open(markdown_file_path, 'w', encoding='utf-8') as md_file:
        md_file.write(markdown_content)

    print(f"Extraction completed. Results are stored in {output_dir}")

    return markdown_file_path, content_list_path, titles

def parse_markdown_with_pages(markdown_content, content_list_path, titles):
    sections = []
    current_section = {"title": None, "content": "", "page_idx": None}
    content_list = []

    if os.path.exists(content_list_path):
        with open(content_list_path, 'r', encoding='utf-8') as file:
            content_list = json.load(file)

    page_map = {item["text"].strip(): item["page_idx"] for item in content_list if item.get("text_level") == 1}
    extracted_titles = {title_entry["text"].strip(): title_entry["page"] for title_entry in titles}

    def is_title_line(line):
        line = line.strip()

        if re.match(r'^#+\s*registro delle modifiche\b.*', line, re.IGNORECASE):
            return True
        
        # New regex pattern for flexible numeric title detection
        title_pattern = r'^\s*\d+(?:\.\d+)*(?:\.)?\s+.*$'

        # If line starts with '#'
        if line.startswith('#'):
            title_text = line.lstrip('#').strip()
            page_idx = extracted_titles.get(title_text) or page_map.get(title_text)

            # If on page 0 or 1, consider it a title regardless of numeric pattern
            if page_idx is not None and page_idx in [0, 1]:
                return True  # Exception: Treat as title on pages 0 and 1

            # Otherwise, check for numeric pattern after '#'
            if re.match(title_pattern, title_text):
                return True  # Valid title with numeric pattern

            return False  # Not a title if # but no numeric pattern and not on page 0 or 1

        # Check for numeric pattern without '#'
        if re.match(title_pattern, line):
            return True  # Line with numeric pattern is a title

        # Check against known title mappings
        return line in extracted_titles or line in page_map

    def clean_line(line):
        return re.sub(r'\s+', ' ', line).strip()

    # Process each line of markdown
    for raw_line in markdown_content.split("\n"):
        cleaned_line = clean_line(raw_line)
        if not cleaned_line:
            continue

        if is_title_line(cleaned_line):
            # Save the current section if valid
            if current_section["title"]:
                sections.append(current_section)

            # Extract title and page number
            title_text = cleaned_line.lstrip('#').strip()
            page_idx = extracted_titles.get(title_text) or page_map.get(title_text)

            # Start a new section
            current_section = {"title": title_text, "content": "", "page_idx": page_idx}
        else:
            current_section["content"] += cleaned_line + "\n"

    # Save the last section if it has a title
    if current_section["title"]:
        sections.append(current_section)

    return sections

    temp.py:
  import re
from typing import Any, List, Optional, Tuple, Dict
import pdfplumber
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTChar
import pandas as pd
import json
import os
import math
from pathlib import Path
import glob

# If you have a custom module for extracting titles, uncomment and ensure it's available
from extract_title import extract_titles_from_pdf

# Helper Functions

def _classify_table_type(data: list) -> str:
    """
    Classifies a table as either a 'Key-Value Table' or 'Row-by-Column Table' based on its structure.

    Args:
    - data (list): Table data as a list of lists.

    Returns:
    - str: The type of table ('Key-Value Table' or 'Row-by-Column Table').
    """
    # Check if the table has exactly two columns
    if len(data[0]) == 2:
        # Analyze the content of the first row
        first_row = data[0]
        second_row = data[1] if len(data) > 1 else []

        # If the first row looks like keys and the second row has longer text, classify as key-value
        if is_key_value_pair(first_row, second_row):
            return "Key-Value Table"

        # Otherwise, classify as row-by-column
        return "Row-by-Column Table"
    
    if len(data[0]) == 1:
        return "Row-Only Table"

    # Default classification for tables with more than two columns
    return "Row-by-Column Table"

def is_key_value_pair(first_row: list, second_row: list) -> bool:
    """
    Determines if the first and second rows form a key-value pair.

    Args:
    - first_row (list): The first row of the table.
    - second_row (list): The second row of the table.

    Returns:
    - bool: True if the rows represent a key-value pair, False otherwise.
    """
    # Check if the first column has shorter text and the second column has longer text
    if len(first_row[0]) < len(first_row[1]) and (not second_row or len(second_row[0]) < len(second_row[1])):
        return True
    return False

def _convert_row_column_table_to_json(df: pd.DataFrame, metadata: Optional[Dict] = None) -> List[Dict]:
    """
    Convert a row-by-column DataFrame to JSON.

    Args:
    - df (pd.DataFrame): The DataFrame to convert.
    - metadata (Optional[Dict]): Additional metadata.

    Returns:
    - List[Dict]: The JSON representation of the table.
    """
    return df.to_dict(orient='records')

def _convert_key_value_table_to_json(df: pd.DataFrame, metadata: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Convert a key-value DataFrame to JSON.

    Args:
    - df (pd.DataFrame): The DataFrame to convert.
    - metadata (Optional[Dict]): Additional metadata.

    Returns:
    - Dict[str, Any]: The JSON representation of the table.
    """
    return dict(zip(df['Key'], df['Value']))

def _convert_row_only_table_to_json(df: pd.DataFrame, metadata: Optional[Dict] = None) -> List[str]:
    """
    Convert a row-only DataFrame to JSON.

    Args:
    - df (pd.DataFrame): The DataFrame to convert.
    - metadata (Optional[Dict]): Additional metadata.

    Returns:
    - List[str]: The JSON representation of the table.
    """
    return df['Row Content'].tolist()

# Main Class Implementation

class PDFTableExtractor:
    def __init__(self, snap_tolerance: int = 10, footer_threshold: int = 50):
        """
        Initializes the PDFTableExtractor with the given parameters.

        Args:
        - snap_tolerance (int): Tolerance for snapping table lines.
        - footer_threshold (int): Threshold to identify footers.
        """
        self._snap_tolerance = snap_tolerance
        self._footer_threshold = footer_threshold

    def _text_extraction(self, element: LTTextContainer) -> Tuple[str, List[Tuple[str, float]]]:
        """
        Extracts text from a layout element, along with its formatting information.

        Args:
            element (LTTextContainer): A layout element from pdfminer.

        Returns:
            Tuple[str, List[Tuple[str, float]]]: Extracted text and a list of unique formatting attributes.
        """
        try:
            line_text = element.get_text()
            line_formats = []
            for text_line in element:
                if isinstance(text_line, LTTextContainer):
                    for character in text_line:
                        if isinstance(character, LTChar):
                            line_formats.append((character.fontname, character.size))
            format_per_line = list(set(line_formats))
            return line_text, format_per_line
        except Exception as e:
            raise Exception(f"Error in text extraction: {e}")

    def _merge_table(
        self, 
        page_num: int, 
        table: pdfplumber.table.Table, 
        prev_page_num: Optional[int], 
        prev_table: Optional[Dict[str, Any]], 
        page_content: pdfplumber.page.Page,
        text_between: bool
    ) -> Tuple[pd.DataFrame, str, bool, Optional[Dict[str, Any]]]:
        """
        Merges tables that should be combined based on the presence of intervening text.
        
        Args:
        - page_num (int): Current page number.
        - table (pdfplumber.table.Table): Current table object from pdfplumber.
        - prev_page_num (Optional[int]): Previous page number where a table was found.
        - prev_table (Optional[Dict[str, Any]]): Previous table data.
        - page_content (pdfplumber.page.Page): Current page content from pdfplumber.
        - text_between (bool): Indicator if there's text between the current and previous table.
        
        Returns:
        - Tuple containing:
            - pd.DataFrame: Merged or current table DataFrame.
            - str: Type of the table.
            - bool: Indicator if merged from previous.
            - Optional[Dict[str, Any]]: New previous table data.
        """
        try:
            # Extract table data as list of lists
            data_table = table.extract()
            if not data_table:
                return pd.DataFrame(), "Unknown", False, prev_table

        
            table_bbox_0 = round(table.bbox[0]) if table.bbox[0] % 1 < 0.5 else math.ceil(table.bbox[0])
            table_bbox_2 = round(table.bbox[2]) if table.bbox[2] % 1 < 0.5 else math.ceil(table.bbox[2])
            merged_from_previous = False

            if prev_table and not text_between:
                prev_table_bbox_0 = round(prev_table["bbox"][0]) if prev_table["bbox"][0] % 1 < 0.5 else math.ceil(prev_table["bbox"][0])

                if (
                    prev_page_num is not None
                    and prev_page_num == page_num - 1
                    and prev_table_bbox_0 == table_bbox_0
                ):
                    prev_data_table = prev_table["data"]
                    # Merge the tables by concatenating the data
                    data_table = prev_data_table + data_table
                    merged_from_previous = True

            df = pd.DataFrame(data_table)
            df = df.replace('\n', '  ', regex=True)

            table_type = _classify_table_type(df.values.tolist())
            if table_type == "Row-by-Column Table":
                header = df.iloc[0]
                df = df[1:]
                df.columns = header
                df = df.reset_index(drop=True)
            elif table_type == "Key-Value Table":
                df = pd.DataFrame(df.values.tolist(), columns=["Key", "Value"])
            elif table_type == "Row-Only Table":
                df = pd.DataFrame(df.values.tolist(), columns=["Row Content"])
            else:
                df = pd.DataFrame(data_table)

            if merged_from_previous:
                # Update the bounding box to cover both tables
                new_bbox = (
                    table_bbox_0, 
                    min(prev_table["bbox"][1], table.bbox[1]), 
                    table_bbox_2, 
                    max(prev_table["bbox"][3], table.bbox[3])
                )
                # Create a new dictionary to represent the merged table
                merged_table = {
                    "data": data_table,
                    "bbox": new_bbox
                }
                return df, table_type, merged_from_previous, merged_table
            else:
                # Create a new dictionary to represent the current table
                current_table = {
                    "data": data_table,
                    "bbox": table.bbox
                }
                return df, table_type, merged_from_previous, current_table

        except Exception as e:
            raise Exception(f"Error in merging tables: {e}")

    def _clean_text(self, text: str) -> str:
        """
        Cleans the extracted text by removing unwanted patterns.

        Args:
        - text (str): The raw extracted text.

        Returns:
        - str: Cleaned text.
        """
        pattern = re.compile(
            r"(?s)Documento e informazioni per circolazione e uso esclusivamente interni[\s\u00A0\n]*"
            r"Agenzia delle Entrate[\s\u00A0\n]*"
            r"Divisione Risorse\s*-\s*Direzione Centrale Tecnologie e Innovazione[\s\u00A0\n]*"
            r"Settore Infrastrutture e Sicurezza\s*-\s*Ufficio Sicurezza Informatica[\s\u00A0\n]*"
            r"Via Giorgione,\s*159\s*–\s*00147 Roma\s*–\s*Tel\.\s*06 50543028\s*-\s*email:\s*dc\.ti\.sicurezzainformatica@agenziaentrate\.it[\s\u00A0\n]*"
            r"(ID:\s*\w{2}-\d{2}\s*pag\.\s*\d+\s*di\s*\d+)?[\s\u00A0\n]*",
            re.MULTILINE
        )
        cleaned_text = pattern.sub("", text)
        return cleaned_text

    def _remove_text_from_page(self, page: pdfplumber.page.Page, cleaned_text: str) -> pdfplumber.page.Page:
        """
        Removes the cleaned text from the page content to isolate tables.

        Args:
        - page (pdfplumber.page.Page): The current page object from pdfplumber.
        - cleaned_text (str): The text to remove.

        Returns:
        - pdfplumber.page.Page: Modified page content with text removed.
        """
        # pdfplumber does not provide a direct method to remove text,
        # so this function can be customized based on specific requirements.
        # For simplicity, we'll return the page as-is.
        return page
    
    def _is_line_in_table(self, line_bbox, table_bboxes):
        lx0, ltop, lx1, lbottom = line_bbox
        for (tx0, ttop, tx1, tbottom) in table_bboxes:
            # Check overlap
            if not (lx1 < tx0 or lx0 > tx1) and not (lbottom < ttop or ltop > tbottom):
                return True
        return False
    
    def _group_words_into_lines(self, words, vertical_tolerance=3):
        """
        Groups words into lines based on their vertical positions.
        
        Args:
            words (list): A list of word dicts from extract_words().
            vertical_tolerance (float): Tolerance in vertical distance to consider words on the same line.

        Returns:
            list: A list of line dicts, each with keys: text, x0, x1, top, bottom.
        """
        words = sorted(words, key=lambda w: (round(w['top']), w['x0']))

        lines = []
        current_line = {
            'text': [],
            'x0': None,
            'x1': None,
            'top': None,
            'bottom': None
        }

        for w in words:
            if current_line['top'] is None:
                # Start a new line
                current_line['top'] = w['top']
                current_line['bottom'] = w['bottom']
                current_line['x0'] = w['x0']
                current_line['x1'] = w['x1']
                current_line['text'].append(w['text'])
            else:
                # Check if the current word is on the same line
                if abs(w['top'] - current_line['top']) <= vertical_tolerance:
                    # Same line
                    current_line['text'].append(w['text'])
                    # Update x1 and bottom if needed
                    current_line['x1'] = max(current_line['x1'], w['x1'])
                    current_line['bottom'] = max(current_line['bottom'], w['bottom'])
                else:
                    # Finish current line and start a new one
                    lines.append({
                        'text': ' '.join(current_line['text']),
                        'x0': current_line['x0'],
                        'x1': current_line['x1'],
                        'top': current_line['top'],
                        'bottom': current_line['bottom']
                    })
                    current_line = {
                        'text': [w['text']],
                        'x0': w['x0'],
                        'x1': w['x1'],
                        'top': w['top'],
                        'bottom': w['bottom']
                    }

        # Append the last line if exists
        if current_line['text']:
            lines.append({
                'text': ' '.join(current_line['text']),
                'x0': current_line['x0'],
                'x1': current_line['x1'],
                'top': current_line['top'],
                'bottom': current_line['bottom']
            })

        return lines

    def _extract_table_bboxes(self, page):
        """
        Extracts bounding boxes of tables in the page.
        
        Args:
            page (pdfplumber.page.Page): The PDF page object.

        Returns:
            list: A list of bounding boxes for tables.
        """
        table_bboxes = []
        try:
            tables = page.find_tables()
            for tbl in tables:
                table_bboxes.append(tbl.bbox)  # (x0, top, x1, bottom)
        except Exception as e:
            print(f"Error extracting table bboxes on page {page.page_number}: {e}")
        return table_bboxes

    def _is_line_in_table(self, line_bbox, table_bboxes):
        """
        Determines if a line is within any of the table bounding boxes.
        
        Args:
            line_bbox (tuple): The bounding box of the line (x0, top, x1, bottom).
            table_bboxes (list): A list of table bounding boxes.

        Returns:
            bool: True if the line is within a table, False otherwise.
        """
        for table_bbox in table_bboxes:
            tx0, ttop, tx1, tbottom = table_bbox
            lx0, ltop, lx1, lbottom = line_bbox
            # Check overlap
            if not (lx1 < tx0 or lx0 > tx1) and not (lbottom < ttop or ltop > tbottom):
                return True
        return False

    def _clean_text(self, text: str) -> str:
        """
        Cleans the extracted text by removing unwanted patterns and lines.

        Args:
            text (str): The raw extracted text.

        Returns:
            str: Cleaned text.
        """
        # Define a list of regex patterns to remove unwanted text
        patterns = [
            re.compile(r"Documento e informazioni per circolazione e uso esclusivamente interni", re.IGNORECASE),
            re.compile(r"Agenzia delle Entrate", re.IGNORECASE),
            re.compile(r"Divisione Risorse\s*-\s*Direzione Centrale Tecnologie e Innovazione", re.IGNORECASE),
            re.compile(r"Settore Infrastrutture e Sicurezza\s*-\s*Ufficio Sicurezza Informatica", re.IGNORECASE),
            re.compile(r"Via Giorgione,\s*159\s*–\s*00147 Roma\s*–\s*Tel\.\s*06\s*\d{8}\s*-\s*email:\s*[\w\.-]+@[\w\.-]+", re.IGNORECASE),
            re.compile(r"ID:\s*\w{2}-\d{2}\s*pag\.\s*\d+\s*di\s*\d+", re.IGNORECASE),
            re.compile(r"allegato a AGE\.AGEDC\d{3}\.REGISTRO UFFICIALE\.\d{7}\.\d{2}-\d{2}-\d{4}\.U", re.IGNORECASE),
            re.compile(r"_{5,}", re.IGNORECASE),  # Lines with multiple underscores
            re.compile(r"-{5,}", re.IGNORECASE),  # Lines with multiple hyphens
            # Add more patterns as needed
        ]

        cleaned_text = text
        for pattern in patterns:
            cleaned_text = pattern.sub("", cleaned_text)
        
        # Remove extra spaces and trim
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        return cleaned_text

    def _is_unwanted_line(self, text: str) -> bool:
        """
        Determines if a line is unwanted based on predefined keywords or patterns.

        Args:
            text (str): The text of the line.

        Returns:
            bool: True if the line is unwanted, False otherwise.
        """
        # Define a list of unwanted keywords or phrases
        unwanted_keywords = [
            "documento e informazioni per circolazione",  # Partial match
            "agenzia delle entrate",
            "divisione risorse",
            "direzione centrale tecnologie e innovazione",
            "settore infrastrutture e sicurezza",
            "ufficio sicurezza informatica",
            "via giorgione",
            "id:",
            "allegato a age.agedc",
            # Add more keywords as needed
        ]

        text_lower = text.lower()
        for keyword in unwanted_keywords:
            if keyword in text_lower:
                return True
        return False

    def _collect_all_lines(self, pdf_path):
        """
        Extracts all lines and titles from the PDF in a single pass.
        Returns a structure that can be processed afterward.
        """
        all_pages_data = []
        header_lines = {}
        footer_lines = {}

        with pdfplumber.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                words = page.extract_words()
                if not words:
                    # Even if no words, record empty lines for continuity
                    all_pages_data.append({
                        'page_number': page_number,
                        'lines': []
                    })
                    continue

                lines = self._group_words_into_lines(words)
                table_bboxes = self._extract_table_bboxes(page)

                # Filter out lines that appear inside tables and clean the text
                filtered_lines = []
                for line in lines:
                    line_bbox = (line['x0'], line['top'], line['x1'], line['bottom'])
                    if self._is_line_in_table(line_bbox, table_bboxes):
                        continue  # Skip lines inside tables

                    # Clean the text
                    cleaned_text = self._clean_text(line['text'])

                    if not cleaned_text:
                        continue  # Skip empty lines after cleaning

                    # Additional filtering based on content
                    if self._is_unwanted_line(cleaned_text):
                        continue  # Skip unwanted lines based on keywords

                    # Optionally, remove headers/footers based on position
                    page_height = page.height
                    top_margin = 50  # Adjust as needed
                    bottom_margin = 50  # Adjust as needed

                    if line['top'] < top_margin:
                        continue  # Likely a header
                    if (page_height - line['bottom']) < bottom_margin:
                        continue  # Likely a footer

                    # Append the cleaned and filtered line
                    filtered_lines.append({
                        'text': cleaned_text,
                        'x0': line['x0'],
                        'x1': line['x1'],
                        'top': line['top'],
                        'bottom': line['bottom']
                    })

                # Store data for this page
                all_pages_data.append({
                    'page_number': page_number,
                    'lines': filtered_lines,
                })

        return all_pages_data

    def _process_tables(self, file_path: str) -> List[dict]:
        """
        Processes tables in the PDF file, merging them based on the presence of intervening text.
        Stops merging if text is detected on the current page after processing a table.

        Args:
        - file_path (str): Path to the PDF file.

        Returns:
        - List[dict]: A list of JSON objects representing the processed tables.
        """
        try:
            all_pages_data = self._collect_all_lines(file_path)
            merged_tables = []
            prev_table = None
            prev_page_num = None

            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    page_text = page.extract_text()
                    cleaned_text = self._clean_text(page_text) if page_text else ""
                    page_content = self._remove_text_from_page(page, cleaned_text)
                    tables = page_content.find_tables(
                        table_settings={"snap_tolerance": self._snap_tolerance}
                    )

                    # Extract all text lines with their positions
                    lines = all_pages_data[page_num - 1]['lines'] if page_num - 1 < len(all_pages_data) else []

                    if not tables:
                        continue

                    # Sort tables top to bottom based on their 'top' position
                    sorted_tables = sorted(tables, key=lambda tbl: tbl.bbox[1])

                    for table_num, table in enumerate(sorted_tables):
                        try:
                            text_between = False  # Default to no intervening text

                            # Merge tables or process them individually
                            df, table_type, merged_from_previous, new_prev_table = self._merge_table(
                                page_num, table, prev_page_num, prev_table, page_content, text_between
                            )

                            # Always check for leftover text after processing a table
                            leftover_text_detected = False
                            for line in lines:
                                # Check for text below the current table or merged table
                                table_bottom = new_prev_table["bbox"][3] if merged_from_previous else table.bbox[3]
                                if line['top'] > table_bottom:  # Text below the table
                                    leftover_text_detected = True
                                    break

                            # If leftover text is detected, reset merging references
                            if leftover_text_detected:
                                prev_table = None
                                prev_page_num = None
                            else:
                                # Update references for merging if no leftover text
                                prev_table = new_prev_table
                                prev_page_num = page_num

                            if merged_from_previous and merged_tables:
                                merged_tables.pop()

                            # Append the table information
                            merged_tables.append({
                                "df": df,
                                "table_type": table_type,
                                "top": table.bbox[1] if hasattr(table, 'bbox') else None,
                                "page": page_num,
                            })

                        except Exception as e:
                            print(f"Error processing table on page {page_num}, table {table_num}: {e}")

                # Convert merged tables to JSON
                json_tables = self._convert_to_json(merged_tables)

        except Exception as e:
            raise Exception(f"Error in loading data from PDF: {e}")

        return json_tables

    def _convert_to_json(self, unique_tables: List[dict]) -> List[dict]:
        """
        Converts cleaned, unique tables into JSON format.

        Args:
        - unique_tables (List[dict]): A list of unique table dictionaries.

        Returns:
        - List[dict]: A list of JSON objects for the tables.
        """
        json_tables = []
        for table_entry in unique_tables:
            try:
                df = table_entry["df"]
                table_type = table_entry["table_type"]
                top = table_entry["top"]
                page_num = table_entry["page"]

                # Convert the DataFrame to JSON based on its type
                if table_type == "Row-by-Column Table":
                    table_json = _convert_row_column_table_to_json(df, None)
                elif table_type == "Key-Value Table":
                    table_json = _convert_key_value_table_to_json(df, None)
                elif table_type == "Row-Only Table":
                    table_json = _convert_row_only_table_to_json(df, None)
                else:
                    table_json = df.to_dict(orient='records')

                # Create the JSON object
                table_json_entry = {
                    "type": "table",
                    "top": top,
                    "structure": table_type,  # Changed from 'struttura' to 'structure' for consistency
                    "json": table_json,
                    "page_number": page_num,  # Changed from 'n_pag' to 'page_number' for clarity
                }
                json_tables.append(table_json_entry)
            except Exception as e:
                print(f"Error converting table on page {page_num} to JSON: {e}")

        return json_tables

def align_working_directory():
    """Aligns the working directory with the script's directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)


def find_files_in_data_directory(extension: str) -> list:
    """Finds files with the given extension in the 'data' directory."""
    return glob.glob(os.path.join(".", "data", f"*.{extension}"))

def main():
    """
    Main function to extract tables and text from all PDF files in the 'data' directory.
    """
    try:
        # Find all PDF files in the 'data' directory
        pdf_files = "extractor_api/media/extracted/IS-01 Elenco delle applicazioni informatiche validate in sicurezza_layout.pdf"

        # Process each PDF file
        for pdf_path in pdf_files:
            print(f"\nProcessing PDF: {pdf_path}\n")

            titles = extract_titles_from_pdf(pdf_path)

            try:
                # Create an instance of the PDF Table Extractor
                extractor = PDFTableExtractor()

                # Extract the tables and text from the PDF file
                json_content = extractor._process_tables(pdf_path)

                # Save the extracted JSON content to a file
                output_path = os.path.join("extracted", f"{Path(pdf_path).stem}_content.json")
                with open(output_path, "w", encoding="utf-8") as output_file:
                    json.dump(json_content, output_file, indent=4, ensure_ascii=False)
                print(f"Extracted content saved to: {output_path}")

            except Exception as e:
                print(f"Error processing {pdf_path}: {e}")

    except Exception as e:
        print(f"An error occurred during the execution of the main function: {e}")


if __name__ == "__main__":
    main()

    urls.py:
# extractor_api/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('extract/', views.extract_pdf, name='extract_pdf'),
    path('extract_to_qdrant/', views.extract_to_qdrant, name='extract_to_qdrant'),
]

  utils.py
import pdfplumber
from typing import List, Dict, Optional, Any
import math
import re
import pandas as pd
from collections import defaultdict
import fitz
from urllib.parse import urlparse
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def round_or_ceil(value: float) -> int:
    return math.floor(value)

def _group_words_into_lines(words, vertical_tolerance=3):
    words = sorted(words, key=lambda w: (round(w['top']), w['x0']))

    lines = []
    current_line = {
        'text': [],
        'x0': None,
        'x1': None,
        'top': None,
        'bottom': None
    }

    for w in words:
        if current_line['top'] is None:
            # Start a new line
            current_line['top'] = w['top']
            current_line['bottom'] = w['bottom']
            current_line['x0'] = w['x0']
            current_line['x1'] = w['x1']
            current_line['text'].append(w['text'])
        else:
            # Check if the current word is on the same line
            if abs(w['top'] - current_line['top']) <= vertical_tolerance:
                # Same line
                current_line['text'].append(w['text'])
                current_line['x1'] = max(current_line['x1'], w['x1'])
                current_line['bottom'] = max(current_line['bottom'], w['bottom'])
            else:
                # Finish current line and start a new one
                lines.append({
                    'text': ' '.join(current_line['text']),
                    'x0': current_line['x0'],
                    'x1': current_line['x1'],
                    'top': current_line['top'],
                    'bottom': current_line['bottom']
                })
                current_line = {
                    'text': [w['text']],
                    'x0': w['x0'],
                    'x1': w['x1'],
                    'top': w['top'],
                    'bottom': w['bottom']
                }

    # Append the last line if it exists
    if current_line['text']:
        lines.append({
            'text': ' '.join(current_line['text']),
            'x0': current_line['x0'],
            'x1': current_line['x1'],
            'top': current_line['top'],
            'bottom': current_line['bottom']
        })

    return lines

def _extract_table_bboxes(page):
    table_bboxes = []
    try:
        tables = page.find_tables()
        for tbl in tables:
            table_bboxes.append(tbl.bbox)  # (x0, top, x1, bottom)
    except Exception as e:
        print(f"Error extracting table bboxes on page {page.page_number}: {e}")
    return table_bboxes

def _is_line_in_table(line_bbox, table_bboxes):
    for table_bbox in table_bboxes:
        tx0, ttop, tx1, tbottom = table_bbox
        lx0, ltop, lx1, lbottom = line_bbox
        # Check overlap
        if not (lx1 < tx0 or lx0 > tx1) and not (lbottom < ttop or ltop > tbottom):
            return True
    return False

def _collect_all_lines(pdf_path):
    all_pages_data = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            words = page.extract_words()
            if not words:
                all_pages_data.append({'page_number': page_number, 'lines': []})
                continue

            lines = _group_words_into_lines(words)
            table_bboxes = _extract_table_bboxes(page)

            filtered_lines = []
            for line in lines:
                line_bbox = (line['x0'], line['top'], line['x1'], line['bottom'])
                if _is_line_in_table(line_bbox, table_bboxes):
                    continue

                cleaned_text = _clean_text(line['text'])
                if not cleaned_text:
                    continue

                if _is_unwanted_line(cleaned_text):
                    continue

                # Optional header/footer removal
                page_height = page.height
                top_margin = 50  
                bottom_margin = 50

                if line['top'] < top_margin:
                    continue  # Likely a header
                if (page_height - line['bottom']) < bottom_margin:
                    continue  # Likely a footer

                filtered_lines.append({
                    'text': cleaned_text,
                    'x0': line['x0'],
                    'x1': line['x1'],
                    'top': line['top'],
                    'bottom': line['bottom']
                })

            all_pages_data.append({
                'page_number': page_number,
                'lines': filtered_lines,
            })

    return all_pages_data

def _clean_text(text: str) -> str:
    patterns = [
        re.compile(r"Documento e informazioni per circolazione e uso esclusivamente interni", re.IGNORECASE),
        re.compile(r"Agenzia delle Entrate", re.IGNORECASE),
        re.compile(r"Divisione Risorse\s*-\s*Direzione Centrale Tecnologie e Innovazione", re.IGNORECASE),
        re.compile(r"Settore Infrastrutture e Sicurezza\s*-\s*Ufficio Sicurezza Informatica", re.IGNORECASE),
        re.compile(r"Via Giorgione,\s*159\s*–\s*00147 Roma\s*–\s*Tel\.\s*06\s*\d{8}\s*-\s*email:\s*[\w\.-]+@[\w\.-]+", re.IGNORECASE),
        re.compile(r"ID:\s*\w{2}-\d{2}\s*pag\.\s*\d+\s*di\s*\d+", re.IGNORECASE),
        re.compile(r"allegato a AGE\.AGEDC\d{3}\.REGISTRO UFFICIALE\.\d{7}\.\d{2}-\d{2}-\d{4}\.U", re.IGNORECASE),
        re.compile(r"_{5,}", re.IGNORECASE),
        re.compile(r"-{5,}", re.IGNORECASE),
    ]

    cleaned_text = text
    for pattern in patterns:
        cleaned_text = pattern.sub("", cleaned_text)
    
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

def _is_unwanted_line(text: str) -> bool:
    unwanted_keywords = [
        "documento e informazioni per circolazione",
        "agenzia delle entrate",
        "divisione risorse",
        "direzione centrale tecnologie e innovazione",
        "settore infrastrutture e sicurezza",
        "ufficio sicurezza informatica",
        "via giorgione",
        "id:",
        "allegato a age.agedc",
    ]
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in unwanted_keywords)

def _extract_links_from_table(
    table: pdfplumber.table.Table,
    words: List[Dict[str, Any]],
    hyperlinks: List[Dict[str, Any]],
    url_pattern: re.Pattern
) -> List[Dict]:
    row_tolerance = 5
    sorted_cells = sorted(table.cells, key=lambda c: (c[1], c[0]))

    rows = []
    current_row = []
    current_top = None
    for cell in sorted_cells:
        x0, top, x1, bottom = cell
        if current_top is None:
            current_top = top
            current_row = [cell]
        elif abs(top - current_top) <= row_tolerance:
            current_row.append(cell)
        else:
            rows.append(current_row)
            current_row = [cell]
            current_top = top
    if current_row:
        rows.append(current_row)

    link_records = []
    for row_idx, row_cells in enumerate(rows, start=1):
        row_cells_sorted = sorted(row_cells, key=lambda c: c[0])
        for col_idx, cell in enumerate(row_cells_sorted, start=1):
            x0, top, x1, bottom = cell
            cell_words = [
                w for w in words
                if (w["x0"] >= x0 and w["x1"] <= x1
                    and w["top"] >= top and w["bottom"] <= bottom)
            ]
            cell_text = ' '.join(w["text"] for w in cell_words)

            # 1) Regex-based link detection
            url_matches = url_pattern.findall(cell_text)
            for match in url_matches:
                link_records.append(match)

            # 2) Annotation-based link detection
            for link_annot in hyperlinks:
                link_uri = link_annot.get("uri")
                if not link_uri:
                    continue
                lx0, ltop, lx1, lbottom = link_annot["x0"], link_annot["top"], link_annot["x1"], link_annot["bottom"]
                # Overlap check
                if not (lx1 < x0 or lx0 > x1 or lbottom < top or ltop > bottom):
                    link_records.append(link_uri)

    return link_records

def _remove_text_from_page(page: pdfplumber.page.Page, cleaned_text: str) -> pdfplumber.page.Page:
    # Stub—no actual text removal in pdfplumber
    return page

def _merge_table(
    page_num: int, 
    table: pdfplumber.table.Table, 
    prev_page_num: Optional[int], 
    prev_table: Optional[Dict[str, Any]], 
    page_content: pdfplumber.page.Page,
    text_between: bool
):
    """
    Merges tables if:
      - There's a 'prev_table'
      - There's NO 'text_between'
      - The bounding-box x-range is the same
      - The previous page is exactly (current page - 1)
    Also handles link extraction for each table chunk.
    """
    try:
        data_table = table.extract()
        if not data_table:
            return pd.DataFrame(), False, prev_table

        url_pattern = re.compile(r'https?://\S+')
        words = page_content.extract_words()
        hyperlinks = page_content.hyperlinks or []
        table_links = _extract_links_from_table(table, words, hyperlinks, url_pattern)

        table_bbox_0 = round_or_ceil(table.bbox[0])
        table_bbox_2 = round_or_ceil(table.bbox[2])

        merged_from_previous = False

        if prev_table and not text_between:
            prev_table_bbox_0 = round_or_ceil(prev_table["bbox"][0])
            if (
                prev_page_num is not None
                and prev_page_num == (page_num - 1)
                and prev_table_bbox_0 == table_bbox_0
            ):
                # Merge: combine data and links from the previous table
                data_table = prev_table["data"] + data_table
                table_links = prev_table["links"] + table_links
                merged_from_previous = True

        if merged_from_previous:
            # Instead of inheriting the old page_start,
            # update it to the current page_num where the table links are found.
            page_start = page_num
            new_bbox = (
                table_bbox_0,
                min(prev_table["bbox"][1], table.bbox[1]),
                table_bbox_2,
                max(prev_table["bbox"][3], table.bbox[3])
            )
            merged_table = {
                "data": data_table,
                "bbox": new_bbox,
                "links": table_links,
                "page_start": page_start  # Updated to current page
            }
            df = pd.DataFrame(data_table)
            df = df.replace('\n', '  ', regex=True)
            return df, True, merged_table
        else:
            # For a brand new table, use the current page number.
            current_table = {
                "data": data_table,
                "bbox": table.bbox,
                "links": table_links,
                "page_start": page_num
            }
            df = pd.DataFrame(data_table)
            df = df.replace('\n', '  ', regex=True)
            return df, False, current_table

    except Exception as e:
        raise Exception(f"Error in merging tables: {e}")

def _extract_all_page_links(pdf_path: str) -> Dict[str, List[str]]:
    """
    Returns a dict of {page_number_as_str: [list_of_links]} for the entire PDF,
    ignoring those that might appear inside tables. We'll handle that filtering later.
    """
    from urllib.parse import urlparse
    exclude_schemes = {"mailto"}  # example: skip mailto: links

    page_links_dict = {}

    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            annots = page.annots or []
            # If you also want text-based link detection, you can do a regex pass
            # on the entire page text, though you already do so in your table code.

            page_links = []
            for annot in annots:
                uri = annot.get("uri")
                if uri:
                    parsed = urlparse(uri)
                    if parsed.scheme.lower() in exclude_schemes:
                        # skip excluded links
                        continue
                    page_links.append(uri)

            # Put the final page-links into the dictionary under their string page number
            if page_links:
                page_links_dict[str(page_idx)] = page_links

    return page_links_dict

def remove_duplicate_page_links(final_list):
    """
    Ensures that:
    - `page_links` do not contain duplicates within the same page.
    - `page_links` do not contain links already in `table_links` (on any table of the same or previous pages).
    - `page_links` are unique across pages (no duplicates in future pages).
    """
    used_table_links = set()  # Tracks all links used in table_links across all past pages
    assigned_page_links = set()  # Tracks page_links already assigned globally
    page_table_links = defaultdict(set)  # Tracks all table_links per page

    # Step 1: Collect all table_links across all pages (past & current)
    for entry in final_list:
        page_num = entry["page"]
        page_table_links[page_num].update(entry["table_links"])
        used_table_links.update(entry["table_links"])  # Store for cross-page checking

    # Step 2: Filter page_links ensuring no duplicates in past or current pages
    for entry in final_list:
        page_num = entry["page"]

        # Remove links that are in:
        # 1. Any table_links from the same page
        # 2. Any table_links from previous pages
        # 3. Any page_links already assigned globally
        filtered_links = [
            link for link in entry["page_links"]
            if link not in used_table_links  # Remove from ALL past table_links
            and link not in assigned_page_links  # Ensure no duplicates across pages
        ]

        # Store assigned page_links to prevent duplicates across pages
        assigned_page_links.update(filtered_links)

        # Assign the cleaned-up page_links
        entry["page_links"] = filtered_links

    return final_list

def extract_hyperlinks(pdf_path: str) -> List[Dict]:
    """
    Processes tables in the PDF file, merging them based on the presence of intervening text.
    Then returns a list of dicts, each describing one table's links:
      [
          {
              "page": <page_number>,
              "table_number": <table_sequence_number_on_that_page>,
              "table_links": [...],
              "page_links": [...]  # We'll fill this in at the end
          },
          ...
      ]
    """
    all_pages_data = _collect_all_lines(pdf_path)
    pages_dict: Dict[str, Dict[str, Any]] = {}
    prev_table = None
    prev_page_num = None

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text()
            cleaned_text = _clean_text(page_text) if page_text else ""
            page_content = _remove_text_from_page(page, cleaned_text)
            tables = page_content.find_tables()

            lines = (
                all_pages_data[page_num - 1]['lines']
                if (page_num - 1) < len(all_pages_data) else []
            )

            if not tables:
                continue

            # Sort top->bottom
            sorted_tables = sorted(tables, key=lambda tbl: tbl.bbox[1])

            for tbl in sorted_tables:
                try:
                    # text_between could be True if you implement that logic
                    text_between = False
                    df, merged_from_previous, new_prev_table = _merge_table(
                        page_num,
                        tbl,
                        prev_page_num,
                        prev_table,
                        page_content,
                        text_between
                    )

                    prev_table = new_prev_table
                    prev_page_num = page_num

                    # Identify the "start_page" for the (merged) table
                    start_page = str(new_prev_table["page_start"])  
                    links_for_this_table = new_prev_table["links"]

                    # Make sure the dictionary for this start_page exists
                    if start_page not in pages_dict:
                        pages_dict[start_page] = {"_counter": 0}

                    if merged_from_previous:
                        # Remove the last table we inserted for this page
                        last_label = f"table_{pages_dict[start_page]['_counter']}"
                        pages_dict[start_page].pop(last_label, None)
                        # DO NOT increment the counter here
                    else:
                        # Brand new table chunk => increment the counter
                        pages_dict[start_page]["_counter"] += 1

                    # Now reuse the existing or newly incremented counter
                    table_label = f"table_{pages_dict[start_page]['_counter']}"
                    pages_dict[start_page][table_label] = links_for_this_table

                except Exception as e:
                    print(f"Error processing table on page {page_num}: {e}")

    # 2) Remove the "_counter" from each page's dictionary
    for pg in pages_dict:
        pages_dict[pg].pop("_counter", None)

    # 3) Convert that dictionary to the list-of-dicts structure you want
    final_list: List[Dict] = []
    for page_str, table_info in pages_dict.items():
        page_int = int(page_str)  # convert "7" -> 7
        # table_info is a dict like { "table_1": [...], "table_2": [...] }
        for tbl_key, links_list in table_info.items():
            if tbl_key.startswith("table_"):
                table_num = int(tbl_key.split("_")[1])
                final_list.append({
                    "page": page_int,
                    "table_number": table_num,
                    "table_links": links_list,
                    "page_links": []  # We will fill this in a final step
                })

    # (Optional) Sort the final list by page, then table_number
    final_list.sort(key=lambda x: (x["page"], x["table_number"]))
    
    # 4) Gather page-level links (outside tables) using a separate function.
    page_links_dict = _extract_all_page_links(pdf_path)

    # Step 1: Ensure every page in `page_links_dict` exists in `final_list`
    existing_pages = {entry["page"] for entry in final_list}
    
    for page in page_links_dict:
        page_int = int(page)  # Convert string keys to int
        if page_int not in existing_pages:
            # If a page has no tables in `final_list`, add a placeholder entry
            final_list.append({
                "page": page_int,
                "table_number": 0,  # Placeholder value
                "table_links": [],
                "page_links": page_links_dict[page]
            })

    # Step 2: Assign page_links from `page_links_dict`
    for entry in final_list:
        page_str = str(entry["page"])
        if page_str in page_links_dict:
            # Filter out links that are already in table_links for the same entry
            entry["page_links"] = [
                link for link in page_links_dict[page_str]
                if link not in entry["table_links"]
            ]

    # Step 3: Run cleanup function to remove duplicates across pages
    final_list = remove_duplicate_page_links(final_list)

    # Step 4: Sort the final list by page and table_number
    final_list.sort(key=lambda x: (x["page"], x["table_number"]))

    # ----------------------------------------------------------------
    # 5) NOW add your embedding logic for each table
    # ----------------------------------------------------------------
    # Initialize your embedding model
    embedding_llm = HuggingFaceEmbedding(
        model_name="Snowflake/snowflake-arctic-embed-l-v2.0",
        trust_remote_code=True
    )

    # Loop through each table entry and create embeddings for table_links and page_links
    for entry in final_list:
        # 1) Embeddings for table_links
        if entry["table_links"]:
            # Concatenate all table_links into a single string
            text_to_embed = " ".join(str(link) for link in entry["table_links"])
            
            # HuggingFaceEmbeddings.embed_documents expects a list of documents.
            # We'll pass just one combined text, and then take the first embedding result.
            embeddings_list = embedding_llm.get_text_embedding(text_to_embed)
            entry["tbl_embed_vector"] = embeddings_list
        else:
            entry["tbl_embed_vector"] = []

        # 2) Embeddings for page_links
        if entry["page_links"]:
            # Concatenate all page_links into a single string
            text_to_embed = " ".join(str(link) for link in entry["page_links"])
            
            embeddings_list = embedding_llm.get_text_embedding(text_to_embed)
            entry["pg_embed_vector"] = embeddings_list
        else:
            entry["pg_embed_vector"] = []

    # Finally, return the list
    return final_list

def compute_cosine_similarity(vector_a: List[float], vector_b: List[float]) -> float:
    """
    Compute the cosine similarity between two vectors.
    """
    dot = sum(a * b for a, b in zip(vector_a, vector_b))
    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

def merge_markdown_and_tables(
    sections: list[dict],
    table_data: list[dict],
) -> list[dict]:
    """
    For each Markdown 'section':
      - If it contains a <table> tag, find the best-matching table (via cosine similarity).
        Then attach that table's `table_links` AND `page_links` at the end of the content.
      - If NO <table> is found, similarly find the best page-level match (via the `pg_embed_vector`)
        and attach that entry's `page_links`.

    Args:
        sections (list[dict]): Each section with {'content': str, 'page_idx': int}.
        table_data (list[dict]): List of tables with metadata, links, and embeddings.

    Returns:
        list[dict]: Updated sections with embedded table and page links at the end of the content.
    """
    embedding_llm = HuggingFaceEmbedding(
        model_name="Snowflake/snowflake-arctic-embed-l-v2.0",
        trust_remote_code=True
    )

    for sec in sections:
        sec["table_links"] = []
        sec["page_links"] = []
        content_text = sec.get("content", "")
        page_idx = sec.get("page_idx", -1)

        # If no text, skip it
        if not content_text:
            continue

        # Gather only the table_data entries that belong to this page
        page_entries = [tbl for tbl in table_data if tbl.get("page") == page_idx]

        # Check for <table> tags
        table_tags = re.findall(r'(?i)<table', content_text)

        if not table_tags:
            #
            # 1) If there's NO <table> tag, try to find the best chunk match
            #    among the page_entries based on `pg_embed_vector`.
            #
            if not page_entries:
                # No table_data for this page at all
                continue

            # Embed this chunk
            chunk_embedding = embedding_llm.get_text_embedding(content_text)

            best_page_entry = None
            best_score = -1.0
            for entry in page_entries:
                pg_emb = entry.get("pg_embed_vector", [])
                if pg_emb:
                    sim = compute_cosine_similarity(chunk_embedding, pg_emb)
                    if sim > best_score and sim >= 0.4:
                        best_page_entry = entry
                        best_score = sim

            if best_page_entry:
                # We found a good page-level match
                sec["page_links"] = best_page_entry.get("page_links", [])
            else:
                # If you prefer no links if below threshold:
                sec["page_links"] = []
                #
                # Or, if you still want to attach all page links when
                # none meets the similarity threshold, uncomment below:
                # sec["page_links"] = list({
                #     plink
                #     for tbl in page_entries
                #     for plink in tbl.get("page_links", [])
                # })
        else:
            #
            # 2) If there IS a <table> tag, keep your existing table logic as-is.
            #
            chunk_embedding = embedding_llm.get_text_embedding(content_text)
            best_table, best_table_score = None, -1.0

            # Find the best table match across *all* table_data
            for tbl in table_data:
                tbl_vector = tbl.get("tbl_embed_vector", [])
                if tbl_vector:
                    similarity = compute_cosine_similarity(chunk_embedding, tbl_vector)
                    if similarity > best_table_score and similarity >= 0.5:
                        best_table = tbl
                        best_table_score = similarity

            if best_table:
                # Attach both table-level and page-level links for the matched table
                sec["table_links"] = best_table.get("table_links", [])
                sec["page_links"] = best_table.get("page_links", [])
            else:
                # If no good table match, you can still do a best page link match
                # or leave it as you originally do. Here’s a minimal fallback:
                all_page_links = list({
                    plink
                    for tbl in page_entries
                    for plink in tbl.get("page_links", [])
                })
                sec["page_links"] = all_page_links

        #
        # 3) Append the discovered links to the Markdown content
        #
        links_md = ""
        if sec["table_links"]:
            links_md += "\n**La tabella contiene i seguenti link:**\n" \
                        + "\n".join(f"- {link}" for link in sec["table_links"])
        if sec["page_links"]:
            links_md += "\n**La pagina contiene i seguenti link:**\n" \
                        + "\n".join(f"- {link}" for link in sec["page_links"])

        if links_md.strip():
            sec["content"] += links_md

    return sections

def main():
    pdf_path = "extractor_api/media/extracted/IS-01 Elenco delle applicazioni informatiche validate in sicurezza_layout.pdf"
    final_data = extract_hyperlinks(pdf_path)

    import json
    print(json.dumps(final_data, indent=2))

    with open('table_data.json', 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()

  views.py
import os
import re
from django.http import JsonResponse
from rest_framework.decorators import (
    api_view, 
    permission_classes, 
    authentication_classes,
    parser_classes
)
from rest_framework.permissions import AllowAny
from rest_framework.parsers import MultiPartParser, FormParser
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from django.conf import settings
import json
from .services import extract_pdf_content, parse_markdown_with_pages
from .utils import extract_hyperlinks, merge_markdown_and_tables
from markdownify import markdownify as md

pdf_file_param = openapi.Parameter(
    name='file',
    in_=openapi.IN_FORM,
    description="PDF file to extract content from",
    type=openapi.TYPE_FILE,
    required=True
)

@swagger_auto_schema(
    method='post',
    manual_parameters=[pdf_file_param],
    consumes=['multipart/form-data'],
    responses={200: openapi.Response('Markdown content as JSON', schema=openapi.Schema(
        type=openapi.TYPE_OBJECT,
        properties={
            'content': openapi.Schema(type=openapi.TYPE_STRING, description='Extracted markdown content'),
        }
    ))},
)
@api_view(['POST'])
@permission_classes([AllowAny])
@authentication_classes([])
@parser_classes([MultiPartParser, FormParser])
def extract_pdf(request):
    """API endpoint to handle PDF extraction and return the Markdown content in JSON."""
    try:
        if 'file' not in request.FILES:
            return JsonResponse({"error": "No file uploaded"}, status=400)

        pdf_file = request.FILES['file']
        output_directory = os.path.join(settings.MEDIA_ROOT, 'extracted')
        os.makedirs(output_directory, exist_ok=True)

        input_pdf_path = os.path.join(output_directory, pdf_file.name)
        with open(input_pdf_path, 'wb') as f:
            for chunk in pdf_file.chunks():
                f.write(chunk)

        markdown_file_path, _, _ = extract_pdf_content(input_pdf_path, output_directory, method='txt', lang='it')  

        if not os.path.exists(markdown_file_path):
            return JsonResponse({"error": "Extraction failed"}, status=500)

        with open(markdown_file_path, 'r', encoding='utf-8') as md_file:
            content = md_file.read()

        return JsonResponse({"content": content})

    except Exception as e:
        return JsonResponse({"error": "An error occurred", "details": str(e)}, status=500)

@swagger_auto_schema(
    method='post',
    manual_parameters=[pdf_file_param],
    consumes=['multipart/form-data'],
    responses={200: openapi.Response('Extracted structured JSON with page numbers', schema=openapi.Schema(
        type=openapi.TYPE_OBJECT,
        properties={
            'sections': openapi.Schema(
                type=openapi.TYPE_ARRAY,
                items=openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'title': openapi.Schema(type=openapi.TYPE_STRING, description='Section title'),
                        'content': openapi.Schema(type=openapi.TYPE_STRING, description='Section content'),
                        'page_number': openapi.Schema(type=openapi.TYPE_INTEGER, description='Page number')
                    }
                )
            )
        }
    ))},
)
@api_view(['POST'])
@permission_classes([AllowAny])
@authentication_classes([])
@parser_classes([MultiPartParser, FormParser])
def extract_to_qdrant(request):
    """API endpoint to parse extracted markdown and return structured JSON with page numbers."""
    try:
        if 'file' not in request.FILES:
            return JsonResponse({"error": "No file uploaded"}, status=400)

        pdf_file = request.FILES['file']
        output_directory = os.path.join(settings.MEDIA_ROOT, 'extracted')
        os.makedirs(output_directory, exist_ok=True)

        input_pdf_path = os.path.join(output_directory, pdf_file.name)
        with open(input_pdf_path, 'wb') as f:
            for chunk in pdf_file.chunks():
                f.write(chunk)

        # Extract content from PDF
        markdown_file_path, content_list_path, titles = extract_pdf_content(
            input_pdf_path, 
            output_directory, 
            method='txt', 
            lang='it'
        )

        if not os.path.exists(markdown_file_path) or not os.path.exists(content_list_path):
            return JsonResponse({"error": "Extraction failed"}, status=500)

        with open(markdown_file_path, 'r', encoding='utf-8') as md_file:
            markdown_content = md_file.read()
        
        # Extract hyperlinks from PDF
        table_data = extract_hyperlinks(input_pdf_path)
        
        # Parse markdown content with page information
        parsed_sections = parse_markdown_with_pages(markdown_content, content_list_path, titles)
        
        # Merge markdown sections with table data
        merged_sections = merge_markdown_and_tables(parsed_sections, table_data)

        # ------------------- Filtering and Sorting -------------------

        # Define titles to exclude
        excluded_titles = ["Indice generale", "Registro delle modifiche", "Indice"]

        # Filter out unwanted sections
        filtered_sections = [
            section for section in merged_sections
            if not any(excl_title in section.get('title', '') for excl_title in excluded_titles)
            and section.get('page_idx') is not None
            and (section.get('content', '').strip() or section.get('page_idx') in [0, 1])
            # Allow empty content only if page_idx is 0 or 1
        ]
        # Sort the filtered sections by page_idx in ascending order
        sorted_sections = sorted(filtered_sections, key=lambda x: x['page_idx'])

        # Replace merged_sections with the filtered and sorted list
        merged_sections = sorted_sections

        # --------------------------------------------------------------
        #            Convert any HTML content to Markdown
        # --------------------------------------------------------------

        # Regex to detect basic HTML tags (you can adjust as needed)
        html_pattern = re.compile(r'<.*?>')

        for section in merged_sections:
            content = section.get('content', '')
            # If we detect HTML-like tags in the content, convert it
            if html_pattern.search(content):
                section['content'] = md(content)

        # --------------------------------------------------------------
        # Save the parsed sections to JSON file in the output directory
        # --------------------------------------------------------------

        json_output_path = os.path.join(
            output_directory, 
            f"{os.path.splitext(pdf_file.name)[0]}_qdrant.json"
        )
        
        with open(json_output_path, 'w', encoding='utf-8') as json_file:
            json.dump({"sections": merged_sections}, json_file, ensure_ascii=False, indent=4)

        return JsonResponse({
            "sections": merged_sections, 
            "json_file_path": json_output_path
        })

    except Exception as e:
        return JsonResponse({"error": "An error occurred", "details": str(e)}, status=500)
        

if __name__ == "__main__":
    main()
