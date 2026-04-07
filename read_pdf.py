import sys
try:
    from pypdf import PdfReader
    reader = PdfReader(sys.argv[1])
    with open("pdf_content_utf8.txt", "w", encoding="utf-8") as f:
        for page in reader.pages:
            f.write(page.extract_text() + "\n")
except Exception as e:
    print(f"Error reading PDF: {e}")
