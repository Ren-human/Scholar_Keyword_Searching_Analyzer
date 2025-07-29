import os
import requests
import fitz  # PyMuPDF
import serpapi
from collections import Counter
from dotenv import load_dotenv
from keybert import KeyBERT
import csv

kw_model = KeyBERT(model='all-MiniLM-L6-v2')

load_dotenv()
API_KEY = os.getenv("SERPAPI_KEY")
QUERY = "supportive partner"
NUM_RESULTS = 30

# setup
#kw_extractor = yake.KeywordExtractor(lan="en", n=1, top=5)

global_counter = Counter()

# query SerpAPI
params = {
    "api_key": API_KEY,
    "engine": "google_scholar",
    "q": QUERY,
    "num": NUM_RESULTS
}
search = serpapi.search(params)
results = search.as_dict().get("organic_results", [])

# Prepare CSV output file
output_path = "scholar_keywords.csv"
csv_file = open(output_path, mode="w", newline="", encoding="utf-8")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Title", "Top Keywords", "Used PDF", "PDF URL"])


for idx, item in enumerate(results):
    title = item.get("title", "")
    snippet = item.get("snippet", "")
    resources = item.get("resources", [])

    # Check for PDF resource
    pdf_url = None
    for r in resources:
        if r.get("file_format", "").lower() == "pdf":
            pdf_url = r.get("link")
            break

    if pdf_url:
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(pdf_url, headers=headers, timeout=10)

            content_type = resp.headers.get("Content-Type", "").lower()
            if "pdf" not in content_type or len(resp.content) < 50_000:
                raise ValueError("Invalid or tiny PDF")

            doc = fitz.open(stream=resp.content, filetype="pdf")
            text = "".join(page.get_text("text") for page in doc)

            lower_text = text.lower()
            if (
                    len(text.split()) < 100 or
                    "403 forbidden" in lower_text or
                    "enable javascript" in lower_text or
                    "researchgate" in lower_text or
                    "access denied" in lower_text
            ):
                raise ValueError("PDF contains junk, access-block, or wrapper text")

            used_pdf = True

        except Exception as e:
            print(f"Fallback to abstract for result {idx + 1}: {e}")
            text = title + ". " + snippet
            used_pdf = False
    else:
        text = title + ". " + snippet
        used_pdf = False

    # Preprocess and extract keywords
    text = text.lower().strip()
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 3),
        stop_words='english',
        top_n=5
    )
    top_keywords = [kw[0] for kw in keywords]  # ✅ extract only phrases

    # Update frequency count
    global_counter.update(top_keywords)  # ✅ only string list here

    print(f"{idx + 1}. {title}")
    print("   Keywords:", ", ".join(top_keywords))
    if used_pdf:
        print("   (used full-text PDF)")
    else:
        print("   (used snippet/abstract)")
    print()

    # Write to CSV
    csv_writer.writerow([
        title,
        ", ".join(top_keywords),
        "yes" if used_pdf else "no",
        pdf_url if pdf_url else ""
    ])

# Final summary
print("===== Top 5 Keywords Across All Documents =====")
for word, count in global_counter.most_common(5):
    print(f"{word}: {count} times")
csv_file.close()
print(f"\nCSV output saved to: {output_path}")


