"""
patch_articles.py — Add missing articles to articles.json and rebuild index

The initial scraper may have missed some articles that are in folders
not directly linked from category pages, or articles added after the
initial scrape.

This script:
1. Loads existing articles.json
2. Scrapes any additional article URLs you provide (or from search)
3. Deduplicates and saves back to articles.json
4. Rebuilds the ChromaDB index

Usage:
    python patch_articles.py
"""

import json
import time
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE_URL = "https://help.scripbox.com"
OUTPUT_FILE = "articles.json"
DELAY = 0.5

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}

session = requests.Session()
session.headers.update(HEADERS)


# ── Additional search terms to discover missing articles ──────────────────────
SEARCH_TERMS = [
    "minor to major",
    "nominee",
    "guardian",
    "joint holder",
    "bank account change",
    "email change",
    "address update",
    "dividend",
    "SIP",
    "lumpsum",
    "ELSS",
    "tax",
    "NRI",
    "rebalance",
    "switch fund",
    "redeem",
    "capital gains",
    "statement",
    "folio",
    "mandate",
    "ekyc",
    "aadhaar",
    "pan card",
    "mobile number",
    "password",
]

# ── Manually known missing article URLs ── add any you know about here ────────
MANUAL_URLS = [
    "https://help.scripbox.com/support/solutions/articles/3000129111-my-child-has-recently-become-a-major-could-you-please-advise-if-any-changes-are-needed-for-his-inves",
    "https://help.scripbox.com/support/solutions/articles/3000121553-i-couldn-t-place-the-instructions-for-one-of-minor-family-member-why-",
    "https://help.scripbox.com/support/solutions/articles/3000121644-why-do-i-need-to-invest-using-my-child-s-bank-account-for-my-child-s-investments-",
    "https://help.scripbox.com/support/solutions/articles/3000121642-how-do-i-add-an-investor-my-child-to-my-scripbox-investment-account-",
    "https://help.scripbox.com/support/solutions/articles/3000128625-how-do-i-add-an-investor-my-child-to-my-scripbox-investment-account-",
]


def get(url):
    try:
        r = session.get(url, timeout=20)
        r.raise_for_status()
        return BeautifulSoup(r.text, "html.parser")
    except Exception as e:
        print(f"  [WARN] {url}: {e}")
        return None


def search_articles(term):
    """Use Scripbox search to find article URLs for a given term."""
    url = f"{BASE_URL}/support/search?term={requests.utils.quote(term)}"
    soup = get(url)
    time.sleep(DELAY)
    if not soup:
        return []
    urls = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/solutions/articles/" in href:
            full = urljoin(BASE_URL, href)
            if full not in urls:
                urls.append(full)
    return urls


def extract_content(soup):
    selectors = [
        "div.article-body", "div.article__body", "div#article-body",
        "div.solution-article-content", "div.article-content",
    ]
    el = None
    for sel in selectors:
        el = soup.select_one(sel)
        if el:
            break
    if el:
        for tag in el.select("script, style, .feedback-content, .article-rating"):
            tag.decompose()
        text = el.get_text(separator="\n", strip=True)
    else:
        body = soup.find("body")
        if body:
            for tag in body.select("nav, header, footer, script, style"):
                tag.decompose()
            text = body.get_text(separator="\n", strip=True)
        else:
            text = ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines)


def scrape_one(url, category="", folder=""):
    soup = get(url)
    time.sleep(DELAY)
    if not soup:
        return None
    m = re.search(r"/articles/(\d+)", url)
    article_id = m.group(1) if m else url
    h1 = soup.select_one("h1.article-title") or soup.select_one("h1")
    title = h1.get_text(strip=True) if h1 else ""
    content = extract_content(soup)
    if content.startswith(title) and title:
        content = content[len(title):].strip()
    meta = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", attrs={"property": "og:description"})
    meta_desc = meta["content"].strip() if meta and meta.get("content") else ""
    # Try to detect category/folder from breadcrumbs
    if not category:
        bc = soup.select(".breadcrumb a, .breadcrumbs a, nav a")
        crumbs = [b.get_text(strip=True) for b in bc if b.get_text(strip=True)]
        if len(crumbs) >= 2:
            category = crumbs[-2] if len(crumbs) >= 2 else ""
        if len(crumbs) >= 1:
            folder = crumbs[-1] if len(crumbs) >= 1 else ""
    return {
        "id": article_id,
        "title": title,
        "url": url,
        "category": category,
        "folder": folder,
        "meta_description": meta_desc,
        "content": content,
    }


def main():
    # Load existing articles
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        existing = json.load(f)
    existing_ids = {a["id"] for a in existing}
    existing_urls = {a["url"] for a in existing}
    print(f"Existing articles: {len(existing)}")

    # Collect candidate URLs from search + manual list
    candidate_urls = set(MANUAL_URLS)

    print(f"\nSearching KB for {len(SEARCH_TERMS)} terms...")
    for term in SEARCH_TERMS:
        urls = search_articles(term)
        candidate_urls.update(urls)
        print(f"  '{term}' → {len(urls)} results")

    # Filter to only new ones
    new_urls = [u for u in candidate_urls if u not in existing_urls]
    print(f"\nNew article URLs found: {len(new_urls)}")

    # Scrape new articles
    new_articles = []
    for i, url in enumerate(new_urls, 1):
        print(f"[{i}/{len(new_urls)}] {url}")
        art = scrape_one(url)
        if art:
            if art["id"] not in existing_ids:
                new_articles.append(art)
                existing_ids.add(art["id"])
                print(f"  + \"{art['title']}\"")
            else:
                print(f"  ~ already exists (same ID)")
        else:
            print(f"  ✗ skipped")

    if not new_articles:
        print("\nNo new articles to add. Index is up to date!")
        return

    # Merge and save
    all_articles = existing + new_articles
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_articles, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(all_articles)} articles to {OUTPUT_FILE} (added {len(new_articles)} new)")

    # Rebuild index
    print("\nRebuilding vector index...")
    import subprocess, sys
    result = subprocess.run([sys.executable, "build_index.py"], check=True)
    print("Index rebuilt successfully!")


if __name__ == "__main__":
    main()
