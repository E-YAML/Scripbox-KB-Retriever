"""
scraper.py — Scripbox Knowledge Base Crawler (v2)
Crawls all articles from https://help.scripbox.com/support/solutions
and saves them to articles.json

Strategy:
  1. Fetch the main /support/solutions page and extract ALL folder links
  2. For each folder, fetch the folder page and extract ALL article links
  3. Scrape each article for its title and body text
  4. Save as structured JSON
"""

import requests
import json
import time
import re
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE_URL    = "https://help.scripbox.com"
SOLUTIONS_URL = f"{BASE_URL}/support/solutions"
OUTPUT_FILE = "articles.json"
DELAY       = 0.5   # seconds between requests

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

session = requests.Session()
session.headers.update(HEADERS)


# ── HTTP helpers ─────────────────────────────────────────────────────────────

def get(url: str) -> BeautifulSoup | None:
    """Fetch a URL and return a BeautifulSoup object, or None on error."""
    try:
        resp = session.get(url, timeout=20)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        print(f"  [WARN] {url} → {e}")
        return None


def get_all_links(soup: BeautifulSoup, pattern: str) -> list[str]:
    """Return all hrefs containing `pattern`, de-duped, absolute URLs."""
    seen = set()
    result = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if pattern in href:
            full = urljoin(BASE_URL, href)
            if full not in seen:
                seen.add(full)
                result.append(full)
    return result


# ── Discovery phase ──────────────────────────────────────────────────────────

# Known category IDs (from https://help.scripbox.com/support/solutions)
CATEGORY_IDS = [
    "3000005487",  # Account Opening
    "3000005484",  # Trouble with Login?
    "3000005431",  # What's new on Scripbox?
    "3000005481",  # Managing your Profile
    "3000005482",  # Investing and Withdrawing
    "3000005488",  # Reviewing your Portfolio
    "3000005499",  # Tracking Your Wealth
    "3000005500",  # Managing Your Investments
    "3000005526",  # Download mobile app / Contact Us
    "3000005657",  # Scripbox Plans
    "3000005788",  # Usage of Scripbox
]


def discover_folder_urls() -> dict[str, dict]:
    """
    Walk all category pages and return a mapping of
    folder_url → { category, folder_name }
    """
    print("=" * 60)
    print(f"Discovering folders from {len(CATEGORY_IDS)} categories...")
    print("=" * 60)

    folders = {}  # folder_url → {category, folder_name}

    for cat_id in CATEGORY_IDS:
        cat_url = f"{BASE_URL}/support/solutions/{cat_id}"
        soup = get(cat_url)
        time.sleep(DELAY)
        if not soup:
            continue

        # Category title
        cat_name_el = soup.select_one("h1, .category-title, .portal-category-header h1")
        cat_name = cat_name_el.get_text(strip=True) if cat_name_el else cat_id

        # Find all folder links on this category page
        folder_links = get_all_links(soup, "/solutions/folders/")
        for fl in folder_links:
            if fl not in folders:
                # Folder name from the link text
                for a in soup.find_all("a", href=True):
                    if fl.endswith(a["href"]) or a["href"] in fl:
                        fn = a.get_text(strip=True)
                        # Filter out "View all N" type links — we want the folder title
                        fn = re.sub(r"\s*\(\d+\)\s*$", "", fn).strip()
                        if fn and len(fn) > 3:
                            folders[fl] = {"category": cat_name, "folder": fn}
                            break
                if fl not in folders:
                    folders[fl] = {"category": cat_name, "folder": ""}

        print(f"  {cat_name}: {len(folder_links)} folders found")

    print(f"\nTotal folders discovered: {len(folders)}")
    return folders


def discover_article_urls(folders: dict[str, dict]) -> list[tuple[str, str, str]]:
    """
    For each folder, fetch its page and extract all article URLs.
    Returns list of (article_url, category_name, folder_name).
    """
    print("\nDiscovering articles from each folder...")
    print("=" * 60)

    discovered = []
    seen = set()

    for folder_url, meta in folders.items():
        cat_name    = meta["category"]
        folder_name = meta["folder"]

        # Folder pages may be paginated
        page_url = folder_url
        folder_arts = []
        while page_url:
            soup = get(page_url)
            time.sleep(DELAY)
            if not soup:
                break

            arts = get_all_links(soup, "/solutions/articles/")
            for a in arts:
                if a not in seen:
                    seen.add(a)
                    folder_arts.append(a)
                    discovered.append((a, cat_name, folder_name))

            # Check for next-page link
            next_el = soup.select_one("a[rel='next'], li.next a, .pagination .next a")
            if next_el and next_el.get("href"):
                page_url = urljoin(BASE_URL, next_el["href"])
            else:
                break

        print(f"  [{len(folder_arts):3d} articles] {cat_name} / {folder_name}")

    # ── Also pick up any articles linked directly from category pages ─────
    for cat_id in CATEGORY_IDS:
        cat_url = f"{BASE_URL}/support/solutions/{cat_id}"
        soup = get(cat_url)
        time.sleep(DELAY)
        if not soup:
            continue
        for a_url in get_all_links(soup, "/solutions/articles/"):
            if a_url not in seen:
                seen.add(a_url)
                discovered.append((a_url, "", ""))

    print(f"\nTotal unique article URLs: {len(discovered)}")
    return discovered


# ── Article scraping ──────────────────────────────────────────────────────────

def extract_article_content(soup: BeautifulSoup) -> str:
    """Extract clean body text from a Freshdesk article page."""
    def _clean_text(el: "BeautifulSoup") -> str:
        for tag in el.select(
            "script, style, nav, header, footer, "
            ".feedback-content, .article-rating, .rating, .helpful, "
            ".related-articles, .popular-articles, .sidebar, .toc"
        ):
            tag.decompose()
        lines = [ln.strip() for ln in el.get_text(separator="\n", strip=True).splitlines() if ln.strip()]
        # Drop common portal chrome / nav strings that sometimes leak into the body.
        noise_exact = {
            "All Articles",
            "Recent Searches",
            "Clear all",
            "No recent searches",
            "Popular Articles",
            "Articles",
            "Topics",
            "Tickets",
            "Sorry! not found",
            "View all",
        }
        lines = [ln for ln in lines if ln not in noise_exact]
        return "\n".join(lines)

    # Freshdesk often nests the real body inside a rich-text container like `.fr-view`.
    # We try specific selectors first to avoid picking up sidebars like "Popular Articles".
    candidates: list[BeautifulSoup] = []
    selectors = [
        "div.solution-article-content .fr-view",
        "div.solution-article-content",
        "div#article-body .fr-view",
        "div#article-body",
        "div.article-body .fr-view",
        "div.article-body",
        "div.article__body .fr-view",
        "div.article__body",
        "article .fr-view",
        "article",
        "main .fr-view",
        "main",
    ]
    for sel in selectors:
        el = soup.select_one(sel)
        if el:
            candidates.append(el)

    text = ""
    best = ""
    for el in candidates:
        t = _clean_text(el)
        if len(t) > len(best):
            best = t
    text = best

    # Broad fallback — strip nav/header/footer
    if not text:
        body = soup.find("body")
        if body:
            text = _clean_text(body)

    # Heuristic: if we accidentally captured a sidebar label, treat as empty.
    if text.strip().lower() in {"popular articles", "popular article"}:
        text = ""

    return text


def scrape_article(url: str, category: str = "", folder: str = "") -> dict | None:
    """Scrape a single article page and return a structured dict."""
    soup = get(url)
    time.sleep(DELAY)
    if not soup:
        return None

    # Article ID from URL
    m = re.search(r"/articles/(\d+)", url)
    article_id = m.group(1) if m else url

    # Title
    h1 = (
        soup.select_one("h1.article-title")
        or soup.select_one("h1.solution-article-title")
        or soup.select_one("h1")
    )
    title = h1.get_text(strip=True) if h1 else ""

    # Body
    content = extract_article_content(soup)

    # Remove leading duplicate title from content
    if content.startswith(title) and title:
        content = content[len(title):].strip()

    # Meta description
    meta = soup.find("meta", attrs={"name": "description"}) or soup.find(
        "meta", attrs={"property": "og:description"}
    )
    meta_desc = meta["content"].strip() if meta and meta.get("content") else ""

    return {
        "id":               article_id,
        "title":            title,
        "url":              url,
        "category":         category,
        "folder":           folder,
        "meta_description": meta_desc,
        "content":          content,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Phase 1: discover folders
    folders = discover_folder_urls()

    # Phase 2: discover articles inside each folder
    discovered = discover_article_urls(folders)

    # Phase 3: scrape each article
    print("\nScraping articles...")
    print("=" * 60)
    articles = []
    for i, (url, category, folder) in enumerate(discovered, 1):
        label = f"[{i:3d}/{len(discovered)}]"
        print(f"{label} {category or '?'} / {folder or '?'}")
        print(f"        {url}")
        art = scrape_article(url, category=category, folder=folder)
        if art:
            articles.append(art)
            char_count = len(art["content"])
            print(f"        ✓ \"{art['title']}\" ({char_count} chars)")
        else:
            print(f"        ✗ skipped")

    # Save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"✅ Done! Saved {len(articles)} articles → {OUTPUT_FILE}")
    print(f"   Next step: python build_index.py")


if __name__ == "__main__":
    main()
