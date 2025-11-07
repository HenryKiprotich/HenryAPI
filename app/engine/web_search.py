import requests
from bs4 import BeautifulSoup
from typing import List

FINANCIAL_SITES = [
    "https://www.investopedia.com/loan-4014682",
    "https://www.bankrate.com/loans/",
    "https://www.nerdwallet.com/best/loans/personal-loans",
    "https://www.lendingtree.com/personal/",
    "https://www.federalreserve.gov/releases/g19/current/"
]

def fetch_site_text(url: str) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        # Remove scripts/styles
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        return text[:5000]  # Limit to 5000 chars for brevity
    except Exception as e:
        return f"Could not fetch {url}: {e}"

def search_financial_trends(question: str) -> List[str]:
    """
    Fetches and returns relevant text snippets from financial sites.
    """
    results = []
    for url in FINANCIAL_SITES:
        text = fetch_site_text(url)
        # Simple keyword filter (could be improved with embeddings)
        if any(word in text.lower() for word in question.lower().split()):
            results.append(f"Source: {url}\n{text[:1000]}")
        else:
            results.append(f"Source: {url}\n{text[:500]}")
    return results