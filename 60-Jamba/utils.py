import os
from dotenv import load_dotenv, find_dotenv                                                                                                                                   
def load_env():
    _ = load_dotenv(find_dotenv())

def get_ai21_api_key():
    load_env()
    ai21_api_key = os.getenv("AI21_API_KEY")
    return ai21_api_key

import json, requests, re
from bs4 import BeautifulSoup

def get_full_filing_text(ticker: str) -> dict:
    headers = {'User-Agent': 'Company Name CompanyEmail@domain.com'}

    try:
        # Step 1: Get the CIK number
        cik_lookup_url = f"https://www.sec.gov/cgi-bin/browse-edgar?CIK={ticker}&owner=exclude&action=getcompany"
        response = requests.get(cik_lookup_url, headers=headers)
        response.raise_for_status()
        cik_match = re.search(r'CIK=(\d{10})', response.text)
        if not cik_match:
            return {"error": f"CIK not found for ticker {ticker}"}
        cik = cik_match.group(1)
        # Step 2: Get the latest 10-Q filing
        filing_lookup_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=10-Q&dateb=&owner=exclude&count=1"
        response = requests.get(filing_lookup_url, headers=headers)
        response.raise_for_status()
        doc_link_match = re.search(r'<a href="(/Archives/edgar/data/[^"]+)"', response.text)
        if not doc_link_match:
            return {"error": f"Latest 10-Q filing not found for ticker {ticker}"}
        doc_link = "https://www.sec.gov" + doc_link_match.group(1)
        # Step 3: Get the index page of the filing
        response = requests.get(doc_link, headers=headers)
        response.raise_for_status()

        # Step 4: Find the link to the actual 10-Q document
        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.find('table', class_='tableFile')
        if not table:
            return {"error": "Unable to find the document table"}

        for row in table.find_all('tr'):
            cells = row.find_all('td')
            if len(cells) >= 4 and '10-Q' in cells[3].text:
                doc_href = cells[2].a['href']
                full_doc_url = f"https://www.sec.gov{doc_href}"
                break
        else:
            return {"error": "10-Q document link not found in the index"}

        # Remove 'ix?doc=/' from the URL
        full_doc_url = full_doc_url.replace('ix?doc=/', '')
        # Step 5: Get the actual 10-Q document
        response = requests.get(full_doc_url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract text from the document
        full_text = ' '.join(soup.stripped_strings)
        # limit the full_text to 250000 characters
        full_text = full_text[:250000]

        return {
            "ticker": ticker,
            "filing_type": "10-Q",
            "filing_url": full_doc_url,
            "full_text": full_text,
            "full_text_length": len(full_text)
        }

    except requests.RequestException as e:
        return {"error": f"HTTP request failed: {str(e)}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}
