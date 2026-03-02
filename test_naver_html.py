import requests
from bs4 import BeautifulSoup

url = "https://finance.naver.com/item/news_news.nhn?code=005930&page=1"
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
resp = requests.get(url, headers=headers)
resp.encoding = 'euc-kr'
html = resp.text
print(f"Status Code: {resp.status_code}")
print(f"HTML Length: {len(html)}")

soup = BeautifulSoup(html, "html.parser")
print("\nFirst 500 chars of HTML:")
print(html[:500])

titles = soup.select(".title a")
print(f"\nFound {len(titles)} titles.")
if titles:
    print(titles[0].text)
    
dates = soup.select(".date")
print(f"\nFound {len(dates)} dates.")
if dates:
    print(dates[0].text)

# Also try another selector if .title a is wrong.
print("\nAlternative selectors:")
alt_titles = soup.select(".tit")
print(f"Found {len(alt_titles)} via .tit")
