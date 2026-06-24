import urllib, urllib.request
import pandas as pd
import parquet as pq
from io import StringIO


search_terms = 'all:computer-science-'+'-AND-'+'2026'
url = f'http://export.arxiv.org/api/query?search_query={search_terms}&start=0&max_results=1000'
data = urllib.request.urlopen(url)
result = data.read().decode('utf-8', 'ignore')
df = pd.read_xml(StringIO(result))
df.to_parquet("result.parquet")
pd.read_parquet("result.parquet")


print(result)
print(data.code)
print(pd.read_parquet)
#print(len(feed.entries))
#for entry in feed.entries:
#   print(entry.title)
"""
try:
   with open("arxiv.txt", "w") as file:
      file.write(result)
except UnicodeEncodeError:
      print("UnicodeEncodeError")
"""


