import urllib, urllib.request
import pandas as pd
import parquet as pq
import time
from io import StringIO

base_url = 'http://export.arxiv.org/api/query?';
search_query = urllib.parse.quote("ti:computers and society")
i = 0
results_per_iteration = 1000
wait_time = 5
papers = []
year = ""  
print('Searching arXiv for %s' % search_query)

while (year != "2025"): #stop requesting when papers date reach 2018
   print("Results %i - %i" % (i,i+results_per_iteration))
    
   query = 'search_query=%s&start=%i&max_results=%i&sortBy=submittedDate&sortOrder=descending' % (search_query, i,results_per_iteration)
   data = urllib.request.urlopen(base_url+query)
   result = data.read().decode('utf-8', 'ignore')
   i += results_per_iteration
   print(data.code)
   df = pd.read_xml(StringIO(result))
   df.to_parquet("result"+str(i)+".parquet")

   time.sleep(wait_time)




"""
search_terms = 'all:computer-science-'+'-AND-'+'2026-01'
url = f'http://export.arxiv.org/api/query?search_query={search_terms}&start=0&max_results=1000'
data = urllib.request.urlopen(url)

df = pd.read_xml(StringIO(result))
df.to_parquet("result.parquet")
pd.read_parquet("result.parquet")


print(result)

print(pd.read_parquet)
"""
