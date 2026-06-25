import urllib, urllib.request
import pandas as pd
import parquet as pq
import time
from io import StringIO

base_url = 'http://export.arxiv.org/api/query?';
search_query = urllib.parse.quote("cat:cs.CY submittedDate=[20260101TO20260626]")
i = 0
results_per_iteration = 1000
wait_time = 5
print('Searching arXiv for %s' % search_query)

while (i < 10000): #stop requesting when reaching 10000 papers
   print("Results %i - %i" % (i,i+results_per_iteration))
    
   query = 'search_query=%s&start=%i&max_results=%i&sortBy=submittedDate&sortOrder=descending' % (search_query, i,results_per_iteration)
   data = urllib.request.urlopen(base_url+query)
   result = data.read().decode('utf-8', 'ignore')
   i += results_per_iteration
   print(data.code)
   df = pd.read_xml(StringIO(result))
   df.to_parquet("result"+str(i)+".parquet")

   time.sleep(wait_time)




