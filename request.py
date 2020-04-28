import requests

url = 'http://localhost:5000/results'
r = requests.post(url,json={'PROD_CD':5, 'SLSMAN_CD':8, 'PLAN_MONTH':11,'TARGET_IN_EA':200})

print(r.json())

