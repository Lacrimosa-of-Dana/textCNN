import requests
f = open('./resources/train.csv', 'w')
resp = requests.get('http://download.kesci.com/pnu3b0af1/train.csv').text
for line in resp.split('\r\n'):
    f.write(line.lower() + '\n')
