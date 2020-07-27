import re
import time
import requests
import markdown

pattern = 'https?://(?:[-\w.]|(?:%[\da-fA-F]{2})).+?(?=">)'

with open('./README.md') as f:
	document = markdown.markdown(f.read())
	uris = re.findall(pattern, document)
	print(len(uris))
	for uri in uris:
		try:
			r = requests.get(uri)
			print(f'uri {uri} with status: {r.status_code}')
		except:
			print("Connection refused by the server..")
			time.sleep(5)
			uris.append(uri)