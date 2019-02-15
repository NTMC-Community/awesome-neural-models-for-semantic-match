import re

import requests
import markdown

pattern = 'https?://(?:[-\w.]|(?:%[\da-fA-F]{2})).+?(?=">)'

with open('./README.md') as f:
	document = markdown.markdown(f.read())
	uris = re.findall(pattern, document)
	for uri in uris:
		r = requests.get(uri)
		print(f'uri {uri} with status: {r.status_code}')
