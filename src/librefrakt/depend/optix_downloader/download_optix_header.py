from io import StringIO
import re
import html
from html.parser import HTMLParser
import ssl
import urllib.request
import sys
import os
import time

ssl_ctx = ssl.create_default_context()
ssl_ctx.check_hostname = False
ssl_ctx.verify_mode = ssl.CERT_NONE

def download_page(url: str):
    with urllib.request.urlopen(url, context=ssl_ctx) as response:
        return response.read().decode('utf-8')

def last_modified(url: str):
    import email.utils
    req = urllib.request.Request(url, method='HEAD')
    with urllib.request.urlopen(req, context=ssl_ctx) as response:
        return time.mktime(email.utils.parsedate(response.headers["Last-Modified"]))

def make_source_url(filename: str):
    return f'https://raytracing-docs.nvidia.com/optix8/api/{filename.replace("_", "__").replace(".", "_8")}_source.html'

class tag_stripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.text = StringIO()
    
    def handle_data(self, data):
        self.text.write(data)

    def get_data(self):
        return self.text.getvalue()

header = sys.argv[1]
header_url = make_source_url(header)

if os.path.exists(header):
    if os.path.getmtime(header) > last_modified(header_url):
        sys.exit(0)

page = download_page(make_source_url(header)).split('\n')
with open(header, "w") as f:
    s = tag_stripper()
    for line in page:
        if not "<div class=\"line\">" in line:
            continue
        
        line_start = line.find("<div class=\"line\">")
        line = line[line_start:]
        line = re.sub(r'<span class="lineno">.*?</span>', '', line)

        s.feed(line + '\n')
    
    f.write(s.get_data())