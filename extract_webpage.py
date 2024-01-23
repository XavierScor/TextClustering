import csv
import requests
import re
from bs4 import BeautifulSoup

def write_webpage_text(url, out_file_name):
    if url[-3:] == "pdf":
        with open(out_file_name[0:12] + "PDF_file_" + out_file_name[12:], 'w', encoding='utf-8', errors='ignore') as out_file:
            out_file.write("waiting for pdf content\n")
            out_file.write(url)
    else:
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
            req = requests.get(url = url, headers=headers)
            html = req.text
            soup = BeautifulSoup(html, "html.parser")
            if not soup.find_all(re.compile("^h[1-6]$|^p")):
                with open(out_file_name[0:12] + "Empty_file_" + out_file_name[12:], 'w', encoding='utf-8', errors='ignore') as out_file:
                    out_file.write("waiting for content\n")
                    out_file.write(url)
            else:
                with open(out_file_name, 'w', encoding='utf-8', errors='ignore') as out_file:
                    for content in soup.find_all(re.compile("^h[1-6]$|^p")):
                        out_file.write(content.text)
        except:
            with open(out_file_name[0:12] + "Failed_file_" + out_file_name[12:], 'w', encoding='utf-8', errors='ignore') as out_file:
                out_file.write("waiting for content\n")
                out_file.write(url)
            pass

with open('data/clean_quoted_dataset.csv', newline = '', encoding='utf-8') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    index = 0
    for row in spamreader:
        if index >= 1:
            print(str(index) + " " + row[3])
            write_webpage_text(row[3], "webpageText/webpage_" + str(row[0]) + ".txt")
        index = index + 1