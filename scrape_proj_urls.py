import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import regex as re

def get_page(url):
    page = requests.get(url, verify=False)
    if page.status_code != 200:
        print('Failed to retrieve', url)
    # time delay
    time.sleep(2)
    # return BeautifulSoup object with content
    return BeautifulSoup(page.content, 'html.parser')

def find_project_pages(url):
    projects = set()
    while url:
        print("Scraping:", url)
        soup = get_page(url)

        # Find project links
        project_containers = soup.find_all('div', class_='row row-eq-height')
        for project in project_containers:
            project_link = project.find('a', id=re.compile(r'^resultProjectLink'), href=True)
            if project_link:
                href = project_link['href']
                projects.add(href)
                print(href)

        # Find next button and repeat until next button is disabled
        next_button = soup.find('a', class_='next')
        if next_button and next_button.get('href') not in [None, '#'] and 'disabled' not in next_button.get('class', []):
            href = next_button['href']
            if href.startswith('?'):
                href = '/search/project' + href
            url = "https://gtr.ukri.org" + href
            print("NEXT PAGE:", url)
        else:
            print("No next page — stopping.")
            url = None

    return projects

first_url = "https://gtr.ukri.org/search/project?term=circular+economy&fetchSize=25&selectedSortableField=&selectedSortOrder=&fields=pro.gr%2Cpro.t%2Cpro.a%2Cpro.orcidId%2Cper.fn%2Cper.on%2Cper.sn%2Cper.fnsn%2Cper.orcidId%2Cper.org.n%2Cper.pro.t%2Cper.pro.abs%2Cpub.t%2Cpub.a%2Cpub.orcidId%2Corg.n%2Corg.orcidId%2Cacp.t%2Cacp.d%2Cacp.i%2Cacp.oid%2Ckf.d%2Ckf.oid%2Cis.t%2Cis.d%2Cis.oid%2Ccol.i%2Ccol.d%2Ccol.c%2Ccol.dept%2Ccol.org%2Ccol.pc%2Ccol.pic%2Ccol.oid%2Cip.t%2Cip.d%2Cip.i%2Cip.oid%2Cpol.i%2Cpol.gt%2Cpol.in%2Cpol.oid%2Cprod.t%2Cprod.d%2Cprod.i%2Cprod.oid%2Crtp.t%2Crtp.d%2Crtp.i%2Crtp.oid%2Crdm.t%2Crdm.d%2Crdm.i%2Crdm.oid%2Cstp.t%2Cstp.d%2Cstp.i%2Cstp.oid%2Cso.t%2Cso.d%2Cso.cn%2Cso.i%2Cso.oid%2Cff.t%2Cff.d%2Cff.c%2Cff.org%2Cff.dept%2Cff.oid%2Cdis.t%2Cdis.d%2Cdis.i%2Cdis.oid%2Ccpro.rtpc%2Ccpro.rcpgm%2Ccpro.hlt&type=&selectedFacets=c3RhdHVzfENsb3NlZHxzdHJpbmc%3D"
proj_urls = find_project_pages(first_url)

# Write to file
with open("first_proj_urls.txt", "w") as f:
    for url in proj_urls:
        f.write(url + "\n")