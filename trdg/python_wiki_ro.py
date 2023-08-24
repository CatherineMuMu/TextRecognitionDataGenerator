import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET

# 目标维基百科页面的URL
url = 'https://ro.wikipedia.org/wiki/Pagina_principala'

# 发起HTTP请求并获取页面内容
response = requests.get(url)
html = response.text

# 使用BeautifulSoup解析页面内容
soup = BeautifulSoup(html, 'html.parser')

# 创建XML根节点
root = ET.Element("pages")

# 添加当前页面内容到XML根节点
page = ET.SubElement(root, "page")
ET.SubElement(page, "url").text = url
ET.SubElement(page, "content").text = html

# 获取页面中的所有链接
links = soup.find_all('a')

# 遍历链接并生成XML子节点
for link in links:
    # 获取链接的URL
    href = link.get('href')

    # 剔除不符合要求的链接
    if href is not None and href.startswith('/wiki/'):
        # 拼接完整的链接
        link_url = 'https://ro.wikipedia.org' + href

        # 发起HTTP请求并获取链接页面内容
        response = requests.get(link_url)
        link_html = response.text

        # 创建XML子节点
        page = ET.SubElement(root, "page")
        ET.SubElement(page, "url").text = link_url
        ET.SubElement(page, "content").text = link_html

# 创建XML树
xml_tree = ET.ElementTree(root)

# 将XML树写入文件
xml_tree.write("wikipedia_ro.xml", encoding='utf-8', xml_declaration=True)
