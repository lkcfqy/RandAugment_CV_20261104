import pathlib, re, urllib.parse
from html.parser import HTMLParser
from fetch_public import ROOT, fetch
headers=(ROOT/'sources/weight-head.txt.headers').read_text()
assert 'Content-Length: 691901857' in headers
assert 'aa587e4dde3f2de265f33593bc9de6c4' in headers.lower()
fetch('https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/groundingdino_swint_ogc_mmdet-822d7e9d.pth','assets/groundingdino_swint_ogc_mmdet-822d7e9d.pth',691901857,1200)
p=fetch('https://drive.usercontent.google.com/download?id=1Y7NEUZ4IJjHEvbmHuj0NcfnoVhvMasaW&export=download','assets/FISH-initial-response',800*1024**2,1200)
with p.open('rb') as f: signature=f.read(2)
if signature == b'\x1f\x8b':
 p.rename(ROOT/'assets/FISH.tar.gz')
else:
 if p.stat().st_size>1024**2: raise RuntimeError('Unexpected Drive response, stop')
 class Form(HTMLParser):
  def __init__(self): super().__init__();self.action=None;self.params={}
  def handle_starttag(self,tag,attrs):
   d=dict(attrs)
   if tag=='form': self.action=d.get('action')
   if tag=='input' and d.get('name'): self.params[d['name']]=d.get('value','')
 form=Form();form.feed(p.read_text())
 assert form.action=='https://drive.usercontent.google.com/download',form.action
 assert form.params.get('id')=='1Y7NEUZ4IJjHEvbmHuj0NcfnoVhvMasaW'
 assert set(form.params).issubset({'id','export','confirm','uuid'})
 url=form.action+'?'+urllib.parse.urlencode(form.params)
 fetch(url,'assets/FISH.tar.gz',800*1024**2,1200)
