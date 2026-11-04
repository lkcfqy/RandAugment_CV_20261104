import pathlib,hashlib,json,os
p=pathlib.Path('/content/deepfish_resume25_5fa8916620');p.mkdir(exist_ok=False);b=(b'DeepFish public non-sensitive persistence marker v1\n'*200000)[:8388608];f=(p/'marker.bin').open('wb');f.write(b);f.flush();os.fsync(f.fileno());f.close();print('DEEPFISH_PROBE '+json.dumps({'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}))
