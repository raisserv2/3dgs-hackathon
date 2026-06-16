import site, pathlib
sp = site.getsitepackages()[0]
p = pathlib.Path(sp) / 'basicsr/data/degradations.py'
if p.exists():
    txt = p.read_text()
    txt = txt.replace('from torchvision.transforms.functional_tensor', 'from torchvision.transforms.functional')
    p.write_text(txt)
    print('Patched!')
else:
    print('File not found.')
