import struct
path = 'data_ready/aeroplane/sparse/0/cameras.bin'
with open(path, 'rb') as f:
    print('File size:', f.seek(0,2))
    f.seek(0)
    try:
        num = struct.unpack('<Q', f.read(8))[0]
        print(f'Num cameras: {num}')
    except:
        print('Failed to read header')