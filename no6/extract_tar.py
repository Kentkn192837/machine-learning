import tarfile
import os

def read_path(path):
    with open(path, encoding='utf-8') as fp:
        line = fp.readlines()
    return line

datadir = os.path.join(read_path('filepath')[0], 'chapter6')
dataset = 'pedestrians128x64'
datafile = os.path.join(datadir, dataset + '.tar.gz')
extractdir = os.path.join(datadir, dataset)

print("解凍情報")
print(f'datadir: {datadir}')
print(f'dataset: {dataset}')
print(f'datafile: {datafile}')
print(f'extractdir: {extractdir}')

tar = tarfile.open(datafile)
tar.extractall(path=extractdir)
tar.close()
print("解凍が完了しました。")
