import shutil
import os

CURRENT_PATH = os.path.abspath(__file__)
ROOT = os.path.dirname(CURRENT_PATH)
Parent= os.path.dirname(ROOT)
data_path=os.path.join(ROOT,'data2')
xml_path=os.path.join(Parent,'xml')
data_list=os.listdir(data_path)
for data_name in data_list:
    shutil.copy(xml_path+'/'+data_name+'.xml',data_path+'/'+data_name+'/'+data_name+'.xml')