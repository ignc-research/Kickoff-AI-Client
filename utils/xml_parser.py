import xml.etree.ElementTree as ET
import sys
# Force python XML parser not faster C accelerators
# because we can't hook the C implementation
sys.modules['_elementtree'] = None

import numpy as np

class LineNumberingParser(ET.XMLParser):
    def _start_list(self, *args, **kwargs):
        # Here we assume the default XML parser which is expat
        # and copy its element position attributes into output Elements
        element = super(self.__class__, self)._start_list(*args, **kwargs)
        element._start_line_number = self.parser.CurrentLineNumber
        element._start_column_number = self.parser.CurrentColumnNumber
        element._start_byte_index = self.parser.CurrentByteIndex
        return element

    def _end(self, *args, **kwargs):
        element = super(self.__class__, self)._end(*args, **kwargs)
        element._end_line_number = self.parser.CurrentLineNumber
        element._end_column_number = self.parser.CurrentColumnNumber
        element._end_byte_index = self.parser.CurrentByteIndex
        return element

def parse_frame_dump(xml_file, safe_parsing= True):
    '''parse xml file to get welding spots and torch poses'''
    tree = ET.parse(xml_file, parser= LineNumberingParser())
    root = tree.getroot()

    bad_data_counter = 0
    
    total_info = [] # list of all infos about the torch, welding spots and the transformation matrix

    for i,SNaht in enumerate(root.findall('SNaht')):
        
        torch = [SNaht.get('Name'), SNaht.get('ZRotLock'), SNaht.get('WkzName'), SNaht.get('WkzWkl')]
        weld_frames = [] # list of all weld_frames as np.arrays(X,Y,Z) in mm
        pose_frames = [] # list of all pose_frames as 4x4 homogenous transforms
        starting_lines = [] # list of all starting line (The line of the Point <Pos> in each Pose Frame <Frame>)
        snaht_number = []
        snaht_id = []
        
        for j,Kontur in enumerate(SNaht.findall('Kontur')):  
            for Punkt in Kontur.findall('Punkt'):
                X = float(Punkt.get('X'))
                Y = float(Punkt.get('Y'))
                Z = float(Punkt.get('Z'))
                Norm = []
                Rot_X, Rot_Y, Rot_Z, EA3 = 0, 0, 0, 0
                for Fl_Norm in Punkt.findall('Fl_Norm'):
                    Norm_X = float(Fl_Norm.get('X'))
                    Norm_Y = float(Fl_Norm.get('Y'))
                    Norm_Z = float(Fl_Norm.get('Z'))
                    Norm.append(np.array([Norm_X, Norm_Y, Norm_Z]))
                for Rot in Punkt.findall('Rot'):
                    Rot_X = float(Rot.get('X'))
                    Rot_Y = float(Rot.get('Y'))
                    Rot_Z = float(Rot.get('Z'))
                for Ext_Achswerte in Punkt.findall('Ext-Achswerte'):
                    if Ext_Achswerte.get('EA3') == None:
                        EA3 = float(Ext_Achswerte.get('EA4'))
                    else: 
                        EA3 = float(Ext_Achswerte.get('EA3'))
                weld_frames.append({'position': np.array([X, Y, Z]), 'norm': Norm, 'rot': np.array([Rot_X, Rot_Y, Rot_Z]), 'EA': EA3})
                snaht_number.append(i)
                snaht_id.append(SNaht.get('ID'))
        # desired model output
        for Frames in SNaht.findall('Frames'):  
            for Frame in Frames.findall('Frame'):
                torch_frame = np.zeros((4,4))  # 4x4 homogenous transform
                torch_frame[3,3] = 1.0

                for Pos in Frame.findall('Pos'):
                    # 3x1 position
                    X = Pos.get('X')
                    Y = Pos.get('Y')
                    Z = Pos.get('Z')
                    torch_frame[0:3,3] = np.array([X,Y,Z])
                    start_line = 0
                for XVek in Frame.findall('XVek'):
                    # 3x3 rotation
                    Xrot = np.array([XVek.get('X'), XVek.get('Y'), XVek.get('Z')])      
                    torch_frame[0:3, 0] = Xrot
                for YVek in Frame.findall('YVek'):
                    # 3x3 rotation
                    Yrot = np.array([YVek.get('X'), YVek.get('Y'), YVek.get('Z')])      
                    torch_frame[0:3, 1] = Yrot
                for ZVek in Frame.findall('ZVek'):
                    # 3x3 rotation
                    Zrot = np.array([ZVek.get('X'), ZVek.get('Y'), ZVek.get('Z')])      
                    torch_frame[0:3, 2] = Zrot

                #print(torch_frame) 
                pose_frames.append(torch_frame)
                starting_lines.append(start_line)

        if safe_parsing and len(weld_frames) != len(pose_frames) and len(pose_frames) != 0: # For inference, there are not pose_frames, and in other cases a different amount of entries signals bad data
            bad_data_counter +=1
            print('Bad data at ', i)
            continue
        total_info.append({'torch': torch, 'weld_frames': weld_frames, 'pose_frames': pose_frames, 'pose_frames_starting_lines': starting_lines, 'snaht_number' : snaht_number, 'snaht_id' : snaht_id})
    # print(total_info)
    print('bad_data_counter = ', bad_data_counter)
    if safe_parsing:
        return total_info
    else:
        return total_info, False


def list2array(total_info, safe_parsing= True):
    if total_info[-1] == False:
        total_info = total_info[0]
        safe_parsing = False

    res = []
    for info in total_info:
        for i, spot in enumerate(info['weld_frames']):
            weld_info = []
            weld_info.append(info['torch'][0])
            weld_info.append(info['torch'][1])
            weld_info.append(info['torch'][2])
            weld_info.append(info['torch'][3])
            # torch = info['torch'][3]
            # if torch == 'MRW510_10GH' or torch == 'MRW510_CDD_10GH' or torch == 'TL2000_EFF20_17SO':
            #     weld_info.append(0)
            # elif torch  == 'TAND_GERAD_DD':
            #     weld_info.append(1)
            # else:
            #     weld_info.append(2)
            weld_info.append(spot['position'][0])
            weld_info.append(spot['position'][1])
            weld_info.append(spot['position'][2])
            weld_info.append(spot['norm'][0][0])
            weld_info.append(spot['norm'][0][1])
            weld_info.append(spot['norm'][0][2])
            weld_info.append(spot['norm'][1][0])
            weld_info.append(spot['norm'][1][1])
            weld_info.append(spot['norm'][1][2])

            weld_info.append(spot['rot'][0])
            weld_info.append(spot['rot'][1])
            weld_info.append(spot['rot'][2])


            weld_info.append(spot['EA'])

            if len(info['pose_frames']) > 0 and safe_parsing or \
               len(info['pose_frames']) > 0 and i < len(info['pose_frames']) and not safe_parsing:
                weld_info.append(info['pose_frames'][i][0][0])
                weld_info.append(info['pose_frames'][i][1][0])
                weld_info.append(info['pose_frames'][i][2][0])
                weld_info.append(info['pose_frames'][i][0][1])
                weld_info.append(info['pose_frames'][i][1][1])
                weld_info.append(info['pose_frames'][i][2][1])
                weld_info.append(info['pose_frames'][i][0][2])
                weld_info.append(info['pose_frames'][i][1][2])
                weld_info.append(info['pose_frames'][i][2][2])
                weld_info.append(info['pose_frames_starting_lines'][i])
            weld_info.append(info['snaht_number'][i])
            weld_info.append(info['snaht_id'][i])

            res.append(np.asarray(weld_info))
    return np.asarray(res)



if __name__== '__main__':
    tt = parse_frame_dump('../data/Reisch.xml')
    # t = parse_frame_dump('data_error/train/models/201910204483_R1/201910204483_R1.xml')
    t = list2array(tt)
    print(t.shape)

