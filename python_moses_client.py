import os.path
import socket
import struct
import threading
from time import sleep
from client_matching import matching
import configparser
import time
from threading import Thread
# create a c++ struct like the message in the msclient source code
# the original struct is copied below for convenience
"""
typedef struct mscl_nachricht { char sender[50]; //String mit 50 Zeichen,
					 char empfaenger[50]; //String mit 50 Zeichen
					 WORD auftrnr; //eindeutige Nummer für aktuelles Segment (2 Byte)
					 BYTE dienstnr; //Kennung des Datenpakets
					 char daten[256]; //Zeichenkette mit 256 Zeichen
} t_mscl_nachricht;
"""
# word <=> 2 bytes
# byte <=> 1 byte (self-explanatory)

format = "50s 50s H B 256s"  # h is for a short int (2 bytes), b is for signed char (1 byte)
message_struct = struct.Struct(format)
CURRENT_PATH = os.path.abspath(__file__)
ROOT = os.path.dirname(CURRENT_PATH)

def connect_moses_socket(host, port):

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))

    return s

def send_moses_message(con_socket, client_name, server_name, service_number, task_number, data):
    
    # convert the input data into a bytecode message in the form of the above struct
    byte_message = message_struct.pack(client_name.encode(), server_name.encode(), task_number, service_number, data.encode())

    # send message
    con_socket.send(byte_message)

def receive_moses_message(con_socket):

    # while True:
        # await response
        response = message_struct.unpack(con_socket.recv(4096))

        # some cleanup of the response to make it more human readable
        # the enconding for the decode() method may throw errors in some cases, latin-1 seems to work though
        response_sender = response[0].decode('latin-1').split("\x00",1)[0]
        response_target = response[1].decode('latin-1').split("\x00",1)[0]
        response_task_number = response[2]
        response_service_number = response[3]
        response_data = response[4].decode('latin-1').split("\x00",1)[0]

        print("-------------------------")
        print("Message response:")
        print("Sender: " + response_sender)
        print("Target: " + response_target)
        print("Task number: " + str(response_task_number))
        print("Service number: " + str(response_service_number))
        print("Data: " + response_data)
        print("-------------------------")
        return response_sender,response_target,response_task_number,response_service_number, response_data

def recycle(con_socket, client_name, server_name, service_number, task_number,freuquency,flag):
    while flag:
        send_moses_message(con_socket, client_name, server_name, service_number, task_number, 'running')
        time.sleep(freuquency)
    return


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('config.ini')
    host=config.get('config','server_ip')
    port=int(config.get('config','port'))
    xml_dir=config.get('config','xml_dir')
    auto_del=config.get('config','auto_del')
    model = config.get('config','model')
    freuquency=int(config.get('config','frequency'))
    con_socket = connect_moses_socket(host, port)
    send_moses_message(con_socket,"KI_Client", "MOSES_Cl1", 123, 60, 'Verbunden')
    star_time=time.time()
    while True:
        flag=True
        sender,target,task_number,service_number,data=receive_moses_message(con_socket)
        t1 = Thread(target=recycle, args=(con_socket, sender, target, service_number, task_number,freuquency,flag))
        if service_number==61:
            send_moses_message(con_socket,sender,target,service_number,task_number,'Pose Estimation')
            t1.start()
            matching(os.path.join(ROOT,xml_dir),data.rstrip(),model,service_number,auto_del=auto_del)
            send_moses_message(con_socket,target,sender,service_number,task_number,'finished')


        elif service_number==62:
            send_moses_message(con_socket,sender,target,service_number,task_number,'training model '+str(model))
            t1.start()
            matching(os.path.join(ROOT,xml_dir),data.rstrip(),model,service_number,auto_del=auto_del)
            send_moses_message(con_socket,target,sender,service_number,task_number,'finished')

        elif service_number==63:
            send_moses_message(con_socket, sender, target, service_number, task_number, 'Find similar slices using model ' + str(model))
            t1.start()
            matching(os.path.join(ROOT, xml_dir), data.rstrip(), model, service_number,auto_del=auto_del)
            send_moses_message(con_socket, target, sender, service_number, task_number, 'finished')

        elif service_number==64:
            send_moses_message(con_socket, sender, target, service_number, task_number, 'Training Pose Estimation')
            t1.start()
            matching(os.path.join(ROOT, xml_dir), data.rstrip(), model, service_number,auto_del=auto_del)
            send_moses_message(con_socket, target, sender, service_number, task_number, 'finished')

        elif service_number==71:
            send_moses_message(con_socket, sender, target, service_number, task_number, 'PCD for trainingn')
            t1.start()
            matching(os.path.join(ROOT, xml_dir), data.rstrip(), model, service_number,auto_del=auto_del)
            send_moses_message(con_socket, target, sender, service_number, task_number, 'finished')

        elif service_number==72:
            send_moses_message(con_socket, sender, target, service_number, task_number, 'OBJ for training')
            t1.start()
            matching(os.path.join(ROOT, xml_dir), data.rstrip(), model, service_number,auto_del=auto_del)
            send_moses_message(con_socket, target, sender, service_number, task_number, 'finished')

        elif service_number==73:
            send_moses_message(con_socket, sender, target, service_number, task_number, 'Detect Function')
            t1.start()
            matching(os.path.join(ROOT, xml_dir), data.rstrip(), model, service_number,auto_del=auto_del)
            send_moses_message(con_socket, target, sender, service_number, task_number, 'finished')

        elif service_number==74:
            send_moses_message(con_socket, sender, target, service_number, task_number, 'Visualization')
            t1.start()
            matching(os.path.join(ROOT, xml_dir), data.rstrip(), model, service_number,auto_del=auto_del)
            send_moses_message(con_socket, target, sender, service_number, task_number, 'finished')

        flag=False
            # if len(result)==1:
            #     send_moses_message(con_socket,target,sender,123,task_number,'Keine ähnliche Schweißpositionen gefunden')
            # else:
            #     sum_xml=''
            #     send_moses_message(con_socket,sender,target,123,task_number,'Ähnliche Schweißpositionen gefunden')
            #     #the best matching
            #     matching_slice=list(result.keys())
            #     for slice in matching_slice:
            #         if int(slice.split('_')[1])==task_number:
            #             continue
            #         xml_file=slice+'.xml, '
            #         sum_xml+=xml_file
            #     print(sum_xml)
            #     send_moses_message(con_socket,target,sender,123,task_number,sum_xml)
        #
        #
        # client_name = "PierceCSL"
        # server_name = "MOSES Server"
        #
        # service_number = 120
        # task_number    = 0
        #
        # data = "C:/Data/Test/hallo_welt.py"
        #
        # print(listener.is_alive())
        # send_moses_message(con_socket, client_name, server_name, service_number, task_number, data)
        # sleep(0.5)







