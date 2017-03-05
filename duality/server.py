import socketserver
import queue
import sys
import socket
import json
import threading
from http import server as http_server, HTTPStatus


IS_DEBUG=False;


if(not IS_DEBUG):
    from .context import MEMORY_DIR
    from .model_minimal import MikuCore
    miku= MikuCore(data_path=MEMORY_DIR);


INTERVAL = 5
HOST, TCP_PORT, UDP_PORT = "10.141.246.29", 4000, 13000

comment_q = queue.Queue()
# init the client ip address pool, a mechanism to maintain the client pool should be implemented
client_pool = set()


class POSTHandler(http_server.BaseHTTPRequestHandler):
    """a simple HTTP server that only serves POST request with a JSON content"""

    def do_POST(self):
        """Serve a POST request."""
        try:
            length = int(self.headers.get('content-length'));
            print(length);
        except:
            print('No or incorrect content-len gth found.', file=sys.stderr)
        self.data = self.rfile.read(length)
        self.send_head()
        comment_q.put(self.data)
        print(self.data);
        json_data=json.loads(self.data.decode('utf-8'));

        if(not IS_DEBUG):
            miku.process(json_data['content'].strip(', '));

        client_pool.add(self.address_string())
        # you should always send the new bev data to all the client in the pool, where a server has been started

    def send_head(self):
        self.send_response(HTTPStatus.OK)
        self.end_headers()


class ThreadingHTTPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    """a mix-in class that mix threading in a TCPServer"""
    pass


def timer_send_datagrams():
    """use a timer to send datagram at some interval"""
    t = threading.Timer(INTERVAL, send_datagrams)
    t.start()


def udp_send(ip, port, message):
    """send message to ip:port over UDP"""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.sendto(bytes(message, "utf-8"), (ip, port))
        # response = str(sock.recv(1024), "utf-8")
        # print("Received: {}".format(response))


def send_datagrams():
    """send all comment in current queue to all clients in client_pool over UDP"""
    comment_list = list()
    if True:
        # extract the comments from the queue and append them to a list
        while not comment_q.empty():
            comment_list.append(json.loads(comment_q.get().decode('utf-8')))


        # data to be send NEED TO BE MODIFIED

        data= {"action_id": 0, "voice_id": 0, "option": 0, "comments": comment_list}
        if(not IS_DEBUG):
            miku_data=miku.respond();
            data['action_id']=miku_data['action_id'];
            data['voice_id']=miku_data['voice_id'];
            data['option']=miku_data['option'];

        # encode data to json format
        json_data = json.JSONEncoder().encode(data)
        # send to all clients in client_pool
        print("-------push data------");
        print(json_data);
        for host in client_pool:
            udp_send(host, UDP_PORT, json_data)
    timer_send_datagrams()


if __name__ == "__main__":
    timer_send_datagrams()
    server= ThreadingHTTPServer((HOST, TCP_PORT), POSTHandler);
    # Activate the server; this will keep running until you
    # interrupt the program with Ctrl-C
    server.serve_forever()
