#!/usr/bin/env python3
"""
Very simple HTTP server in python for logging requests
Usage::
    ./server.py [<port>]
"""

import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
import re
import csv, os


level_filename = '111.csv'
session_filename = '222.csv'
uid_play_filename = '333.csv'

def check_version(version_name):
    # x1 = re.split("\.", "1.29.1")
    x1 = re.split("\.", "0.0.0")
    x2 = re.split("\.", version_name)
    return(x2 >= x1)

def read_data(log):
        if log.find('SaveLevel') > -1:
            uid=log[log.find('uid=')+4:log.find('&msg')]
            s=log[log.find('level'):]
            level=s[9:s.find(',')]
            s=log[log.find('time_to_play'):]
            time_to_play=s[16:s.find(',')]
            s=log[log.find('ball_popped'):]
            ball_popped=s[15:s.find(',%')]
            s=log[log.find('ball_drop'):]
            ball_drop=s[13:s.find(',%')]
            s=log[log.find('shot'):]
            shot=s[9:s.find(',')]
            s=log[log.find('is_win'):]
            is_win=s[10:s.find(',')]
            s=log[log.find('is_first_time_win'):]
            is_first_time_win=s[21:s.find(',')]
            s=log[log.find('powerball'):]
            powerball=s[13:s.find(',%22date_login')]

            data = [uid, level, time_to_play, ball_popped, ball_drop, shot, is_win, is_first_time_win, powerball]
            with open(level_filename,'a', newline="") as f:
                writer = csv.writer(f)
                writer.writerow(data)
            
        if log.find('StartSession') > -1:
            uid=log[log.find('uid=')+4:log.find('&msg')]
            s=log[log.find('number_of_sessions'):]
            number_of_sessions=s[22:s.find(',')]
            s=log[log.find('length_of_sessions'):]
            length_of_sessions=s[22:s.find(',')]
            s=log[log.find('interval_between_sessions'):]
            interval_between_sessions=s[29:s.find(',')]
            s=log[log.find('playCount'):]
            playCount=s[13:s.find(',')]
            s=log[log.find('levelEnd'):]
            levelEnd=s[12:s.find(',')]
            s=log[log.find('version_name'):]
            version_name = s[19:s.find('%22,%22p')]
        
            data = [uid, number_of_sessions, length_of_sessions, interval_between_sessions, playCount]
            now = datetime.datetime.now()
            time = [uid, now.strftime('%Y-%m-%d %H:%M:%S')]
            with open(session_filename,'a', newline="") as f:
                writer = csv.writer(f)
                writer.writerow(data)
            with open(uid_play_filename,'a', newline="") as f:
                writer = csv.writer(f)
                writer.writerow(time)

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""
    
class S(BaseHTTPRequestHandler):
    def _set_response(self):
        self.send_response(200)

    def do_GET(self):
        self._set_response()
        if not os.path.exists(session_filename):
            with open(session_filename,'w', newline="") as f:
                writer = csv.writer(f)
                writer.writerow(['uid', 'number_of_sessions', 'length_of_sessions', 'interval_between_sessions', 'playCount'])

        if not os.path.exists(level_filename):
            with open(level_filename,'w', newline="") as f:
                writer = csv.writer(f)
                writer.writerow(['uid', 'level', 'time_to_play', 'ball_popped', 'ball_drop', 'shot', 'is_win', 'is_first_time_win', 'powerball'])
        read_data(self.path)

def run(server_class=HTTPServer, handler_class=S, port=8080):
    server_address = ('localhost', port)
    httpd = ThreadedHTTPServer(server_address, S)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
         pass
    httpd.server_close()
    
if __name__ == '__main__':
    from sys import argv

    if len(argv) == 2:
        run(port=int(argv[1]))
    else:
        run()