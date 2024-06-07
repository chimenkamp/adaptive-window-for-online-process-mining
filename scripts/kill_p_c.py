from typing import List

import psutil
import sys


def find_and_kill_process_by_port(ports: List[int]):
    for proc in psutil.process_iter(['pid', 'name', 'connections']):
        try:
            connections = proc.info['connections']
            if connections is not None:
                for conn in connections:
                    if conn.laddr.port in ports:
                        print(f"Found {proc.info['name']} with PID {proc.info['pid']} ->  {conn.laddr.port}.")
                        proc.kill()
                        print(f"Process {proc.info['pid']} killed.")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue


if __name__ == '__main__':
    ports_input = [int(p) for p in sys.argv[1:]]
    find_and_kill_process_by_port(ports_input)
