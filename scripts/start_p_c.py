from subprocess import CREATE_NEW_CONSOLE, Popen


def start_faust_app(script_name):
    cmd = f'faust -A {script_name} worker -l info'
    cmd = ['open', '-a', 'Terminal', '-n', '--args', '/Applications/Nuke6.3v8/Nuke6.3v8.app/Nuke6.3v8']
    return Popen(cmd, creationflags=CREATE_NEW_CONSOLE)


if __name__ == '__main__':
    producer_process = start_faust_app('main_producer')
    # consumer_process = start_faust_app('main_consumer')

    print(f"Producer gestartet mit PID: {producer_process.pid}")
    # print(f"Consumer gestartet mit PID: {consumer_process.pid}")

    producer_process.wait()
    # consumer_process.wait()
