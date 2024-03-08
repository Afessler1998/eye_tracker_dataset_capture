import time

def set_timeout(event, timeout=1):
    """
    Thread to set a timeout for a process
    Uses an event to signal the timeout and a flag to indicate the process is waiting

    :param event: event to set after the timeout
    :param timeout: timeout for the process
    """
    while True:
        event.wait()
        time.sleep(timeout)
        event.clear()