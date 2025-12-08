# example of getting a child process by name
from time import sleep
from multiprocessing import active_children
from multiprocessing import Process
 
# return a child process with a given name or none
def get_process_by_name(name):
    # get all child processes
    processes = active_children()
    # return the process with the name
    for process in processes:
        # check for match
        if process.name == name:
            return process
    # no match
    return None
 
# function executed in a child process
def task():
    # block for a while
    sleep(3)
    print('sleep 3sec in task')
 
# protect the entry point
if __name__ == '__main__':
    # configure child processes
    children = [Process(target=task) for _ in range(10)]
    # start child processes
    for child in children:
        child.start()
    # wait a moment
    sleep(1)
    print('sleep 1sec')
    # get the child by name
    child = get_process_by_name('Process-2')
    print(f'Found: {child}')