import subprocess
import logging
import sys
import time
from datetime import datetime
from multiprocessing import Process, Queue, current_process
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--max_processes', type=int, default=1, help='Maximum number of processes to run concurrently')
args = parser.parse_args()

# logging 모듈 설정, 로그 파일의 이름은 현재 시간을 포함하여 자동으로 생성
logging.basicConfig(
    level=logging.INFO,  # 로그 레벨 설정
    format='%(asctime)s %(levelname)s %(message)s',  # 로그 메시지 포맷
    datefmt='%Y-%m-%d-%H:%M:%S',  # 시간 포맷
    handlers=[
        logging.FileHandler(f'./logs/{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.log'),
        logging.StreamHandler(sys.stdout)  # 콘솔 출력 핸들러
    ]
)

def worker(script_path, args, queue):
    """각 프로세스에서 실행될 작업 함수"""
    queue.put((logging.INFO, f"Running script at {script_path} with args: {args}"))
    command = ['python', script_path] + list(args)
    # 프로세스 시작
    process = subprocess.Popen(
        command,  # 실행할 스크립트
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        text=True  # 텍스트 모드 활성화 (Python 3.6 이상),
    )

    # 표준 출력과 표준 오류를 읽어서 로그에 기록
    while True:
        output = process.stdout.read()
        error = process.stderr.read()
        
        if error:
            queue.put((logging.ERROR, error))
        
        if output:
            queue.put((logging.INFO, output))
        
        if process.poll() is not None:
            break
        time.sleep(0.1)  # 짧은 대기 시간 추가

    queue.put((logging.INFO, f"Process {current_process().name} finished with exit code {process.poll()}"))

def logger(queue):
    """로그 메시지를 기록하는 함수"""
    while True:
        level, message = queue.get()
        if message == "DONE":
            break
        logging.log(level, message)

if __name__ == "__main__":
    prompt_list = [
        'Can you explain Jensen-Shannon Divergence?',
        'Can you explain F-divergence?',
        'Can you explain Wasserstein Distance?',
    ]
    model_list = [
        'microsoft/Phi-3-mini-4k-instruct',
        'microsoft/Phi-3-mini-128k-instruct',
    ]
    
    queue = Queue()
    log_process = Process(target=logger, args=(queue,))
    log_process.start()

    processes = []
    task_queue = []

    for prompt in prompt_list:
        for model in model_list:
            script_path = f"runw/inference-phi3.py"
            python_args = ["--model", model, "--prompt", prompt]
            task_queue.append((script_path, python_args))

    while task_queue or processes:
        # 실행할 수 있는 프로세스가 있고, 현재 실행 중인 프로세스 수가 최대 프로세스 수보다 적으면
        while task_queue and len(processes) < args.max_processes:
            script_path, python_args = task_queue.pop(0)
            p = Process(target=worker, args=(script_path, python_args, queue))
            p.start()
            processes.append(p)

        # 완료된 프로세스 제거
        for p in processes:
            if not p.is_alive():
                p.join()
                processes.remove(p)

        time.sleep(0.1)

    queue.put((logging.INFO, "All processes completed."))
    queue.put((logging.INFO, "DONE"))
    log_process.join()
