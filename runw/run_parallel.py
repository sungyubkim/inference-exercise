import subprocess
import logging
import sys
import time
from datetime import datetime
from multiprocessing import Process, Queue, current_process
import argparse

parser = argparse.ArgumentParser(description='Run inference on multiple models and prompts in parallel.')
parser.add_argument('--max_processes', type=int, default=1, help='Maximum number of processes to run in parallel.')
args = parser.parse_args()

def setup_logger(log_file):
    """새로운 로거를 설정하여 특정 파일에 로그를 저장"""
    logger = logging.getLogger(str(current_process().pid))
    logger.setLevel(logging.INFO)
    
    # 파일 핸들러 설정
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # 콘솔 핸들러 설정
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # 포맷터 설정
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%d-%H:%M:%S')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 핸들러를 로거에 추가
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def worker(script_path, args, queue, log_file):
    """각 프로세스에서 실행될 작업 함수"""
    logger = setup_logger(log_file)
    logger.info(f"Running script at {script_path} with args: {args}")
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
            logger.error(error)
        
        if output:
            logger.info(output)
        
        if process.poll() is not None:
            break
        time.sleep(0.1)  # 짧은 대기 시간 추가

    logger.info(f"Process {current_process().name} finished with exit code {process.poll()}")

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

    processes = []
    task_queue = []

    for prompt in prompt_list:
        for model in model_list:
            script_path = f"runw/inference-phi3.py"
            py_args = ["--model", model, "--prompt", prompt]
            model_replaced = model.replace('/','-')
            prompt_replaced = prompt.replace('/','-')
            log_file = f'./logs/{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}-{model_replaced}-{prompt_replaced}.log'
            task_queue.append((script_path, py_args, log_file))

    while task_queue or processes:
        # 실행할 수 있는 프로세스가 있고, 현재 실행 중인 프로세스 수가 최대 프로세스 수보다 적으면
        while task_queue and len(processes) < args.max_processes:
            script_path, py_args, log_file = task_queue.pop(0)
            p = Process(target=worker, args=(script_path, py_args, queue, log_file))
            p.start()
            processes.append(p)

        # 완료된 프로세스 제거
        for p in processes:
            if not p.is_alive():
                p.join()
                processes.remove(p)

        time.sleep(0.1)
