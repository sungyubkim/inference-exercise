import subprocess
import logging
import sys
import time
from datetime import datetime

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

def run_script_with_args(script_path, *args):
    logging.info(f"Running script at {script_path} with args: {args}")
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
            logging.error(error)
        
        if output:
            logging.info(output)
        
        if process.poll() is not None:
            break
        time.sleep(0.1)  # 짧은 대기 시간 추가

    return process.poll()

# 예제 사용법
prompt_list = [
    'Can you explain Jensen-Shannon Divergence?',
    'Can you explain F-divergence?',
    'Can you explain Wasserstein Distance?',
]
model_list = [
    'microsoft/Phi-3-mini-4k-instruct',
    'microsoft/Phi-3-mini-128k-instruct',
]

for prompt in prompt_list:
    for model in model_list:
        script_path = f"runw/inference-phi3.py"
        args = ["--model", model, "--prompt", prompt]
        run_script_with_args(script_path, *args)