import time
from concurrent import futures
import os
import threading


def cumulative(num: int):
    print(f"{str(num)} 계산을 시작합니다.| {os.getpid()} process | {threading.get_ident()} thread")
    result = 0
    for i in range(num):
        result += i
    print(f"{str(num)} 실행이 완료되었습니다.")
    return result


if __name__ == "__main__":
    targets = [300000000, 20000, 9000000, 50000000, 400000, 40000000]
    start = time.time()
    # multithreading
    with futures.ThreadPoolExecutor(max_workers=10) as executor:

    # multiprocessing
    # with futures.ProcessPoolExecutor(max_workers=10) as executor:

        results = [executor.submit(cumulative, target) for target in targets]
        print(f"results 값 출력 {results}")
        for f in futures.as_completed(results):
            print(f"계산결과는 {f.result()}입니다.")

####################################
#     for target in targets:
#         result = cumulative(target)
#         print(f"{target}의 계산결과는 {result} 입니다.")
# ####################################

    end = time.time()
    print(f"실행 완료까지 걸린 시간 : {end-start:.2f}")