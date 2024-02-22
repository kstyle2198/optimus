import time
import asyncio
import os
import threading


async def cumulative(num: int):
    result = 0
    print(f"{str(num)} 계산을 시작합니다.| {os.getpid()} process | {threading.get_ident()}")
    for i in range(num):
        result += i
    print(f"{str(num)} 실행이 완료되었습니다.")
    print(result)
    return result


async def main():
    result = await asyncio.gather(*[cumulative(target) for target in targets])
    print(result)
    return result


if __name__ == "__main__":
    targets = [300000000, 20000, 9000000, 50000000, 400000, 40000000]
    start = time.time()

    asyncio.run(main())

    end = time.time()
    print(f"실행 완료까지 걸린 시간 : {end-start:.2f}")