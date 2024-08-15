import time
import os

TIME_TO_SLEEP = 60 * 60 * 6.5  # 6 and a half hours


def main():
    while True:
        print(f"Restarting in {TIME_TO_SLEEP / 60 / 60:.2f} hours")
        time.sleep(TIME_TO_SLEEP)
        os.system("./start_mainnet_validator.sh")


if __name__ == "__main__":
    main()