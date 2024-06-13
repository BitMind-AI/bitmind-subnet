import json
import os
from datetime import date


class ProxyCounter:
    def __init__(self, save_path):
        self.save_path = save_path
        if os.path.exists(save_path):
            try:
                self.proxy_logs = json.load(open(save_path))
            except Exception as e:
                print(f"Error loading proxy logs: {e}")
                self.proxy_logs = {}
        else:
            self.proxy_logs = {}

    def update(self, is_success):
        today = str(date.today())
        self.proxy_logs.setdefault(today, {"success": 0, "fail": 0})
        if is_success:
            self.proxy_logs[today]["success"] += 1
        else:
            self.proxy_logs[today]["fail"] += 1

    def save(self):
        json.dump(self.proxy_logs, open(self.save_path, "w"))
