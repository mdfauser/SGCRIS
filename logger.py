class SimpleLogger:
    metrics = {}

    def aggregate(self, data):
        for key, val in data.items():
            if key in self.metrics:
                self.metrics[key].append(val.item())
            else:
                self.metrics[key] = [val.item()]

    def print(self):
        str = ""
        for key, l in self.metrics.items():
            v = sum(l) / len(l)
            str += f"{key}: {v:.3f} "
        return str

    def flush(self):
        self.metrics = {}
