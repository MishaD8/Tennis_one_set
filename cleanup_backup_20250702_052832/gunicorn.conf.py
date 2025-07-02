import multiprocessing

bind = "0.0.0.0:5001"
workers = min(4, (multiprocessing.cpu_count() * 2) + 1)
worker_class = "sync"
timeout = 120
keepalive = 2
max_requests = 1000
max_requests_jitter = 100
preload_app = True