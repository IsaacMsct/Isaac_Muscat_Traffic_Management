import subprocess
import threading
import time
import datetime
import os
import keyboard

def log_to_file(message):
    with open("run_script_logs.txt", "a") as log_file:
        log_file.write(f"{datetime.datetime.now()} - {message}\n")

def run_script_forever(script_name, log_file, stop_event):
    venv_python_path = "D:/school/TrafficDetection/venv/TrafficDetection/Scripts/python.exe"
    script_path = f"D:/school/TrafficDetection/venv/{script_name}"
    
    # Start the script immediately
    log_to_file(f"Starting {script_name}")
    process = subprocess.Popen([venv_python_path, script_path])
    last_check_time = time.time()
    initial_check_done = False

    while not stop_event.is_set():
        # Check if initial delay is over or if it's the first check
        if time.time() - last_check_time > 120 or not initial_check_done:  
            if not initial_check_done:  # Apply initial delay only once
                time.sleep(240)  # Initial delay of 4 minutes before the first log check
                initial_check_done = True

            if not is_log_file_updated(log_file):
                log_to_file(f"Log file not updated, restarting {script_name}")
                process.terminate()
                process = subprocess.Popen([venv_python_path, script_path])  # Restart the script
                last_check_time = time.time()  # Reset the timer after restart
                time.sleep(300)
            else:
                log_to_file(f"Log file for {script_name} is updated.")
            last_check_time = time.time()

        if stop_event.is_set():
            process.terminate()
            log_to_file(f"Terminating {script_name}")
        time.sleep(0.5)

def is_log_file_updated(log_file):
    try:
        with open(log_file, 'r') as file:
            lines = file.readlines()
            if lines:
                last_line = lines[-1]
                last_time = datetime.datetime.strptime(last_line.split(',')[0], "%Y-%m-%d %H:%M:%S")
                if (datetime.datetime.now() - last_time).total_seconds() < 300:
                    return True
            return False
    except Exception as e:
        log_to_file(f"Error reading log file {log_file}: {e}")
        return False

if __name__ == "__main__":
    stop_event = threading.Event()
    keyboard.add_hotkey('esc', lambda: stop_event.set())

    # Initial test log to confirm file writing capability
    log_to_file("Initializing script execution...")

    scripts = [
        ("M_TrafficDensityDetectionAutoRestart_yolo.py", "msida_traffic_report.txt"),
        ("MH_TrafficDensityDetectionAutoRestart_yolo.py", "marsahamrun_traffic_report.txt"),
        ("QC_TrafficDensityDetectionAutoRestart_yolo.py", "qormicanonroad_traffic_report.txt"),
        #("QM_TrafficDensityDetectionAutoRestart_yolo.py", "qormimdinaroad_traffic_report.txt")#this webcam's offline
    ]

    threads = []
    for script, log_file in scripts:
        thread = threading.Thread(target=run_script_forever, args=(script, log_file, stop_event))
        thread.start()
        threads.append(thread)
        time.sleep(10)  # Staggered start to avoid overload

    for thread in threads:
        thread.join()  # Wait for threads to finish

    log_to_file("All scripts have been terminated or completed.")