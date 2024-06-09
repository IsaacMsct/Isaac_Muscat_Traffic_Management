import subprocess
import threading
import time
import keyboard

def run_script_forever(script_name, stop_event):
    venv_python_path = "D:/school/TrafficDetection/venv/TrafficDetection/Scripts/python.exe"
    script_path = f"D:/school/TrafficDetection/venv/{script_name}"
    
    while not stop_event.is_set():
        print(f"Starting {script_name}")
        process = subprocess.Popen([venv_python_path, script_path])
        # Wait for the script process to complete or stop event to be set
        while process.poll() is None and not stop_event.is_set():
            time.sleep(0.5)
        
        if stop_event.is_set():
            process.terminate()  # Attempt to terminate the process if stop event is set
            print(f"Terminating {script_name}")
            break
        
        print(f"{script_name} exited, restarting...")
        time.sleep(5)  # Delay before restarting the script

def on_esc_press(stop_event):
    stop_event.set()  # Set the stop event when ESC is pressed

if __name__ == "__main__":
    stop_event = threading.Event()
    keyboard.on_press_key("esc", lambda _: on_esc_press(stop_event))

    scripts = [
        "M_TrafficDensityDetectionAutoRestart_yolo.py",
        "MH_TrafficDensityDetectionAutoRestart_yolo.py",
        #"QM_TrafficDensityDetectionAutoRestart_yolo.py",
        "QC_TrafficDensityDetectionAutoRestart_yolo.py"
    ]

    threads = []
    for script in scripts:
        thread = threading.Thread(target=run_script_forever, args=(script, stop_event))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()  # Wait for threads to finish (they only finish if ESC is pressed)
