import subprocess, time, csv, os
from datetime import datetime

def stamp():
    return datetime.now().isoformat(timespec="milliseconds")

CMD = ["ffmpeg", "-y", "-loglevel", "error",
       "-framerate", "25", "-i", "data/MVI_20011/images/img%05d.jpg",
       "-frames:v", "200", "-c:v", "libx265", "-crf", "28", "-preset", "medium",
       "-x265-params", "log-level=error", "-pix_fmt", "yuv420p",
       "data/encoded/test_power.mp4"]

REPS = 5
IDLE = 20   # secondes de repos avant chaque exécution

os.makedirs("results", exist_ok=True)
rows = []
print("Ne touchez à rien pendant ce test (environ 1 minute 30).")
for rep in range(1, REPS + 1):
    idle_start = stamp()
    time.sleep(IDLE)
    start = stamp()
    subprocess.run(CMD, check=True)
    end = stamp()
    rows.append(["encode_crf28", rep, idle_start, start, end])
    print(f"rep {rep}: {start} -> {end}")
time.sleep(IDLE)

with open("results/runs_log.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["label", "rep", "idle_start", "start", "end"])
    w.writerows(rows)
print("Heures enregistrées dans results/runs_log.csv")