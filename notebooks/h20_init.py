
import time
import h2o

#h2o.init(ip="127.0.0.1", port=54321, nthreads=-1, max_mem_size="4G", verbose=True)

import subprocess, shlex
try:
    out = subprocess.check_output(shlex.split("java -version"), stderr=subprocess.STDOUT, timeout=8)
    print(out.decode("utf-8", "ignore"))
except Exception as e:
    print("Java not callable:", e)

t0=time.time()
try:
    import h2o
    print("import h2o OK in", round(time.time()-t0,2), "s")
except Exception as e:
    print("import h2o FAILED:", e)

