import os
import sys
from datetime import datetime, timedelta
import atexit
import shutil

class Documenter:
    def __init__(self, run_name, existing_run=None):
        self.run_name = run_name
        if existing_run is None:
            now = datetime.now()
            while True:
                if os.getenv("DOC_LONG_NAME", "0") == "1":
                    full_run_name = now.strftime("%Y%m%d_%H%M%S") + "_" + run_name
                else:
                    full_run_name =  run_name + "_" + now.strftime("%m%d_%H%M")
                self.basedir = os.path.join("output", full_run_name)
                try:
                    os.mkdir(self.basedir)
                    break
                except FileExistsError:
                    now += timedelta(seconds=1)
        else:
            self.basedir = os.path.join("output", existing_run)

        self.tee = Tee(self.add_file("log.txt", False))
        atexit.register(self.close)

    def add_file(self, name, add_run_name=True):
        new_file = self.get_file(name, add_run_name)
        old_dir = os.path.join(self.basedir, "old")
        if os.path.exists(new_file):
            os.makedirs(old_dir, exist_ok=True)
            shutil.move(new_file, os.path.join(old_dir, os.path.basename(new_file)))
        return new_file

    def get_file(self, name, add_run_name=True):
        if add_run_name:
            name_base, name_ext = os.path.splitext(name)
            name = f"{name_base}_{self.run_name}{name_ext}"
        return os.path.join(self.basedir,name)

    def close(self):
        self.tee.close()
        os.system("chmod -R 755 " + self.basedir)

class Tee(object):
    def __init__(self, log_file):
        self.log_file = open(log_file, "w")
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self

    def close(self):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.log_file.close()

    def write(self, data):
        self.log_file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.log_file.flush()
