import os
import subprocess
import yaml
import json
import sys

class SUMORunner:
    """Manages SUMO instance, network generation, and trace extraction."""
    def __init__(self, config_path, output_dir="mobility_output"):
        self.config_path = config_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def run(self, seed=42):
        sumo_home = os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\Eclipse\Sumo")
        if not os.path.isdir(sumo_home):
            raise EnvironmentError(f"SUMO directory not found at: {sumo_home}. Please set SUMO_HOME.")

        netgenerate_bin = os.path.join(sumo_home, "bin", "netgenerate")
        random_trips_script = os.path.join(sumo_home, "tools", "randomTrips.py")
        sumo_bin = os.path.join(sumo_home, "bin", "sumo")

        density = self.config.get("density")
        density_suffix = f"_{density}" if density else ""
        prefix = f"{self.config['scenario']}{density_suffix}_seed{seed}"
        net_file = os.path.join(self.output_dir, f"{prefix}.net.xml")
        trips_file = os.path.join(self.output_dir, f"{prefix}.trips.xml")
        routes_file = os.path.join(self.output_dir, f"{prefix}.rou.xml")
        fcd_file = os.path.join(self.output_dir, f"{prefix}.fcd.xml")
        record_file = os.path.join(self.output_dir, f"{prefix}.record.json")

        # 1. netgenerate
        net_cmd = [
            netgenerate_bin,
            "--grid",
            "--grid.x-number", str(self.config.get("x_number", 5)),
            "--grid.y-number", str(self.config.get("y_number", 5)),
            "--grid.length", str(self.config.get("grid_length", 200)),
            "--default.lanenumber", str(self.config.get("num_lanes", 2)),
            "--output-file", net_file,
            "--seed", str(seed)
        ]

        # 2. randomTrips
        period = 1.0 / self.config.get("insertion_rate", 2.0)
        trips_cmd = [
            sys.executable, random_trips_script,
            "-n", net_file,
            "-o", trips_file,
            "-r", routes_file,
            "-p", str(period),
            "-e", str(self.config.get("duration_s", 60)),
            "--seed", str(seed),
            "--max-distance", "2000"
        ]

        # 3. sumo
        sumo_cmd = [
            sumo_bin,
            "-n", net_file,
            "-r", routes_file,
            "--fcd-output", fcd_file,
            "--step-length", str(self.config.get("step_length_s", 0.1)),
            "--seed", str(seed),
            "--no-step-log",
            "--max-num-vehicles", str(self.config.get("max_vehicles", 300))
        ]

        commands = {
            "netgenerate": net_cmd,
            "randomTrips": trips_cmd,
            "sumo": sumo_cmd
        }

        # Save record for reproducibility
        with open(record_file, 'w') as f:
            json.dump(commands, f, indent=2)

        subprocess.run(net_cmd, check=True)
        subprocess.run(trips_cmd, check=True)
        subprocess.run(sumo_cmd, check=True)

        return fcd_file
