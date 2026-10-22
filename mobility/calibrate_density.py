"""Tune SUMO departure rates to realized FCD occupancy per road-centerline km."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SUMO_HOME = Path(os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\Eclipse\Sumo"))
WARMUP = 300.0
DURATION = 10.0
DT = 0.1
SEED = 42


def centerline_length_km(net_path):
    """Count each physical bidirectional grid road once, junction center to center.

    The frozen networks have straight edges. Reverse edges and parallel lanes
    share a centerline; internal junction movement arcs are not extra roads.
    """
    root = ET.parse(net_path).getroot()
    junctions = {j.get("id"): (float(j.get("x")), float(j.get("y")))
                 for j in root.findall("junction") if j.get("type") != "internal"}
    segments = {}
    for edge in root.findall("edge"):
        if edge.get("function") == "internal" or not edge.findall("lane"):
            continue
        a, b = edge.get("from"), edge.get("to")
        if a not in junctions or b not in junctions:
            raise ValueError("Missing external junction coordinates")
        # All external edges in these netgenerate grids allow passenger traffic.
        lanes = edge.findall("lane")
        if not any("passenger" not in l.get("disallow", "").split()
                   and (not l.get("allow") or "passenger" in l.get("allow").split()
                        or "all" in l.get("allow").split()) for l in lanes):
            continue
        segments[tuple(sorted((a, b)))] = math.dist(junctions[a], junctions[b])
    length = sum(segments.values()) / 1000
    if length <= 0:
        raise ValueError("No drivable centerline length")
    return length, len(segments)


def measure_fcd(path, road_km, begin, end):
    samples = []
    for _, elem in ET.iterparse(path, events=("end",)):
        if elem.tag == "timestep":
            t = float(elem.get("time"))
            if begin - 1e-8 <= t < end - 1e-8:
                vehicles = elem.findall("vehicle")
                samples.append((t, len(vehicles), sum(float(v.get("speed")) for v in vehicles)))
            elem.clear()
    expected = round((end - begin) / DT)
    if len(samples) != expected:
        raise ValueError(f"Expected {expected} FCD samples, found {len(samples)}")
    for i, (t, _, _) in enumerate(samples):
        if abs(t - (begin + i * DT)) > 1e-7:
            raise ValueError("Nonuniform FCD sample times")
    mean_n = sum(s[1] for s in samples) / len(samples)
    vehicle_samples = sum(s[1] for s in samples)
    return dict(mean_active_vehicles=mean_n, realized_density_veh_per_km=mean_n / road_km,
                min_active_vehicles=min(s[1] for s in samples), max_active_vehicles=max(s[1] for s in samples),
                mean_vehicle_speed_m_s=sum(s[2] for s in samples) / max(1, vehicle_samples),
                measured_begin_s=begin, measured_end_s=end, sample_count=len(samples), step_length_s=DT)


def run(command, log):
    with open(log, "w") as stream:
        subprocess.run([str(v) for v in command], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)


def calibrate(scenario, density, target):
    work = ROOT / "mobility_output" / "realized_density" / f"{scenario}_{density}"
    work.mkdir(parents=True, exist_ok=True)
    # Reuse the existing frozen topology without regeneration or lane changes.
    net = ROOT / "mobility_output" / f"{scenario}_seed42.net.xml"
    road_km, segment_count = centerline_length_km(net)
    target_count = road_km * target
    rate = target_count / 60.0
    lower, upper = None, None
    attempts = []
    for iteration in range(18):
        prefix = work / f"attempt_{iteration:02d}"
        trips = prefix.with_suffix(".trips.xml")
        routes = prefix.with_suffix(".rou.xml")
        fcd = prefix.with_suffix(".fcd.xml")
        trip_cmd = [sys.executable, "-c",
                    "import sys,runpy; sys.path.insert(0,sys.argv.pop(1)); sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__')",
                    SUMO_HOME / "tools", SUMO_HOME / "tools" / "randomTrips.py",
                    "-n", net, "-o", trips, "-r", routes, "-p", str(1 / rate),
                    "-e", str(WARMUP + DURATION + DT), "--seed", str(SEED), "--max-distance", "2000"]
        sumo_cmd = [SUMO_HOME / "bin" / "sumo.exe", "-n", net, "-r", routes,
                    "--fcd-output", fcd, "--device.fcd.begin", str(WARMUP),
                    "--step-length", str(DT), "--end", str(WARMUP + DURATION + DT),
                    "--seed", str(SEED), "--no-step-log", "--duration-log.disable",
                    "--time-to-teleport", "-1"]
        run(trip_cmd, prefix.with_suffix(".trips.log"))
        run(sumo_cmd, prefix.with_suffix(".sumo.log"))
        measurements = measure_fcd(fcd, road_km, WARMUP, WARMUP + DURATION)
        realized = measurements["realized_density_veh_per_km"]
        attempt = dict(iteration=iteration, departure_rate_veh_s=rate, **measurements,
                       commands=[[str(v) for v in trip_cmd], [str(v) for v in sumo_cmd]])
        attempts.append(attempt)
        (work / "calibration.json").write_text(json.dumps(attempts, indent=2))
        print(f"{scenario}_{density}: rate={rate:.5f}/s, N={measurements['mean_active_vehicles']:.2f}, "
              f"rho={realized:.3f}/{target} veh/km, mean speed={measurements['mean_vehicle_speed_m_s']:.2f} m/s", flush=True)
        if abs(realized / target - 1) <= 0.05:
            output = ROOT / f"{scenario}_{density}.xml"
            tree = ET.parse(fcd)
            for step in tree.getroot().findall("timestep"):
                step.set("time", f"{float(step.get('time')) - WARMUP:.2f}")
            tree.write(output, encoding="utf-8", xml_declaration=True)
            verified = measure_fcd(output, road_km, 0, DURATION)
            if abs(verified["realized_density_veh_per_km"] / target - 1) > 0.05:
                raise AssertionError("Exported trace failed density verification")
            report = dict(scenario=scenario, density=density, target_density_veh_per_km=target,
                          tolerance_fraction=0.05, accepted=True, **verified,
                          road_centerline_km=road_km, physical_road_segments=segment_count,
                          road_length_definition="unique drivable external junction pairs, center to center; reverse directions counted once, lanes not multiplied; internal turn arcs excluded",
                          warmup_s=WARMUP, source_measured_interval_s=[WARMUP, WARMUP + DURATION],
                          endpoint_note="t=10.0 retained only for interpolation; measured interval is [0,10)",
                          seed=SEED, departure_rate_veh_s=rate, network=str(net.relative_to(ROOT)),
                          network_sha256=hashlib.sha256(net.read_bytes()).hexdigest(),
                          trace=str(output.relative_to(ROOT)), trace_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                          calibration_attempts=attempts)
            output.with_suffix(".density.json").write_text(json.dumps(report, indent=2))
            return report
        if realized < target:
            lower = rate
        else:
            upper = rate
        if lower is not None and upper is not None:
            rate = (lower + upper) / 2
        elif realized > 0:
            rate *= min(2.0, max(0.5, target / realized))
        else:
            rate *= 2
    raise RuntimeError(f"Could not reach {scenario}/{density} target; calibration retained for inspection")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["highway", "manhattan"])
    parser.add_argument("--density", choices=["low", "high"])
    args = parser.parse_args()
    for scenario in ([args.scenario] if args.scenario else ["highway", "manhattan"]):
        for density in ([args.density] if args.density else ["low", "high"]):
            calibrate(scenario, density, {"low": 50, "high": 150}[density])


if __name__ == "__main__":
    main()
