"""Generate reproducible SUMO mobility traces used by the testbed.

SUMO is an external runtime. This command creates the network, routes and FCD
trajectory; generated XML is intentionally excluded from the source release.
"""
import argparse
from pathlib import Path
from mobility.sumo_runner import SUMORunner

def main():
    p=argparse.ArgumentParser(); p.add_argument('--scenario',choices=['highway','manhattan'],default='highway'); p.add_argument('--seed',type=int,default=42); p.add_argument('--config',type=Path); p.add_argument('--output',type=Path,default=Path('mobility_output')); a=p.parse_args()
    config=a.config or Path('configs')/f'sumo_{a.scenario}.yaml'
    print(SUMORunner(config, a.output).run(a.seed))

if __name__=='__main__': main()
