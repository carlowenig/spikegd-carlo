from hyperparam_scan_util import GridScan

scan = GridScan.load("main", root="results")
scan.clean("marvin_01")
