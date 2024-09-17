import shutil
from pathlib import Path

import paramiko

HOST = "marvin.hpc.uni-bonn.de"
USERNAME = "s6caweni_hpc"

archive_name = "results.temp.tar.gz"

remote_root_path = Path(f"/home/{USERNAME}/spikegd-carlo/experiments/heidelberg")
remote_results_path = remote_root_path / "results"
remote_archive_path = remote_root_path / archive_name

local_root_path = Path(__file__).parent
local_results_path = local_root_path / "results"
local_archive_path = local_root_path / archive_name

print("Downloading results")
print(f"  from: {remote_results_path}")
print(f"  to:   {local_results_path}")

if local_results_path.exists():
    print("Removing existing results")
    shutil.rmtree(local_results_path)

local_results_path.mkdir()

print("Connecting to the cluster via SSH")
with paramiko.SSHClient() as ssh_client:
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(HOST, username=USERNAME)

    def run_ssh(command):
        stdin, stdout, stderr = ssh_client.exec_command(command, get_pty=True)

        for line in iter(stdout.readline, ""):
            print(f"[SSH] {line}", end="")

        for line in iter(stderr.readline, ""):
            print(f"[SSH] {line}", end="")

    print("Compressing results on the cluster")
    run_ssh(
        f"tar -C {remote_results_path.as_posix()!r} -czf {remote_archive_path.as_posix()!r} ."
    )

    print("Downloading archive from cluster")
    local_archive_path = local_root_path / archive_name
    with ssh_client.open_sftp() as sftp_client:
        sftp_client.get(
            remote_archive_path.as_posix(),
            local_archive_path,
            callback=lambda sent, total: print(
                f"  downloaded {sent/total:>4.0%}", end="\r"
            ),
        )

    print("Removing archive on the cluster")
    run_ssh(f"rm {remote_archive_path.as_posix()!r}")

print("Extracting archive")
shutil.unpack_archive(local_archive_path, local_results_path)

print("Removing local archive")
local_archive_path.unlink()

print("Done")
