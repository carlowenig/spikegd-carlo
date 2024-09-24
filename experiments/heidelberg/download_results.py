import shutil
from pathlib import Path, PurePath
from typing import Iterable

import paramiko

HOST = "marvin.hpc.uni-bonn.de"
USERNAME = "s6caweni_hpc"


def run_ssh(ssh_client: paramiko.SSHClient, command: str):
    print(f"[SSH] > {command}")

    stdin, stdout, stderr = ssh_client.exec_command(command, get_pty=True)

    for line in iter(stdout.readline, ""):
        print(f"  (out) {line}", end="")

    for line in iter(stderr.readline, ""):
        print(f"  (err) {line}", end="")


def create_tar_command(
    archive_path: PurePath, dir_path: PurePath, exclude: Iterable[str] = ()
):
    tar_command = f"tar -C {dir_path.as_posix()!r}"

    for pattern in exclude:
        tar_command += f" --exclude={pattern!r}"

    tar_command += f" -czf {archive_path.as_posix()!r} ."

    return tar_command


def download_results(
    archive_name="results.temp.tar.gz",
    remote_root_path=PurePath(f"/home/{USERNAME}/spikegd-carlo/experiments/heidelberg"),
    local_root_path=Path(__file__).parent,
    exclude: Iterable[str] = ("*_backup_*",),
):
    remote_results_path = remote_root_path / "results"
    remote_archive_path = remote_root_path / archive_name

    local_results_path = local_root_path / "results"
    local_archive_path = local_root_path / archive_name

    print("Downloading results")
    print(f"  from: {remote_results_path}")
    print(f"  to:   {local_results_path}")

    print("Connecting to the cluster via SSH")
    with paramiko.SSHClient() as ssh_client:
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_client.connect(HOST, username=USERNAME)

        print("Compressing results on the cluster")
        run_ssh(
            ssh_client,
            create_tar_command(remote_archive_path, remote_results_path, exclude),
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
        run_ssh(ssh_client, f"rm {remote_archive_path.as_posix()!r}")

    if local_results_path.exists():
        print("Removing existing local results")
        shutil.rmtree(local_results_path)

    local_results_path.mkdir()

    print("Extracting local archive")
    shutil.unpack_archive(local_archive_path, local_results_path)

    print("Removing local archive")
    local_archive_path.unlink()

    print("Done")


if __name__ == "__main__":
    download_results()
