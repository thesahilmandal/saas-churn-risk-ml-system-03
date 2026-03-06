import subprocess
import sys
from pathlib import Path
from typing import List, Union

from src.exception import CustomerChurnException
from src.logging import logging


PathLike = Union[str, Path]


class S3Sync:
    """
    Production-grade AWS S3 sync utility.

    Responsibilities:
    - Normalize Path / str inputs
    - Execute AWS CLI safely
    - Provide structured logging
    - Surface subprocess errors clearly
    """

    # --------------------------------------------------
    # INTERNAL COMMAND RUNNER
    # --------------------------------------------------

    def _run(self, command: List[PathLike]) -> None:
        """
        Execute AWS CLI command safely.

        Converts all command arguments to string
        to prevent PosixPath-related failures.
        """
        try:
            # Convert all arguments to strings to avoid Path issues
            normalized_command = [str(arg) for arg in command]

            logging.info("[S3] %s", " ".join(normalized_command))

            result = subprocess.run(
                normalized_command,
                check=True,
                capture_output=True,
                text=True,
            )

            # Optional but useful for debugging AWS CLI output
            if result.stdout:
                logging.info(result.stdout.strip())

        except subprocess.CalledProcessError as exc:
            logging.error("[S3 ERROR]")
            logging.error(exc.stderr)
            raise CustomerChurnException(exc, sys)

    # --------------------------------------------------
    # DOWNLOAD SINGLE FILE
    # --------------------------------------------------

    def download_file(self, s3_uri: str, local_path: PathLike) -> None:
        """
        Download a single file from S3.
        """
        local_path = Path(local_path)

        # Ensure parent directory exists
        local_path.parent.mkdir(parents=True, exist_ok=True)

        self._run(
            [
                "aws",
                "s3",
                "cp",
                s3_uri,
                local_path,
            ]
        )

    # --------------------------------------------------
    # SYNC LOCAL → S3
    # --------------------------------------------------

    def sync_folder_to_s3(
        self,
        folder: PathLike,
        aws_bucket_url: PathLike,
    ) -> None:
        """
        Sync local folder to S3 bucket.
        """
        self._run(
            [
                "aws",
                "s3",
                "sync",
                folder,
                aws_bucket_url,
            ]
        )

    # --------------------------------------------------
    # SYNC S3 → LOCAL
    # --------------------------------------------------

    def sync_folder_from_s3(
        self,
        folder: PathLike,
        aws_bucket_url: PathLike,
    ) -> None:
        """
        Sync S3 folder to local directory.
        """
        folder = Path(folder)

        # Ensure local directory exists
        folder.mkdir(parents=True, exist_ok=True)

        self._run(
            [
                "aws",
                "s3",
                "sync",
                aws_bucket_url,
                folder,
            ]
        )