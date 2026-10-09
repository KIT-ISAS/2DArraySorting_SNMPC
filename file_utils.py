import os
import shutil


def copy_file(src, dst):
    """Copies a file from src to dst.

    :param src: A string, the source file path.
    :param dst: A string, the destination path (pointing to a file or directory).
    """
    # create a directory if it does not exist
    dir_path = os.path.dirname(dst)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    shutil.copy(src, dst)
