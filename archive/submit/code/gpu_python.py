"""Launch Python with pip NVIDIA library paths; works as a normal launcher on Mac."""
import os
import sys
from pathlib import Path
from importlib import metadata


def main():
    folders = set()
    for dist in metadata.distributions():
        if dist.metadata.get('Name', '').lower().startswith('nvidia-'):
            for entry in dist.files or []:
                path = Path(dist.locate_file(entry))
                if '.so' in path.name and path.is_file():
                    folders.add(str(path.parent.resolve()))
    env = os.environ.copy()
    paths = sorted(folders)
    if env.get('LD_LIBRARY_PATH'):
        paths.append(env['LD_LIBRARY_PATH'])
    if paths:
        env['LD_LIBRARY_PATH'] = ':'.join(paths)
    os.execve(sys.executable, [sys.executable, *sys.argv[1:]], env)


if __name__ == '__main__':
    main()
