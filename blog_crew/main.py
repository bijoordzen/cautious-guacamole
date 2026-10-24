import sys
from pathlib import Path

# Add src to sys.path so blog_crew can be imported directly
src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from blog_crew.main import run

if __name__ == "__main__":
    run()

