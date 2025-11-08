import argparse
from pathlib import Path

def  parse_args():
    p = argparse.ArgumentParser(description="tiny argument parser demo")
    p.add_argument("--num", type=int, default=10, help="an integer, default=10")
    p.add_argument("--name", type=str, default="Alex", help="a str, default='Alex'")
    p.add_argument("--name_2", type=str, required=True, help="required second name")
    p.add_argument("--infile", type=Path, help="path to an input file")
    p.add_argument("--flag", action="store_true", help="boolean switch (default False)")
    return p.parse_args()

def main():
    args = parse_args()
    print("num:", args.num)
    print("name:", args.name)
    print("name_2:", args.name_2)
    print("infile:",  args.infile)
    print("flag:", args.flag)
    

if __name__ == "__main__":
    main()