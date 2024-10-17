import sys

def print_syspath():
    for path in sys.path:
        print(path)

if __name__ == "__main__":
    print_syspath()
