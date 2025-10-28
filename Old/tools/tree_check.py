import os, sys

def print_tree(root: str, max_depth: int = 3):
    root = os.path.abspath(root)
    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath[len(root):].count(os.sep)
        if depth > max_depth:
            # prune deeper traversal
            dirnames[:] = []
            continue
        indent = "  " * depth
        base = os.path.basename(dirpath) or dirpath
        print(f"{indent}📁 {base}")
        for name in sorted(filenames):
            print(f"{indent}  └─ {name}")

def main():
    print("== tree_check ==\n")
    cwd = os.getcwd()
    print(f"CWD: {cwd}\n")
    print_tree(cwd, max_depth=3)

if __name__ == "__main__":
    main()
