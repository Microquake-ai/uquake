from pathlib import Path

def write_rst_for_package(root_path: Path, rst_file, project_root: Path):
    for child in sorted(root_path.iterdir()):
        if child.is_dir():
            write_rst_for_package(child, rst_file, project_root)
        elif child.suffix == '.py' and child.stem != '__init__':
            module_path = child.relative_to(project_root).with_suffix('')
            module_name = str(module_path).replace('/', '.')

            rst_file.write(f".. automodule:: {module_name}\n")
            rst_file.write("   :members:\n\n")


if __name__ == '__main__':
    output_rst_path = 'source/autogen_modules.rst'
    root_package_path = Path('../uquake')
    project_root_path = Path('../')  # Or wherever the root of your Python package is

    with open(output_rst_path, 'w') as rst_file:
        rst_file.write("Package Modules\n")
        rst_file.write("===================\n\n")

        write_rst_for_package(root_package_path, rst_file, project_root_path)
