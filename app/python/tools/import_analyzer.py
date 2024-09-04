import os
import ast
from typing import Dict, List, Tuple, Optional

def analyze_imports(project_path: str) -> Dict[str, List[Tuple[str, str]]]:
    """
    Scan the project for imports that violate the "import from definition" principle.

    Args:
    project_path (str): The root path of the project to analyze.

    Returns:
    Dict[str, List[Tuple[str, str]]]: A dictionary where keys are file paths and values are
    lists of tuples containing (import_statement, violation_type).
    """
    violations = {}

    for root, _, files in os.walk(project_path):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                file_violations = analyze_file_imports(file_path, project_path)
                if file_violations:
                    violations[file_path] = file_violations

    return violations

def analyze_file_imports(file_path: str, project_path: str) -> List[Tuple[str, str]]:
    """
    Analyze imports in a single file.

    Args:
    file_path (str): The path of the file to analyze.
    project_path (str): The root path of the project.

    Returns:
    List[Tuple[str, str]]: A list of tuples containing (import_statement, violation_type).
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        try:
            tree = ast.parse(file.read())
        except SyntaxError:
            return [("SyntaxError in file", "parsing_error")]

    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                violation = check_import_violation(alias.name, file_path, project_path)
                if violation:
                    violations.append((f"import {alias.name}", violation))
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:  # absolute import
                violation = check_import_violation(node.module, file_path, project_path)
                if violation:
                    imports = ', '.join(alias.name for alias in node.names)
                    violations.append((f"from {node.module} import {imports}", violation))

    return violations

def check_import_violation(module: str, file_path: str, project_path: str) -> Optional[str]:
    """
    Check if an import violates the "import from definition" principle.

    Args:
    module (str): The name of the imported module.
    file_path (str): The path of the file containing the import.
    project_path (str): The root path of the project.

    Returns:
    Optional[str]: The type of violation, or None if no violation.
    """
    if module.startswith('.'):
        return None  # Relative imports are allowed

    module_parts = module.split('.')
    if module_parts[0] in ['app', 'python']:
        # Check if the module exists in the project structure
        module_path = os.path.join(project_path, *module_parts) + '.py'
        if os.path.exists(module_path):
            return None
        return "internal_absolute_import"

    # Check if it's a standard library or installed package
    try:
        __import__(module_parts[0])
        return None
    except ImportError:
        pass

    return "external_import"

def print_violations(violations: Dict[str, List[Tuple[str, str]]]):
    """
    Print the violations in a readable format.

    Args:
    violations (Dict[str, List[Tuple[str, str]]]): The dictionary of violations.
    """
    if not violations:
        print("No violations found.")
        return

    print("Import violations found:")
    for file_path, file_violations in violations.items():
        print(f"\nIn {file_path}:")
        for violation, violation_type in file_violations:
            print(f"  - {violation} ({violation_type})")

if __name__ == "__main__":
    project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    violations = analyze_imports(project_path)
    print_violations(violations)
