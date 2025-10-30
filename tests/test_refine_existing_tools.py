"""
Test refinement on existing tools.py files

This test is used to:
1. Read existing generated tools.py file
2. Parse out individual tool functions
3. Call refine component for consistency optimization
4. Save refined results
"""

import json
import re
import ast
from pathlib import Path
from typing import Dict, Any, List

from worldInteract.core.build_environment import ToolGenerator


def parse_tools_from_file(tools_file_path: str) -> Dict[str, str]:
    """
    Parse individual tool functions from tools.py file
    
    Args:
        tools_file_path: Path to tools.py file
        
    Returns:
        Dictionary mapping tool names to their code
    """
    with open(tools_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Parse AST
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        print(f"Error parsing tools file: {e}")
        return {}
    
    tools = {}
    
    # Extract all function definitions
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Get function name
            tool_name = node.name
            
            # Get function source code (by line numbers)
            start_line = node.lineno - 1  # AST line numbers start from 1
            end_line = node.end_lineno
            
            lines = content.split('\n')
            tool_code = '\n'.join(lines[start_line:end_line])
            
            tools[tool_name] = tool_code
            print(f"Extracted tool: {tool_name} ({end_line - start_line} lines)")
    
    return tools


def load_api_collection(domain: str = "file_operations") -> Dict[str, Any]:
    """
    Load API collection from domain graph file
    
    Args:
        domain: Domain name (default: file_operations)
        
    Returns:
        API collection
    """
    # Default path for file_operations domain graph
    api_collection_path = Path(__file__).parent.parent / "data" / "domain_graphs" / "bfcl_ticket_domain_graphs" / "domains" / "ticket_management.json"
    
    if not api_collection_path.exists():
        print(f"❌ Error: API collection file not found: {api_collection_path}")
        return None
    
    with open(api_collection_path, 'r', encoding='utf-8') as f:
        api_collection = json.load(f)
    
    return api_collection


def load_test_cases(source: str = "ticket_management") -> Dict[str, List[Dict[str, Any]]]:
    """
    Load test cases from generated environment
    
    Args:
        domain: Domain name (default: file_operations)
        source: Source directory name (default: refined_file_operations)
        
    Returns:
        Test cases dictionary
    """
    # Load from refined_file_operations by default
    test_cases_path = Path(__file__).parent.parent / "data" / "generated_env" / "domains" / source / "test_cases.json"
    
    if not test_cases_path.exists():
        print(f"❌ Error: Test cases file not found: {test_cases_path}")
        return None
    
    with open(test_cases_path, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)
    
    return test_cases


def test_refine_existing_tools():
    """Test refinement on existing tools.py file"""
    
    # Configuration paths
    domain = "ticket_management"
    base_dir = Path(__file__).parent.parent / "data" / "generated_env" / "domains" / domain
    
    tools_file = base_dir / "tools.py"
    schema_file = base_dir / "schema.json"
    initial_state_file = base_dir / "initial_state.json"
    
    print("=" * 80)
    print("Starting Tool Refinement Test")
    print("=" * 80)
    print(f"Domain: {domain}")
    print(f"Tools file: {tools_file}")
    
    # Check if files exist
    if not tools_file.exists():
        print(f"❌ Error: Tools file not found: {tools_file}")
        return
    
    if not schema_file.exists():
        print(f"❌ Error: Schema file not found: {schema_file}")
        return
    
    if not initial_state_file.exists():
        print(f"❌ Error: Initial state file not found: {initial_state_file}")
        return
    
    # 1. Parse tools.py file
    print("\n" + "-" * 80)
    print("Step 1: Parse tools.py file")
    print("-" * 80)
    tools = parse_tools_from_file(str(tools_file))
    print(f"✓ Successfully parsed {len(tools)} tools")
    
    # Show tool list
    tool_names = list(tools.keys())
    print(f"Tool list: {', '.join(tool_names)}")
    
    # 2. Load schema and initial_state
    print("\n" + "-" * 80)
    print("Step 2: Load schema and initial_state")
    print("-" * 80)
    
    with open(schema_file, 'r', encoding='utf-8') as f:
        schema = json.load(f)
    print(f"✓ Loaded schema: {len(schema)} tables")
    
    with open(initial_state_file, 'r', encoding='utf-8') as f:
        initial_state = json.load(f)
    print(f"✓ Loaded initial_state: {len(initial_state)} tables")
    
    # 3. Load API collection and test cases
    print("\n" + "-" * 80)
    print("Step 3: Load API collection and test cases")
    print("-" * 80)
    
    # Load API collection from domain graph
    api_collection = load_api_collection(domain)
    if not api_collection:
        print("❌ Failed to load API collection")
        return
    print(f"✓ Loaded API collection: {len(api_collection['tools'])} tools")
    
    # Load test cases from refined_file_operations
    test_cases = load_test_cases(source=domain)
    if not test_cases:
        print("❌ Failed to load test cases")
        return
    print(f"✓ Loaded test cases: {len(test_cases)} tools")
    
    # 4. Call refinement
    print("\n" + "-" * 80)
    print("Step 4: Execute Tool Refinement")
    print("-" * 80)
    print("Note: This will call LLM for code analysis and optimization...")
    print("Estimated time: 30-60 seconds")
    
    tool_generator = ToolGenerator()
    
    try:
        refined_tools, refinement_changes = tool_generator.refine_tools(
            tools=tools,
            api_collection=api_collection,
            schema=schema,
            initial_state=initial_state,
            test_cases=test_cases,
            requirements=[]
        )
        
        print(f"✓ Refinement completed")
        print(f"  - Refined tools: {len(refined_tools)}")
        print(f"  - Tools with changes: {len(refinement_changes)}")
        
    except Exception as e:
        print(f"❌ Refinement failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 5. Show change details
    if refinement_changes:
        print("\n" + "-" * 80)
        print("Step 5: Refinement change details")
        print("-" * 80)
        
        for tool_name, changes in refinement_changes.items():
            print(f"\nTool: {tool_name}")
            for change in changes:
                print(f"  ✓ {change}")
    else:
        print("\n⚠️  Warning: No refinement changes or refinement failed")
    
    # 6. Save refined results
    print("\n" + "-" * 80)
    print("Step 6: Save refined results")
    print("-" * 80)
    
    # Create output directory
    output_dir = base_dir / "refined_output"
    output_dir.mkdir(exist_ok=True)
    
    # Save refined tools with consolidated imports
    refined_tools_file = output_dir / "tools_refined.py"
    
    # Extract and deduplicate imports from all tools
    all_imports = set()
    tool_functions = {}
    
    for tool_name, tool_code in refined_tools.items():
        # Split code into lines
        lines = tool_code.split('\n')
        function_lines = []
        
        in_imports = True
        for line in lines:
            stripped = line.strip()
            # Check if this is an import line
            if in_imports and (stripped.startswith('import ') or 
                              stripped.startswith('from ') or
                              stripped.startswith('#') or
                              stripped == ''):
                if stripped.startswith('import ') or stripped.startswith('from '):
                    all_imports.add(stripped)
                # Skip comments and empty lines in import section
            else:
                in_imports = False
                function_lines.append(line)
        
        # Store the function code without imports
        tool_functions[tool_name] = '\n'.join(function_lines).strip()
    
    # Sort imports for consistency (alphabetically)
    sorted_imports = sorted(all_imports)
    
    # Build header with only the imports found in tool codes
    header = sorted_imports + [
        "",
        '"""',
        "Refined tool implementations for domain operations.",
        f"Domain: {domain}",
        f"Total tools: {len(refined_tools)}",
        '"""',
        ""
    ]
    
    # Write the combined file
    with open(refined_tools_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(header))
        
        for tool_name, function_code in tool_functions.items():
            f.write(f"\n\n{function_code}\n")
    
    print(f"✓ Refined tools saved to: {refined_tools_file}")
    
    # Save refinement changes
    changes_file = output_dir / "refinement_changes.json"
    with open(changes_file, 'w', encoding='utf-8') as f:
        json.dump(refinement_changes, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Refinement changes saved to: {changes_file}")
    
    # Save comparison report
    report_file = output_dir / "refinement_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Tool Refinement Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Domain: {domain}\n")
        f.write(f"Original tools file: {tools_file}\n")
        f.write(f"Refined tools file: {refined_tools_file}\n\n")
        f.write(f"Total tools: {len(refined_tools)}\n")
        f.write(f"Tools with changes: {len(refinement_changes)}\n\n")
        
        if refinement_changes:
            f.write("-" * 80 + "\n")
            f.write("Detailed Changes:\n")
            f.write("-" * 80 + "\n\n")
            
            for tool_name, changes in refinement_changes.items():
                f.write(f"Tool: {tool_name}\n")
                for change in changes:
                    f.write(f"  - {change}\n")
                f.write("\n")
    
    print(f"✓ Refinement report saved to: {report_file}")
    
    # 7. Summary
    print("\n" + "=" * 80)
    print("Test Completed!")
    print("=" * 80)
    print(f"✓ Original tools: {len(tools)}")
    print(f"✓ Refined tools: {len(refined_tools)}")
    print(f"✓ Tools with changes: {len(refinement_changes)}")
    print(f"\nOutput directory: {output_dir}")
    print("\nNext steps:")
    print("1. View tools_refined.py and compare with original file")
    print("2. View refinement_changes.json to understand specific changes")
    print("3. View refinement_report.txt for complete report")
    print("\nComparison command:")
    print(f"  diff {tools_file} {refined_tools_file}")


def test_refine_specific_domain(domain: str, source_dir: str = None):
    """
    Test refinement for a specific domain
    
    Args:
        domain: Domain name (e.g., "ticket_management", "file_operations")
        source_dir: Source directory for tools.py, schema.json, initial_state.json 
                   (default: same as domain name)
    """
    if source_dir is None:
        source_dir = domain
    
    base_dir = Path(__file__).parent.parent / "data" / "generated_env" / "domains" / source_dir
    
    if not base_dir.exists():
        print(f"❌ Error: Domain directory not found: {base_dir}")
        return
    
    # Configuration paths
    tools_file = base_dir / "tools.py"
    schema_file = base_dir / "schema.json"
    initial_state_file = base_dir / "initial_state.json"
    
    print("=" * 80)
    print(f"Starting Tool Refinement Test for {domain}")
    print("=" * 80)
    print(f"Domain: {domain}")
    print(f"Source directory: {source_dir}")
    print(f"Tools file: {tools_file}")
    
    # Check if files exist
    if not tools_file.exists():
        print(f"❌ Error: Tools file not found: {tools_file}")
        return
    
    if not schema_file.exists():
        print(f"❌ Error: Schema file not found: {schema_file}")
        return
    
    if not initial_state_file.exists():
        print(f"❌ Error: Initial state file not found: {initial_state_file}")
        return
    
    # 1. Parse tools.py file
    print("\n" + "-" * 80)
    print("Step 1: Parse tools.py file")
    print("-" * 80)
    tools = parse_tools_from_file(str(tools_file))
    print(f"✓ Successfully parsed {len(tools)} tools")
    
    # Show tool list
    tool_names = list(tools.keys())
    print(f"Tool list: {', '.join(tool_names)}")
    
    # 2. Load schema and initial_state
    print("\n" + "-" * 80)
    print("Step 2: Load schema and initial_state")
    print("-" * 80)
    
    with open(schema_file, 'r', encoding='utf-8') as f:
        schema = json.load(f)
    print(f"✓ Loaded schema: {len(schema)} tables")
    
    with open(initial_state_file, 'r', encoding='utf-8') as f:
        initial_state = json.load(f)
    print(f"✓ Loaded initial_state: {len(initial_state)} tables")
    
    # 3. Load API collection and test cases
    print("\n" + "-" * 80)
    print("Step 3: Load API collection and test cases")
    print("-" * 80)
    
    # Load API collection from domain graph
    api_collection = load_api_collection(domain)
    if not api_collection:
        print("❌ Failed to load API collection")
        return
    print(f"✓ Loaded API collection: {len(api_collection['tools'])} tools")
    
    # Try to load test cases, fallback if not found
    test_cases = load_test_cases(domain, source=source_dir)
    if not test_cases:
        print("⚠️  Warning: Could not load test cases, trying alternate location...")
        # Try refined_{domain}
        test_cases = load_test_cases(domain, source=f"refined_{domain}")
        if not test_cases:
            print("❌ Failed to load test cases from any location")
            return
    print(f"✓ Loaded test cases: {len(test_cases)} tools")
    
    # 4. Execute refinement (same as in test_refine_existing_tools)
    print("\n" + "-" * 80)
    print("Step 4: Execute Tool Refinement")
    print("-" * 80)
    print("Note: This will call LLM for code analysis and optimization...")
    print("Estimated time: 30-60 seconds")
    
    tool_generator = ToolGenerator()
    
    try:
        refined_tools, refinement_changes = tool_generator.refine_tools(
            tools=tools,
            api_collection=api_collection,
            schema=schema,
            initial_state=initial_state,
            test_cases=test_cases,
            requirements=[]
        )
        
        print(f"✓ Refinement completed")
        print(f"  - Refined tools: {len(refined_tools)}")
        print(f"  - Tools with changes: {len(refinement_changes)}")
        
    except Exception as e:
        print(f"❌ Refinement failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 5. Show change details
    if refinement_changes:
        print("\n" + "-" * 80)
        print("Step 5: Refinement change details")
        print("-" * 80)
        
        for tool_name, changes in refinement_changes.items():
            print(f"\nTool: {tool_name}")
            for change in changes:
                print(f"  ✓ {change}")
    else:
        print("\n⚠️  Warning: No refinement changes or refinement failed")
    
    # 6. Save refined results
    print("\n" + "-" * 80)
    print("Step 6: Save refined results")
    print("-" * 80)
    
    # Create output directory
    output_dir = base_dir / "refined_output"
    output_dir.mkdir(exist_ok=True)
    
    # Save refined tools with consolidated imports
    refined_tools_file = output_dir / "tools_refined.py"
    
    # Extract and deduplicate imports from all tools
    all_imports = set()
    tool_functions = {}
    
    for tool_name, tool_code in refined_tools.items():
        # Split code into lines
        lines = tool_code.split('\n')
        function_lines = []
        
        in_imports = True
        for line in lines:
            stripped = line.strip()
            # Check if this is an import line
            if in_imports and (stripped.startswith('import ') or 
                              stripped.startswith('from ') or
                              stripped.startswith('#') or
                              stripped == ''):
                if stripped.startswith('import ') or stripped.startswith('from '):
                    all_imports.add(stripped)
                # Skip comments and empty lines in import section
            else:
                in_imports = False
                function_lines.append(line)
        
        # Store the function code without imports
        tool_functions[tool_name] = '\n'.join(function_lines).strip()
    
    # Sort imports for consistency (alphabetically)
    sorted_imports = sorted(all_imports)
    
    # Build header with only the imports found in tool codes
    header = sorted_imports + [
        "",
        '"""',
        "Refined tool implementations for domain operations.",
        f"Domain: {domain}",
        f"Total tools: {len(refined_tools)}",
        '"""',
        ""
    ]
    
    # Write the combined file
    with open(refined_tools_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(header))
        
        for tool_name, function_code in tool_functions.items():
            f.write(f"\n\n{function_code}\n")
    
    print(f"✓ Refined tools saved to: {refined_tools_file}")
    
    # Save refinement changes
    changes_file = output_dir / "refinement_changes.json"
    with open(changes_file, 'w', encoding='utf-8') as f:
        json.dump(refinement_changes, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Refinement changes saved to: {changes_file}")
    
    # Save comparison report
    report_file = output_dir / "refinement_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Tool Refinement Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Domain: {domain}\n")
        f.write(f"Source directory: {source_dir}\n")
        f.write(f"Original tools file: {tools_file}\n")
        f.write(f"Refined tools file: {refined_tools_file}\n\n")
        f.write(f"Total tools: {len(refined_tools)}\n")
        f.write(f"Tools with changes: {len(refinement_changes)}\n\n")
        
        if refinement_changes:
            f.write("-" * 80 + "\n")
            f.write("Detailed Changes:\n")
            f.write("-" * 80 + "\n\n")
            
            for tool_name, changes in refinement_changes.items():
                f.write(f"Tool: {tool_name}\n")
                for change in changes:
                    f.write(f"  - {change}\n")
                f.write("\n")
    
    print(f"✓ Refinement report saved to: {report_file}")
    
    # 7. Summary
    print("\n" + "=" * 80)
    print("Test Completed!")
    print("=" * 80)
    print(f"✓ Original tools: {len(tools)}")
    print(f"✓ Refined tools: {len(refined_tools)}")
    print(f"✓ Tools with changes: {len(refinement_changes)}")
    print(f"\nOutput directory: {output_dir}")
    print("\nNext steps:")
    print("1. View tools_refined.py and compare with original file")
    print("2. View refinement_changes.json to understand specific changes")
    print("3. View refinement_report.txt for complete report")
    print("\nComparison command:")
    print(f"  diff {tools_file} {refined_tools_file}")


if __name__ == "__main__":
    import sys
    
    # Support command line arguments for domain and source directory specification
    # Usage:
    #   python test_refine_existing_tools.py
    #   python test_refine_existing_tools.py <domain>
    #   python test_refine_existing_tools.py <domain> <source_dir>
    
    if len(sys.argv) > 1:
        domain = sys.argv[1]
        source_dir = sys.argv[2] if len(sys.argv) > 2 else None
        print(f"Testing refinement for domain: {domain}")
        if source_dir:
            print(f"Using source directory: {source_dir}")
        test_refine_specific_domain(domain, source_dir)
    else:
        # Default test file_operations
        print("Testing refinement for default domain: file_operations")
        print("Tip: Use 'python test_refine_existing_tools.py <domain> [<source_dir>]' to test other domains")
        print()
        test_refine_existing_tools()

