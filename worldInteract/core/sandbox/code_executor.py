"""
Code executor for sandbox environment that supports Python execution and pip install.
"""

import os
import sys
import subprocess
import tempfile
import shutil
import json
import copy
import traceback
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

from worldInteract.utils.config_manager import config_manager


logger = logging.getLogger(__name__)


class CodeExecutor:
    """Secure code executor with Python and pip install support."""
    
    def __init__(self):
        """Initialize code executor."""
        self.config = config_manager.get_environment_config("code_agent")
        self.timeout = self.config.get("sandbox_timeout", 30)
        self.enable_sandbox = self.config.get("enable_sandbox", True)
        
    def execute_code(
        self,
        code: str,
        requirements: List[str],
        test_cases: List[Dict[str, Any]],
        initial_state: Dict[str, Any],
        tool_name: str
    ) -> Tuple[bool, str, List[Dict[str, Any]]]:
        """
        Execute tool code with test cases in sandbox environment.
        
        Args:
            code: Python code to execute
            requirements: List of pip requirements
            test_cases: List of test cases to run
            initial_state: Initial database state
            tool_name: Name of the tool being tested
            
        Returns:
            Tuple of (success, output_message, test_results)
        """
        if not self.enable_sandbox:
            logger.warning("Sandbox execution is disabled, using direct execution")
            return self._execute_directly(code, test_cases, initial_state, tool_name)
            
        # Create temporary workspace
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            
            try:
                # Setup environment
                self._setup_environment(workspace, code, requirements)
                
                # Execute test cases
                test_results = []
                for i, test_case in enumerate(test_cases):
                    logger.info(f"Executing test case {i+1}/{len(test_cases)}")
                    result = self._execute_single_test(
                        workspace, test_case, initial_state, tool_name
                    )
                    test_results.append(result)
                
                # Check if all tests passed
                all_passed = all(result.get("success", False) for result in test_results)
                
                if all_passed:
                    return True, "all test cases run without error", test_results
                else:
                    failed_count = sum(1 for result in test_results if not result.get("success", False))
                    return False, f"{failed_count}/{len(test_results)} test cases failed", test_results
                    
            except Exception as e:
                logger.error(f"Sandbox execution failed: {e}")
                return False, f"Sandbox execution error: {str(e)}", []
    
    def _setup_environment(self, workspace: Path, code: str, requirements: List[str]) -> None:
        """Setup sandbox environment with code and dependencies."""
        # Create main code file
        code_file = workspace / "tool_code.py"
        
        # Add necessary imports
        full_code = self._add_imports(code)
        
        with open(code_file, 'w', encoding='utf-8') as f:
            f.write(full_code)
        
        # Install requirements if any
        if requirements:
            self._install_requirements(workspace, requirements)
    
    def _add_imports(self, code: str) -> str:
        """Add necessary imports to the code."""
        imports = [
            "import json",
            "import uuid",
            "import datetime",
            "import copy",
            "import sys",
            "import os",
            "from typing import Dict, Any, List, Optional",
            "",
        ]
        
        return '\n'.join(imports) + '\n' + code
    
    def _install_requirements(self, workspace: Path, requirements: List[str]) -> None:
        """Install pip requirements in an isolated environment."""
        if not requirements:
            return
            
        # Create requirements.txt
        req_file = workspace / "requirements.txt"
        with open(req_file, 'w', encoding='utf-8') as f:
            for req in requirements:
                f.write(f"{req}\n")
        
        # Create local site-packages directory for isolation
        local_packages = workspace / "site-packages"
        local_packages.mkdir(exist_ok=True)
        
        # Install requirements to local directory (isolated)
        try:
            cmd = [
                sys.executable, "-m", "pip", "install", 
                "-r", str(req_file),
                "--target", str(local_packages),  # Install to local directory
                "--no-deps",  # Don't install dependencies to avoid conflicts
                "--disable-pip-version-check"
            ]
            
            logger.info(f"Installing requirements to isolated directory: {' '.join(cmd)}")
            logger.info(f"Target directory: {local_packages}")
            
            result = subprocess.run(
                cmd,
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=60  # 1 minute timeout for pip install
            )
            
            if result.returncode != 0:
                logger.warning(f"Failed to install requirements. Command: {' '.join(cmd)}")
                logger.warning(f"Error output: {result.stderr}")
                if result.stdout:
                    logger.warning(f"Standard output: {result.stdout}")
                
                # Fallback: try installing without --no-deps
                logger.info("Retrying without --no-deps...")
                cmd_fallback = [
                    sys.executable, "-m", "pip", "install", 
                    "-r", str(req_file),
                    "--target", str(local_packages),
                    "--disable-pip-version-check"
                ]
                
                result_fallback = subprocess.run(
                    cmd_fallback,
                    cwd=workspace,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result_fallback.returncode == 0:
                    logger.info(f"Successfully installed requirements with dependencies: {requirements}")
                else:
                    logger.warning(f"Fallback install also failed: {result_fallback.stderr}")
            else:
                logger.info(f"Successfully installed requirements: {requirements}")
                if result.stdout:
                    logger.debug(f"Install output: {result.stdout}")
                
        except subprocess.TimeoutExpired:
            logger.warning("Pip install timed out")
        except Exception as e:
            logger.warning(f"Error installing requirements: {e}")
    
    def _execute_single_test(
        self,
        workspace: Path,
        test_case: Dict[str, Any],
        initial_state: Dict[str, Any],
        tool_name: str
    ) -> Dict[str, Any]:
        """Execute a single test case."""
        try:
            # Create execution script
            exec_script = self._create_execution_script(
                workspace, test_case, initial_state, tool_name
            )
            
            # Execute with timeout
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._run_script, exec_script, workspace)
                try:
                    result = future.result(timeout=self.timeout)
                    return result
                except FutureTimeoutError:
                    return {
                        "success": False,
                        "error": f"Test execution timed out after {self.timeout} seconds",
                        "test_case": test_case
                    }
                    
        except Exception as e:
            return {
                "success": False,
                "error": f"Test execution failed: {str(e)}",
                "test_case": test_case,
                "traceback": traceback.format_exc()
            }
    
    def _create_execution_script(
        self,
        workspace: Path,
        test_case: Dict[str, Any],
        initial_state: Dict[str, Any],
        tool_name: str
    ) -> Path:
        """Create Python script to execute the test case."""
        script_content = f'''
import sys
import json
import traceback
import copy
from pathlib import Path

# Add workspace and local packages to path
sys.path.insert(0, "{workspace}")
sys.path.insert(0, "{workspace}/site-packages")

try:
    # Import the tool code
    from tool_code import {tool_name}
    
    # Prepare test data - make a copy to track changes
    initial_state = {initial_state}
    current_state = copy.deepcopy(initial_state)
    test_params = {test_case.get("parameters", {})}
    
    # Execute the tool with the mutable state
    result = {tool_name}(current_state, **test_params)
    
    # Parse result if it's a string
    if isinstance(result, str):
        try:
            parsed_result = json.loads(result)
        except json.JSONDecodeError:
            parsed_result = {{"raw_result": result}}
    else:
        parsed_result = result
    
    # Add after_execution_state to the result automatically
    if isinstance(parsed_result, dict):
        parsed_result["after_execution_state"] = current_state
    else:
        # If result is not a dict, wrap it and add state
        parsed_result = {{
            "tool_result": parsed_result,
            "after_execution_state": current_state
        }}
    
    # Prepare success response
    response = {{
        "success": True,
        "result": parsed_result,
        "test_case": {test_case}
    }}
    
    print("EXECUTION_RESULT:", json.dumps(response))

except Exception as e:
    # Prepare error response - include initial state on error
    error_response = {{
        "success": False,
        "error": str(e),
        "test_case": {test_case},
        "traceback": traceback.format_exc(),
        "after_execution_state": {initial_state}
    }}
    
    print("EXECUTION_RESULT:", json.dumps(error_response))
'''
        
        script_file = workspace / "execute_test.py"
        with open(script_file, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        return script_file
    
    def _run_script(self, script_file: Path, workspace: Path) -> Dict[str, Any]:
        """Run the execution script and capture output."""
        try:
            cmd = [sys.executable, str(script_file)]
            result = subprocess.run(
                cmd,
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            # Parse output
            output_lines = result.stdout.strip().split('\n')
            for line in output_lines:
                if line.startswith("EXECUTION_RESULT:"):
                    result_json = line[len("EXECUTION_RESULT:"):].strip()
                    return json.loads(result_json)
            
            # If no result found, return error
            return {
                "success": False,
                "error": "No execution result found in output",
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Script execution timed out after {self.timeout} seconds"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Script execution failed: {str(e)}",
                "traceback": traceback.format_exc()
            }
    
    def _execute_directly(
        self,
        code: str,
        test_cases: List[Dict[str, Any]],
        initial_state: Dict[str, Any],
        tool_name: str
    ) -> Tuple[bool, str, List[Dict[str, Any]]]:
        """Execute code directly without sandbox (fallback method)."""
        try:
            # Prepare execution environment
            exec_globals = {
                'json': json,
                'uuid': __import__('uuid'),
                'datetime': __import__('datetime'),
                'copy': copy,
                'Dict': Dict,
                'Any': Any,
                'List': List,
                'Optional': Optional
            }
            
            # Execute the tool code to define the function
            exec(code, exec_globals)
            
            # Find the tool function
            if tool_name not in exec_globals:
                return False, f"Tool function '{tool_name}' not found in code", []
            
            tool_function = exec_globals[tool_name]
            
            # Execute test cases
            test_results = []
            for i, test_case in enumerate(test_cases):
                try:
                    # Create a copy of initial state for each test
                    current_state = copy.deepcopy(initial_state)
                    test_params = test_case.get("parameters", {})
                    
                    # Execute the tool
                    result = tool_function(current_state, **test_params)
                    
                    # Parse result and add state automatically
                    if isinstance(result, str):
                        try:
                            parsed_result = json.loads(result)
                        except json.JSONDecodeError:
                            parsed_result = {"raw_result": result}
                    else:
                        parsed_result = result
                    
                    # Add after_execution_state to the result automatically
                    if isinstance(parsed_result, dict):
                        parsed_result["after_execution_state"] = current_state
                    else:
                        # If result is not a dict, wrap it and add state
                        parsed_result = {
                            "tool_result": parsed_result,
                            "after_execution_state": current_state
                        }
                    
                    test_results.append({
                        "success": True,
                        "result": parsed_result,
                        "test_case": test_case
                    })
                    
                except Exception as e:
                    test_results.append({
                        "success": False,
                        "error": str(e),
                        "test_case": test_case,
                        "traceback": traceback.format_exc(),
                        "after_execution_state": initial_state  # Use initial state on error
                    })
            
            # Check if all tests passed
            all_passed = all(result.get("success", False) for result in test_results)
            
            if all_passed:
                return True, "ALL TEST CASES PASSED", test_results
            else:
                failed_count = sum(1 for result in test_results if not result.get("success", False))
                return False, f"{failed_count}/{len(test_results)} test cases failed", test_results
                
        except Exception as e:
            logger.error(f"Direct execution failed: {e}")
            return False, f"Direct execution error: {str(e)}", []
