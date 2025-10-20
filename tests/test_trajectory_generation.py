"""
Test script for trajectory generation modules.
"""

import os
import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_imports():
    """Test if all modules can be imported."""
    print("Testing imports...")
    
    try:
        from worldInteract.trajectories import TaskPreparer, TaskAgent, TrajectoryGenerator
        print("✓ All modules imported successfully")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_task_preparer():
    """Test TaskPreparer module."""
    print("\nTesting TaskPreparer...")
    
    try:
        from worldInteract.trajectories import TaskPreparer
        
        # Create instance
        preparer = TaskPreparer()
        print("✓ TaskPreparer initialized successfully")
        
        # Load test data
        random_walk_path = os.path.join(
            project_root,
            "data/random_walks/file_operations_random_walks/0c4a385e-28ce-4170-b1da-1a3b6920fcad.json"
        )
        
        if not os.path.exists(random_walk_path):
            print(f"⚠ Test data not found: {random_walk_path}")
            return True  # Skip but don't fail
        
        with open(random_walk_path, 'r', encoding='utf-8') as f:
            random_walk = json.load(f)
        
        domain_tools_path = os.path.join(
            project_root,
            "data/domain_graphs/my_domain_graphs/domains/file_operations.json"
        )
        
        if not os.path.exists(domain_tools_path):
            print(f"⚠ Domain tools not found: {domain_tools_path}")
            return True  # Skip but don't fail
        
        with open(domain_tools_path, 'r', encoding='utf-8') as f:
            domain_tools = json.load(f)
        
        print("✓ Test data loaded successfully")
        print(f"  Random walk: {random_walk['id']}")
        print(f"  Sequence: {random_walk['sequence']}")
        
        # Note: We don't actually call generate_user_queries here to avoid API calls
        print("✓ TaskPreparer test passed (API calls skipped)")
        return True
        
    except Exception as e:
        print(f"✗ TaskPreparer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_task_agent():
    """Test TaskAgent module."""
    print("\nTesting TaskAgent...")
    
    try:
        from worldInteract.trajectories import TaskAgent
        
        domain_tools_path = os.path.join(
            project_root,
            "data/domain_graphs/my_domain_graphs/domains/file_operations.json"
        )
        
        env_domain_path = os.path.join(
            project_root,
            "data/generated_env/domains/file_operations"
        )
        
        if not os.path.exists(domain_tools_path):
            print(f"⚠ Domain tools not found: {domain_tools_path}")
            return True
        
        if not os.path.exists(env_domain_path):
            print(f"⚠ Environment not found: {env_domain_path}")
            return True
        
        with open(domain_tools_path, 'r', encoding='utf-8') as f:
            domain_tools = json.load(f)
        
        # Create instance
        agent = TaskAgent(domain_tools, env_domain_path)
        print("✓ TaskAgent initialized successfully")
        print(f"  Domain: {agent.domain_tools['domain']}")
        print(f"  Environment: {env_domain_path}")
        print(f"  Initial state loaded: {len(agent.initial_state)} tables")
        
        # Note: We don't actually call execute_task here to avoid API calls
        print("✓ TaskAgent test passed (API calls skipped)")
        return True
        
    except Exception as e:
        print(f"✗ TaskAgent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_trajectory_generator():
    """Test TrajectoryGenerator module."""
    print("\nTesting TrajectoryGenerator...")
    
    try:
        from worldInteract.trajectories import TrajectoryGenerator
        
        domain_tools_path = os.path.join(
            project_root,
            "data/domain_graphs/my_domain_graphs/domains/file_operations.json"
        )
        
        env_domain_path = os.path.join(
            project_root,
            "data/generated_env/domains/file_operations"
        )
        
        if not os.path.exists(domain_tools_path):
            print(f"⚠ Domain tools not found: {domain_tools_path}")
            return True
        
        if not os.path.exists(env_domain_path):
            print(f"⚠ Environment not found: {env_domain_path}")
            return True
        
        # Create instance
        generator = TrajectoryGenerator(domain_tools_path, env_domain_path)
        print("✓ TrajectoryGenerator initialized successfully")
        print(f"  Domain: {generator.domain_tools['domain']}")
        print(f"  Tools count: {len(generator.domain_tools['tools'])}")
        
        # Note: We don't actually call generate_trajectory here to avoid API calls
        print("✓ TrajectoryGenerator test passed (API calls skipped)")
        return True
        
    except Exception as e:
        print(f"✗ TrajectoryGenerator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*80)
    print("Trajectory Generation Module Tests")
    print("="*80)
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("TaskPreparer", test_task_preparer()))
    results.append(("TaskAgent", test_task_agent()))
    results.append(("TrajectoryGenerator", test_trajectory_generator()))
    
    # Print summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{name:30} {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

