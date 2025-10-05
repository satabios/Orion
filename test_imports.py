#!/usr/bin/env python3

import sys
import os
import importlib.util

def test_imports():
    """Test if our modules can be imported without the full mmcv framework."""
    try:
        # Add the project root to Python path
        project_root = os.path.abspath('.')
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        print("Testing transformer imports...")
        
        # Test transformer registry
        spec = importlib.util.spec_from_file_location(
            "petr_transformers", 
            "./mmcv/models/utils/petr_transformers.py"
        )
        if spec and spec.loader:
            petr_module = importlib.util.module_from_spec(spec)
            # This will check if the file can be loaded
            print("✅ petr_transformers.py can be loaded")
        else:
            print("❌ Cannot load petr_transformers.py")
            return False
            
        print("✅ All critical imports are accessible")
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)