"""
Setup Verification Script
Run this script to verify that all dependencies are correctly installed
"""

import sys

def check_python_version():
    """Check Python version"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 9:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} (OK)")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} (Need 3.9+)")
        return False

def check_dependencies():
    """Check if all required dependencies are installed"""
    dependencies = {
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'tensorflow': 'TensorFlow',
        'keras': 'Keras',
        'cv2': 'OpenCV',
        'PIL': 'Pillow',
        'sklearn': 'Scikit-learn',
        'flask': 'Flask',
        'fastapi': 'FastAPI',
        'uvicorn': 'Uvicorn',
    }
    
    print("\nChecking dependencies...")
    all_ok = True
    
    for module, name in dependencies.items():
        try:
            if module == 'cv2':
                import cv2
                version = cv2.__version__
            elif module == 'PIL':
                from PIL import Image
                version = Image.__version__ if hasattr(Image, '__version__') else 'installed'
            elif module == 'sklearn':
                import sklearn
                version = sklearn.__version__
            else:
                mod = __import__(module)
                version = mod.__version__ if hasattr(mod, '__version__') else 'installed'
            
            print(f"✓ {name:20s} {version}")
        except ImportError:
            print(f"✗ {name:20s} NOT INSTALLED")
            all_ok = False
    
    return all_ok

def check_gpu():
    """Check if GPU is available"""
    print("\nChecking GPU availability...")
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✓ GPU available: {len(gpus)} device(s)")
            for gpu in gpus:
                print(f"  - {gpu.name}")
            return True
        else:
            print("⚠ No GPU detected (CPU will be used)")
            return False
    except Exception as e:
        print(f"⚠ Could not check GPU: {e}")
        return False

def check_directories():
    """Check if required directories exist"""
    print("\nChecking project structure...")
    import os
    
    required_dirs = [
        'task1_problem_definition',
        'task2_data_preprocessing',
        'task3_model_training',
        'task4_evaluation',
        'task5_frontend',
        'task6_advanced_optimization',
        'task7_deployment',
        'models',
        'archive',
    ]
    
    all_ok = True
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✓ {dir_name}")
        else:
            print(f"✗ {dir_name} (missing)")
            all_ok = False
    
    return all_ok

def check_dataset():
    """Check if dataset is available"""
    print("\nChecking dataset...")
    import os
    
    images_dir = 'archive/images'
    annotations_dir = 'archive/annotations'
    
    if os.path.exists(images_dir):
        image_count = len([f for f in os.listdir(images_dir) if f.endswith('.png')])
        print(f"✓ Images directory: {image_count} images")
    else:
        print(f"✗ Images directory not found")
        return False
    
    if os.path.exists(annotations_dir):
        annotation_count = len([f for f in os.listdir(annotations_dir) if f.endswith('.xml')])
        print(f"✓ Annotations directory: {annotation_count} annotations")
    else:
        print(f"✗ Annotations directory not found")
        return False
    
    return True

def test_import_scripts():
    """Test if main scripts can be imported"""
    print("\nTesting script imports...")
    
    scripts = {
        'task1_problem_definition.analyze_dataset': 'Task 1',
        'task2_data_preprocessing.preprocess_data': 'Task 2',
        'task3_model_training.train_model': 'Task 3',
        'task4_evaluation.evaluate_model': 'Task 4',
    }
    
    all_ok = True
    for script, name in scripts.items():
        try:
            __import__(script)
            print(f"✓ {name} script")
        except Exception as e:
            print(f"✗ {name} script: {e}")
            all_ok = False
    
    return all_ok

def main():
    """Run all verification checks"""
    print("=" * 60)
    print("FACE MASK DETECTION - SETUP VERIFICATION")
    print("=" * 60)
    
    results = []
    
    # Run checks
    results.append(("Python Version", check_python_version()))
    results.append(("Dependencies", check_dependencies()))
    results.append(("GPU", check_gpu()))
    results.append(("Directories", check_directories()))
    results.append(("Dataset", check_dataset()))
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    for name, status in results:
        status_str = "✓ PASS" if status else "✗ FAIL"
        print(f"{name:20s}: {status_str}")
    
    all_passed = all(status for _, status in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL CHECKS PASSED!")
        print("You are ready to run the project.")
        print("\nNext steps:")
        print("1. python task1_problem_definition/analyze_dataset.py")
        print("2. python task2_data_preprocessing/preprocess_data.py")
        print("3. python task3_model_training/train_model.py")
    else:
        print("✗ SOME CHECKS FAILED!")
        print("Please fix the issues above before proceeding.")
        print("\nCommon solutions:")
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Ensure dataset is in the archive/ directory")
        print("3. Check Python version (need 3.9+)")
    print("=" * 60)

if __name__ == "__main__":
    main()
