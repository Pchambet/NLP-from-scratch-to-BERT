import sys
import importlib.util

def check_import(module_name):
    if importlib.util.find_spec(module_name) is not None:
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", "unknown")
            print(f"✅ {module_name} is installed (version: {version})")
            return True
        except Exception as e:
            print(f"❌ {module_name} is installed but could not be imported: {e}")
            return False
    else:
        print(f"❌ {module_name} is NOT installed")
        return False

required_libraries = [
    "pandas",
    "numpy",
    "nltk",
    "sklearn",
    "gensim",
    "matplotlib",
    "seaborn",
    "transformers",
    "datasets",
    "torch",
    "accelerate"
]

print("--- NLP Project Environment Verification ---\n")

all_good = True
for lib in required_libraries:
    if not check_import(lib):
        all_good = False

print("\n-------------------------------------------")
if all_good:
    print("✨ All dependencies are correctly installed! You are ready to go.")
else:
    print("⚠️ Some dependencies are missing or broken. Please check the output above.")
    sys.exit(1)
