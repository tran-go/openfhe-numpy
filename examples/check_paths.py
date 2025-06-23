import sys
print("Python search paths:")
for path in sys.path:
    print(f"  {path}")
try:
    import openfhe
    print(f"\nOpenFHE is installed at: {openfhe.__file__}")
except ImportError:
    print("\nOpenFHE is not installed")
