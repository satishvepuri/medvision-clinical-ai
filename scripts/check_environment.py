import platform
import sys

print("MedVision environment check")
print("---------------------------")
print("Python:", sys.version.split()[0])
print("Operating system:", platform.platform())

modules = [
    "torch", "transformers", "peft", "cv2", "mlflow",
    "fastapi", "pandas", "sklearn", "PIL"
]

failed = []
for name in modules:
    try:
        module = __import__(name)
        version = getattr(module, "__version__", "installed")
        print(f"[OK] {name}: {version}")
    except Exception as exc:
        failed.append((name, str(exc)))
        print(f"[MISSING] {name}: {exc}")

if failed:
    raise SystemExit("\nSome packages are missing. Run setup_windows.bat again.")

print("\nEnvironment looks ready.")
