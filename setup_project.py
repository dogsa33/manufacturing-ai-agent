import subprocess
import sys


STEPS = [
    (
        "Dataset Download",
        [sys.executable, "-m", "src.download_data"],
    ),
    (
        "DuckDB Setup",
        [sys.executable, "-m", "src.setup_database"],
    ),
    (
        "Model Training",
        [sys.executable, "-m", "src.train_model"],
    ),
]


def run_step(name, command):
    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    subprocess.run(
        command,
        check=True,
    )


def main():
    print("=" * 70)
    print("Manufacturing AI Agent - Project Setup")
    print("=" * 70)

    for name, command in STEPS:
        run_step(
            name,
            command,
        )

    print()
    print("=" * 70)
    print("SETUP COMPLETE")
    print("=" * 70)

    print()
    print("Run Streamlit:")
    print(
        "python -m streamlit run "
        ".\\app\\streamlit_app.py"
    )


if __name__ == "__main__":
    main()