import subprocess
import sys
import os

def run_preprocessing(script_dir, file_name):

    datasetDir = os.path.join(script_dir, "../data/Features/")
    LTMGDir = os.path.join(script_dir, "../data/scGNN/")

    cmd = [
        sys.executable, "-W", "ignore", os.path.join(script_dir, "../scGNN/PreprocessingscGNN.py"),
        "--datasetName", file_name,
        "--datasetDir", datasetDir,
        "--LTMGDir", LTMGDir,
        "--filetype", "CSV",
        "--geneSelectnum", "2000",
        "--inferLTMGTag"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Preprocessing completed successfully!")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Preprocessing failed with return code {e.returncode}")
        print(f"Error output: {e.stderr}")
        return False

def run_scgnn(script_dir):

    datasetDir = os.path.join(script_dir, "../data/")
    LTMGDir = os.path.join(script_dir, "../data/")
    outputDir = os.path.join(script_dir, "../data/scGNN/")


    cmd = [
        sys.executable, "-W", "ignore", "../scGNN/scGNN.py",
        "--datasetName", "scGNN",
        "--datasetDir", datasetDir,
        "--LTMGDir", LTMGDir,
        "--outputDir", outputDir,
        "--EM-iteration", "2",
        "--Regu-epochs", "50",
        "--EM-epochs", "20",
        "--regulized-type", "LTMG",
        "--useBothembedding",  
        "--converge_type", "graph"  
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Training completed successfully!")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Training failed with return code {e.returncode}")
        print(f"Error output: {e.stderr}")
        return False

# Run both commands
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if len(sys.argv) < 2:
        print("Usage: python run_scgnn.py <filename>")
        sys.exit(1)
    
    file_name = sys.argv[1]

    # Add debugging prints
    print("Current working directory:", os.getcwd())
    print("Script directory:", script_dir)

    if run_preprocessing(script_dir, file_name):
        run_scgnn(script_dir)