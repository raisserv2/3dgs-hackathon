import subprocess

def create_requirements_txt(filename="requirements.txt"):
    try:
        # Runs 'pip freeze' and captures the output
        result = subprocess.run(['pip', 'freeze'], capture_output=True, text=True, check=True)
        
        # Writes the output to the specified file
        with open(filename, "w") as f:
            f.write(result.stdout)
            
        print(f"Successfully created {filename}")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    create_requirements_txt()
