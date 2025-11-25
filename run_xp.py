import os

# Define the values of eps
## To do 14
eps_values = [10**-1, 10**-2, 10**-3]  # Example values; replace with actual desired values

# Loop through each value and execute the script
for eps in eps_values:
    print(f"Running evaluate.py with eps={eps}")
    command = f"python evaluate.py --path cnn --model cnn --attack fgsm --epsilon {eps}"
    os.system(command)
