# numpy to csv conversion

import numpy as np
import pandas as pd

# Path to your .npy file
npy_file_path = "x_train.npy"

# Load .npy file
data = np.load(npy_file_path)

# Convert numpy array to pandas DataFrame
df = pd.DataFrame(data)

# Path to save the .csv file
csv_file_path = "x_train.csv"

# Save DataFrame to .csv file
df.to_csv(csv_file_path, index=False)

print("Conversion completed successfully!")

# CSV to numpy conversion
import numpy as np
import pandas as pd

# Path to your CSV file
csv_file_path = "x_train.csv"

# Read CSV file into pandas DataFrame
df = pd.read_csv(csv_file_path)

# Convert DataFrame to numpy array
data = df.to_numpy()

# Path to save the .npy file
npy_file_path = "x_train.npy"

# Save numpy array to .npy file
np.save(npy_file_path, data)

print("Conversion completed successfully!")
