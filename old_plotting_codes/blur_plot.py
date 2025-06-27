import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter

import sys

plot_filename = sys.argv[1]
# Step 2: Open the saved plot image and apply blur
image = Image.open(plot_filename)
blurred_image = image.filter(ImageFilter.GaussianBlur(radius=5))

# Save the blurred image
blurred_plot_filename = plot_filename.replace('.png', '_blurred.png')
assert blurred_plot_filename != plot_filename
blurred_image.save(blurred_plot_filename)

print(f"Original plot saved as {plot_filename}")
print(f"Blurred plot saved as {blurred_plot_filename}")