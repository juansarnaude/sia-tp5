from PIL import Image, ImageDraw, ImageFont

# List of emojis
emoji_chars = [
    "😨","😭","😱","😖",
    "😤","🤬","💀","😎"
]
# "😨","😭","😱","😖",
# "😤","🤬","💀","😎",

# Image size
emoji_size = 20  # Set each emoji size to 20x20
rows, cols = 2,4   # 5 rows and 6 columns (with last row having 3 emojis)

# Calculate total image size
width, height = cols * emoji_size, rows * emoji_size

# Create a new white image for the grid
img = Image.new('RGB', (width, height), color=(255, 255, 255))

# Load the font (assuming Segoe UI Emoji is installed)
try:
    font = ImageFont.truetype("C:\\Windows\\Fonts\\seguiemj.ttf", emoji_size)
except IOError:
    print("Segoe UI Emoji font not found! Please ensure it's installed.")
    font = ImageFont.load_default()

# Function to draw an emoji at a specific position
def draw_emoji(emoji, x, y):
    # Create a 20x20 image for each emoji
    emoji_img = Image.new('RGB', (emoji_size, emoji_size), color=(255, 255, 255))
    draw = ImageDraw.Draw(emoji_img)

    # Calculate the width and height of the emoji using textbbox
    bbox = draw.textbbox((x, y), emoji, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Adjust vertical placement to move the emoji down
    y_offset = emoji_size - text_height + 5.5  # Move the emoji slightly down by adding a small offset

    # Positioning the emoji within the 20x20 box
    x_offset = (emoji_size - text_width) // 2
    draw.text((x_offset, y_offset), emoji, font=font, fill=(0, 0, 0))

    return emoji_img

# Loop to add each emoji to the image grid
for i, emoji in enumerate(emoji_chars):
    row = i // cols
    col = i % cols
    x = col * emoji_size
    y = row * emoji_size
    emoji_img = draw_emoji(emoji, x, y)  # Create the emoji image
    img.paste(emoji_img, (x, y))  # Paste the emoji image into the grid

# Save and show the image
img.save("emojis.png")
img.show()
