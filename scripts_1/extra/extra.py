# WRITE TILL 200 TO TXT FILE
# with open("check.txt", "w") as file_object:
#     for i in range(1,34):
#         file_object.write(f"{i}.jpg \n")


# RENAME ALL IMG NAMES IN FOLDER
# import os

# folder_path = r"D:\programming\projects\ai_proje\dlp3\test_photo\check"

# files = os.listdir(folder_path)
# files.sort()  # sort so order is stable

# # Step 1 — rename to temp files
# temp_names = []
# for i, filename in enumerate(files):
#     ext = os.path.splitext(filename)[1]
#     # ext = '.png'
#     temp_name = f"_{i}{ext}"

#     old = os.path.join(folder_path, filename)
#     new = os.path.join(folder_path, temp_name)

#     os.rename(old, new)
#     temp_names.append(temp_name)

# # Step 2 — rename temp files to final names
# for i, temp_name in enumerate(temp_names, start=1):
#     ext = os.path.splitext(temp_name)[1]
#     old = os.path.join(folder_path, temp_name)
#     new = os.path.join(folder_path, f"{i}{ext}")
#     os.rename(old, new)

# print("Renaming complete!")

#  CHANGE PNG TO 
# from PIL import Image
# import os

# input_folder = r"D:\programming\projects\ai_proje\dlp3\test_photo\models_guys\sudipta_labels"
# output_folder = "output_images"

# os.makedirs(output_folder, exist_ok=True)

# for filename in os.listdir(input_folder):
#     print('hello')
#     if filename.endswith(".png"):
#         img_path = os.path.join(input_folder, filename)
#         img = Image.open(img_path).convert("RGBA")

#         # Create white background
#         white_bg = Image.new("RGB", img.size, (255, 255, 255))
#         white_bg.paste(img, mask=img.split()[3])  # use alpha channel as mask

#         output_path = os.path.join(output_folder, filename.replace(".png", ".jpg"))
#         white_bg.save(output_path, "PNG", quality=95)

# print("Done! All images processed.")



# CROP IMG if width 1936
# from PIL import Image
# import os

# input_folder = r"D:\programming\projects\ai_proje\dlp3\test_photo\all"        # folder where your images are
# output_folder = "output_images"      # folder to save processed images

# os.makedirs(output_folder, exist_ok=True)

# TARGET_WIDTH = 1936
# CROP_WIDTH = 1900
# CROP_height = 1880

# for filename in os.listdir(input_folder):
#     if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
#         path = os.path.join(input_folder, filename)
#         img = Image.open(path)
#         width, height = img.size

#         # If width == 1936, crop to 1900
#         if height == TARGET_WIDTH:
#             # Crop from left: keep left 1900px, remove right 36px
#             cropped = img.crop((0, 0, width, CROP_height))
#             save_path = os.path.join(output_folder, filename)
#             cropped.save(save_path)
#             print(f"✔ Cropped {filename} → {CROP_height}px wide")
#         else:
#             # Just copy image unchanged
#             # img.save(os.path.join(output_folder, filename))
#             print(f"Skipped {filename}: width = {width}")

