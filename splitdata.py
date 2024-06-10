import os
import shutil
from sklearn.model_selection import train_test_split

# List of common fruits based on the dataset provided
common_fruits = [
    "apricot", "rambutan", "pineapple", "plum", "elderberry", "apple", "grapefruit",
    "dragonfruit", "raspberry", "mangosteen", "coconut", "guava", "durian", "olive",
    "black_berry", "fig", "gooseberry", "avocado", "mock_strawberry", "jackfruit",
    "longan", "passion_fruit", "mango", "cranberry", "grape", "papaya", "banana",
    "pomegranate", "cashew", "eggplant", "jalapeno", "brazil_nut", "sea_buckthorn",
    "prikly_pear", "feijoa", "kumquat", "quince", "medlar", "emblic", "jambul",
    "mabolo", "cluster_fig", "abiu", "custard_apple", "kiwi", "orange", "watermelon",
    "lemon", "lychee", "pear", "carambola", "cherry", "blueberry"
]

# Paths to the original, additional, and new datasets
original_dataset_path = 'Fruit/train'
additional_dataset_path = 'my_data'
third_dataset_path = 'Fruits Classification/train'
new_dataset_path = 'Newdata'

# Create the new dataset directories for train, valid, and test if they don't exist
for split in ['train', 'valid', 'test']:
    split_path = os.path.join(new_dataset_path, split)
    if not os.path.exists(split_path):
        os.makedirs(split_path)

# Merge 'plumcot' and 'yellow_plum' into 'plum'
plum_path = os.path.join(original_dataset_path, 'plum')
plumcot_path = os.path.join(original_dataset_path, 'plumcot')
yellow_plum_path = os.path.join(original_dataset_path, 'yellow_plum')

if not os.path.exists(plum_path):
    os.makedirs(plum_path)

for source_path in [plumcot_path, yellow_plum_path]:
    if os.path.exists(source_path):
        for img in os.listdir(source_path):
            if img.endswith(('.png', '.jpg', '.jpeg')):
                shutil.copy(os.path.join(source_path, img), plum_path)

# Helper function to split data and copy files
def split_and_copy(fruit, fruit_paths):
    images = []
    for fruit_path in fruit_paths:
        if os.path.exists(fruit_path):
            images.extend([os.path.join(fruit_path, img) for img in os.listdir(fruit_path) if img.endswith(('.png', '.jpg', '.jpeg'))])

    if not images:
        print(f"No images found for {fruit}, skipping.")
        return

    train_images, temp_images = train_test_split(images, test_size=0.05)  # 95% for training
    valid_images, test_images = train_test_split(temp_images, test_size=0.4)  # 2% for validation, 3% for testing

    for img_set, split in [(train_images, 'train'), (valid_images, 'valid'), (test_images, 'test')]:
        split_path = os.path.join(new_dataset_path, split, fruit)
        if not os.path.exists(split_path):
            os.makedirs(split_path)
        for img in img_set:
            shutil.copy(img, split_path)

# Filter and copy images of common fruits to the new dataset
for fruit in common_fruits:
    fruit_paths = [
        os.path.join(original_dataset_path, fruit),
        os.path.join(additional_dataset_path, fruit),
        os.path.join(third_dataset_path, fruit)
    ]
    split_and_copy(fruit, fruit_paths)

print("New dataset created with common fruits, split into train, valid, and test.")
